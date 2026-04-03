import streamlit as st
import tempfile
import os
import random
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from trulens.core import TruSession
from trulens.apps.llamaindex import TruLlama
from trulens.providers.litellm import LiteLLM
from trulens.core import Feedback
import numpy as np

os.environ["TRULENS_OTEL_TRACING"] = "0"  # disable OTEL mode so old selectors work

# FOR DIFFERENT API keys
GEMINI_API_KEYS = [
    "AIzaSyDeyu7vJ4bzlhfqcOvOF-6EHhhRZzUyIdA",
    "AIzaSyAi5k6l25tR0y5sgz5keFcsAelT6iCJeJc",
    "AIzaSyA6fk2H19reCPRsGZV9draZ91EhRmUVyhc",
]

class APIKeyManager: # Manages a pool of API keys with randomization and fallback.
    def __init__(self, keys):
        self.keys = keys.copy()
        random.shuffle(self.keys)      # Randomize order at startup
        self.index = 0                 # Start at the first random key
        self.failed_keys = set()       # Keep track of exhausted keys

    def current_key(self):
        return self.keys[self.index]   # Return the key we are currently using

    def rotate(self):
        """
        Move to the next working key
        Skips any keys already marked as failed
        """
        for _ in range(len(self.keys)):
            self.index = (self.index + 1) % len(self.keys)  # Wrap around
            if self.keys[self.index] not in self.failed_keys:
                print("   Rotated → key #", self.index + 1)
                return self.keys[self.index]

        print("    All keys have failed!")
        return None

    def mark_failed(self, key):
        """
        Mark a key as failed
        Automatically rotates to the next available key
        """
        self.failed_keys.add(key)
        print("     Key marked failed:", key)
        return self.rotate()
    
    def all_failed(self):
        """Check if every key in the pool has been marked as failed."""
        return len(self.failed_keys) >= len(self.keys)


# using session state so the manager can be used when we rerun and keeps track of failed keys
if "key_manager" not in st.session_state:
    st.session_state.key_manager = APIKeyManager(GEMINI_API_KEYS)

def apply_settings(api_key): # Apply llama and gemini settings using the given api key.
    #the settings is the inbuilt fuction for llama and gemini connectivity together.. if u try to do it individually then llama uses open ai
    Settings.llm = Gemini(api_key=api_key, model_name="models/gemini-2.0-flash-lite") #this for gemini and llama
    Settings.embed_model = GeminiEmbedding(api_key=api_key, model_name="models/gemini-embedding-001") # this is for embedding help
    #reducing the amount of chunks taken to embed it
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

# apply settings with the current key at startup
apply_settings(st.session_state.key_manager.current_key())

def query_with_fallback(query_engine, prompt):
    """
    Run a query and automatically rotate to the next key if quota is exceeded.
    Keeps retrying until a working key is found or all keys are exhausted.
    """
    while True:
        try:
            return query_engine.query(prompt)
        except Exception as e:
            error_msg = str(e).lower()
            # check if the error is quota/rate limit related
            if any(word in error_msg for word in ["quota", "rate limit", "exhausted", "429", "resource exhausted"]):
                current_key = st.session_state.key_manager.current_key()
                st.warning("API quota exceeded. Rotating to next key...")
                # mark the current key as failed and rotate
                next_key = st.session_state.key_manager.mark_failed(current_key)
                if next_key is None or st.session_state.key_manager.all_failed():
                    # all keys are dead, nothing we can do
                    st.error("All API keys have been exhausted. Please add more keys.")
                    return None
                # apply the new key to llama settings and retry
                apply_settings(next_key)
                st.info(f"Switched to key #{st.session_state.key_manager.index + 1}. Retrying...")
            else:
                # not a quota error, raise it normally
                raise e

# this is the trulens session so it can track and evaluate all queries
tru_session = TruSession()

def get_feedbacks():
    # using gemini itself as the evaluation provider through litellm
    provider = LiteLLM(model_engine="gemini/gemini-2.0-flash-lite")

    # Groundedness: this is the main check - makes sure gemini only answers from the pdf
    f_groundedness = (
        Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on_input_output()
    )

    # Answer relevance: is the answer relevant to the question?
    f_answer_relevance = (
        Feedback(provider.relevance_with_cot_reasons, name="Answer relevance")
        .on_input_output()
    )

    # Context relevance: this checks whether the chunks pulled from vector db were actually useful
    f_context_relevance = (
        Feedback(provider.context_relevance_with_cot_reasons, name="Context relevance")
        .on_input()
        .on(TruLlama.select_source_nodes().node.text)
        .aggregate(np.mean)
    )

    return [f_groundedness, f_answer_relevance, f_context_relevance]


st.title("Med-Buddy")

#the cache is for store the vector db so that it doesn't re-run again
@st.cache_resource

#this has the normal code and tempfile module cuz u need the pathfile... llama reads and gives the bytes to it's hard to access the file
def build_index(file_bytes): #function so that it doesn't re run each time
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name
    documents = SimpleDirectoryReader(input_files=[tmp_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    os.remove(tmp_path)
    query_engine = index.as_query_engine(similarity_top_k=2) #we can add thing of our own in the session state fo query engine is the vector db

    # trulens wraps around the query engine so it can intercept and evaluate every query
    tru_query_engine = TruLlama(
        query_engine,
        app_name="Med-Buddy",
        feedbacks=get_feedbacks(),
    )
    return query_engine, tru_query_engine  # returning both so sidebar can store both


#sidebar things uk
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF to chat with it", type="pdf")
    if uploaded_file is not None:
        query_engine, tru_query_engine = build_index(uploaded_file.read())
        st.session_state.query_engine = query_engine
        st.session_state.tru_query_engine = tru_query_engine

    # shows how many keys are still working so u know when to add more
    st.divider()
    st.subheader("API Key Status")
    km = st.session_state.key_manager
    for i, key in enumerate(km.keys):
        if key in km.failed_keys:
            st.error(f"Key #{i+1} —  exhausted")
        elif i == km.index:
            st.success(f"Key #{i+1} —  active")
        else:
            st.info(f"Key #{i+1} —  standby")


# this is for checking session state..basically a dictionary for role, content and query engine... built-in streamlit things
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#asking the user the query
if prompt := st.chat_input("Ask about your document"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    #made sure we can upload the file
    if "query_engine" not in st.session_state:
        response_text = "Please upload a PDF first before asking questions."
    else:
        # the with block tells trulens to record everything that happens during this query
        with st.session_state.tru_query_engine as recording:
            #answers based on the vector db using gemini - with automatic key rotation on quota errors
            response = query_with_fallback(st.session_state.query_engine, prompt)
            response_text = str(response) if response else "All API keys exhausted. Please add more keys."

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})