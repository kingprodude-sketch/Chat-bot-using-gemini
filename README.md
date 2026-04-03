# Med-Buddy 🏥

A medical document chatbot that lets you upload PDFs and ask questions about them using RAG (Retrieval-Augmented Generation).

## Features
- Upload and query medical PDF documents
- Powered by Google Gemini + LlamaIndex
- Response caching and multi-key API rotation
- TruLens evaluation integration

## Tech Stack
- Python, Streamlit
- LlamaIndex, Google Gemini API
- TruLens for evaluation

## Getting Started

### Prerequisites
- Python 3.9+
- A Google Gemini API key

### Installation
```bash
git clone https://github.com/yourusername/med-buddy.git
cd med-buddy
pip install -r requirements.txt
```

### Setup
Create a `keys.env` file in the project root with your Google Gemini API keys (multiple keys for rotation):
```
GEMINI_API_KEYS=[
    "your_first_api_key_here",
    "your_second_api_key_here",
    "your_third_api_key_here"
]
```

### Run
```bash
streamlit run app.py
```
