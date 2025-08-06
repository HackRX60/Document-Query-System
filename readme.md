# Document Query Processing

This project allows you to query documents (PDFs) using natural language questions. It extracts text from a PDF, chunks and embeds the content, builds a vector index, and uses Gemini to answer your questions.

## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yogifly/Document-Query-Processing.git
cd Document-Query-Processing
```

### 2. Create and Activate a Virtual Environment (Recommended)

```sh
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

If `requirements.txt` is missing, install these manually:

```sh
pip install fastapi uvicorn python-dotenv requests PyMuPDF nltk sentence-transformers google-generativeai numpy
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root and add your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5. Download NLTK Data

The code will automatically download the required NLTK data (`punkt`) on first run.

### 6. Run the API Server

```sh
uvicorn main:app --reload
```

### 7. Query the API

Use Postman or `curl` to send a POST request to:

```
POST http://localhost:8000/ask
Content-Type: application/json
```

Example body:

```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
  "questions": [
    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
    "What is the waiting period for pre-existing diseases (PED) to be covered?"
  ]
}
```
