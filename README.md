<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=7928CA&height=200&section=header&text=%20Document%20Query%20System&fontSize=45&fontColor=ffffff&animation=fadeIn&fontAlignY=38" />
</p>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&pause=1000&center=true&vCenter=true&width=700&lines=⚡+LLM-Powered+Document+Query+%26+Retrieval;🚀+Asynchronous+Document+Ingestion;🔍+Vector+Search+%7C+Semantic+QA;🛠️+Built+with+FastAPI+%26+Transformers" alt="Typing SVG" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LLM-Powered-purple?logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square" />
</p>

## ✨ Overview  

The **Document Query System** is an **LLM-powered intelligent document QA pipeline**.  
It ingests enterprise documents (**PDF, DOCX, EML, scanned files**) ➝ processes them into **semantic chunks** ➝ vectorizes the content ➝ enables **semantic search + question answering** using embeddings and Pinecone.  

🔎 Think of it as **ChatGPT for your enterprise documents** — reliable, scalable, and secure.  

## 🔥 Features  

✅ **Asynchronous Document Ingestion** with validation (size, type, hash caching)  
✅ **Advanced Text Extraction** (PDFs, DOCX, Emails, OCR for scans)  
✅ **Smart Chunking** → preserves semantic coherence (sections, paragraphs, tables)  
✅ **Embeddings & Vector Search** via Hugging Face + Pinecone  
✅ **Secure API** with Bearer authentication  
✅ **Fallback Search Strategies** → Vector search ➝ Keyword fallback  
✅ **Scalable & Concurrent** ingestion + query processing  


## 🛠️ Tech Stack  

<table align="center">
  <tr>
    <th style="text-align:center">Category</th>
    <th style="text-align:center">Technologies</th>
    <th style="text-align:center">Description</th>
  </tr>
  <tr>
    <td align="center"><b>Backend</b></td>
    <td align="center">
      <img src="https://skillicons.dev/icons?i=python" height="40"/> 
      <img src="https://skillicons.dev/icons?i=fastapi" height="40"/>
    </td>
    <td align="center">API development & async backend with FastAPI</td>
  </tr>
  <tr>
    <td align="center"><b>ML & AI</b></td>
    <td align="center">
      <img src="https://skillicons.dev/icons?i=pytorch" height="40"/> 
      <img src="https://skillicons.dev/icons?i=huggingface" height="40"/>
    </td>
    <td align="center">Embeddings, Transformers & LLM integration</td>
  </tr>
  <tr>
    <td align="center"><b>Vector DB</b></td>
    <td align="center">
      <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-0A192F?style=for-the-badge&logo=pinecone&logoColor=white" height="28"/>
    </td>
    <td align="center">Semantic search & vector storage</td>
  </tr>
  <tr>
    <td align="center"><b>Tools & Testing</b></td>
    <td align="center">
      <img src="https://skillicons.dev/icons?i=postman" height="40"/> 
      <img src="https://skillicons.dev/icons?i=docker" height="40"/>
    </td>
    <td align="center">API testing, containerization & dev tools</td>
  </tr>
</table>

## 🚀 Getting Started  

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/document-query-system.git
cd document-query-system
```
### 2️⃣ Install dependencies
```sh
pip install -r requirements.txt
``` 

### 3️⃣ Setup environment
Create a .env file with your API keys:
```sh
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=gcp-starter
```
### 4️⃣ Run the FastAPI server
```sh
uvicorn src.main:app --reload --port 8000
```
📌 Swagger Docs available at 👉 http://localhost:8000/docs

## 📡 API Endpoints

🔑 Authentication

All endpoints require Bearer Token:
```sh
Authorization: Bearer <your_token>
```
### ▶️ Run Query
```sh
POST /api/v1/hackrx/run

{
  "documents": ["policy.pdf", "contract.docx"],
  "questions": ["What is the grace period?", "Does it cover maternity?"]
}

```
Response Example:
```sh
{
  "answers": [
    {
      "question": "What is the grace period?",
      "summary": "Found 3 relevant sections...",
      "search_method": "vector_search"
    }
  ],
  "processed_documents": [
    {
      "document_id": "policy",
      "chunks_count": 120,
      "total_characters": 45000
    }
  ]
}
```
## 🧪 Testing

Run the included testing scripts:

```sh
# Process the HackRx policy document
python src/scripts/main.py

# Run ingestion test
python src/services/document_processing/test.py
```

Vector Engine examples in src/services/vector_engine/readme.md:

1. End-to-end pipeline
2. Vectorization testing
3. Semantic search queries

## 📂 Project Structure

```sh
hackrx60-document-query-system/
│── src/
│   ├── api/                 # API endpoints (FastAPI)
│   ├── auth/                # Authentication
│   ├── schemas/             # Pydantic models
│   ├── scripts/             # Example scripts
│   ├── services/
│   │   ├── document_processing/  # Ingestion & text extraction
│   │   └── vector_engine/        # Embeddings + Pinecone
│   └── main.py              # FastAPI app entrypoint
│── requirements.txt
│── README.md

```

### 🤝 Contributing

We ❤️ contributions!
Fork, raise an issue, or open a PR 🚀

### 📜 License

This project is licensed under the MIT License.

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&reversal=true&color=7928CA&height=120&section=footer&text=Made%20with%20%E2%9D%A4%EF%B8%8F%20for%20HackRx6.0&fontSize=28&fontColor=ffffff&animation=fadeIn&fontAlignY=70" />
</p>


