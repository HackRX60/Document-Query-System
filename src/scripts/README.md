# Integrated Pipeline Usage Guide

This guide explains how to use the integrated pipeline that combines document processing with advanced query processing.

## Overview

The integration connects your `query_processing` module with the existing vector embedding pipeline in `scripts/main.py`. The system now supports:

1. **Document Processing**: PDF → Text Extraction → Vector Embeddings → Pinecone Storage
2. **Advanced Query Processing**: Natural Language → Semantic Search → AI-Powered Answers

## Files Created

- `integrated_pipeline.py` - Complete integrated pipeline with class-based approach
- Updated `main.py` - Enhanced with query processing capabilities

## Quick Start

### 1. Environment Setup
Ensure your `.env` file has all required keys:
```bash
# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=gcp-starter

# Google Gemini Configuration
GEMINI_API_KEY=your_gemini_key
```

### 2. Run Complete Workflow
```bash
cd src/scripts
python main.py
```

This will:
1. Process the policy document
2. Create vector embeddings in Pinecone
3. Test with advanced query processing using Gemini

### 3. Use Individual Components

#### Process a Document
```python
import asyncio
from integrated_pipeline import process_document_with_query_support

async def main():
    result = await process_document_with_query_support("your_document_url.pdf")
    print(result)

asyncio.run(main())
```

#### Query Processed Documents
```python
import asyncio
from integrated_pipeline import query_processed_documents

async def main():
    result = await query_processed_documents("What is the grace period for premium payment?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")

asyncio.run(main())
```

## Usage Examples

### Using the IntegratedPipeline Class
```python
import asyncio
from integrated_pipeline import IntegratedPipeline

async def example():
    pipeline = IntegratedPipeline()
    await pipeline.initialize()
    
    # Process document
    doc_result = await pipeline.process_document("document_url.pdf")
    
    # Query the document
    query_result = await pipeline.process_query("What does this policy cover?")
    
    # Process multiple queries
    queries = ["Question 1", "Question 2"]
    batch_results = await pipeline.process_queries_batch(queries)

asyncio.run(example())
```

### Using the Enhanced main.py
```python
# Run complete workflow
python main.py

# Or use individual functions:
# python -c "import asyncio; from main import process_hackrx_policy_document; asyncio.run(process_hackrx_policy_document())"
```

## Features Comparison

| Feature | Basic Vectorizer | Gemini Query Processor |
|---------|------------------|----------------------|
| Natural Language Understanding | ❌ | ✅ |
| Confidence Scoring | ❌ | ✅ |
| Reasoning Explanation | ❌ | ✅ |
| Contextual Answers | ❌ | ✅ |
| Entity Recognition | ❌ | ✅ |
| Query Intent Analysis | ❌ | ✅ |

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Ensure `GEMINI_API_KEY` is set in `.env`
   - Ensure `PINECONE_API_KEY` is set in `.env`

2. **Import Errors**
   - Install required packages: `pip install -r requirements.txt`
   - Check Python path: `export PYTHONPATH="${PYTHONPATH}:/path/to/project"`

3. **Pinecone Index Issues**
   - Verify index name matches: "hackrx-insurance-docs"
   - Check region: "gcp-starter"

### Testing Individual Components

```bash
# Test document processing only
python -c "import asyncio; from services.vector_engine.pipeline import complete_pipeline_with_free_vectorization; from services.vector_engine.config import EmbeddingConfig, PineconeConfig; import os; asyncio.run(complete_pipeline_with_free_vectorization('test.pdf', EmbeddingConfig(), PineconeConfig(os.getenv('PINECONE_API_KEY'), 'gcp-starter', 'hackrx-insurance-docs')))"

# Test query processing only
python -c "import asyncio; from services.query_processor.processor import GeminiQueryProcessor; import os; processor = GeminiQueryProcessor(os.getenv('GEMINI_API_KEY'), os.getenv('PINECONE_API_KEY'), 'gcp-starter', 'hackrx-insurance-docs'); result = asyncio.run(processor.process_query('test query')); print(result)"
```

## Next Steps

1. **Customize Domain Context**: Modify the `domain_context` parameter in query processing
2. **Add Custom Queries**: Extend the test queries list
3. **Batch Processing**: Use `process_queries_batch()` for multiple queries
4. **Integration**: Use the `IntegratedPipeline` class in your applications
