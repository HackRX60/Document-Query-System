# import os
# import asyncio
# from dotenv import load_dotenv
# load_dotenv()
# from services.vector_engine.config import EmbeddingConfig, PineconeConfig
# from services.vector_engine.vectorizer import DocumentVectorizer
# from services.vector_engine.pipeline import complete_pipeline_with_free_vectorization
# from services.query_processor.processor import GeminiQueryProcessor
# from services.query_processor.query_config import QueryProcessorConfig

# class IntegratedPipeline:
#     """Integrated pipeline combining document processing with advanced query processing"""
    
#     def __init__(self):
#         # Initialize configurations
#         self.embedding_config = EmbeddingConfig()
#         self.embedding_config.model_name = "BAAI/bge-large-en-v1.5"
#         self.embedding_config.dimensions = 1024
#         self.embedding_config.batch_size = 32
        
#         self.pinecone_config = PineconeConfig(
#             api_key=os.getenv("PINECONE_API_KEY"),
#             region=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
#             index_name="hackrx-insurance-docs"
#         )
        
#         # Initialize query processor config
#         self.query_config = QueryProcessorConfig()
        
#         # Initialize components
#         self.vectorizer = None
#         self.query_processor = None
    
#     async def initialize(self):
#         """Initialize all components"""
#         self.vectorizer = DocumentVectorizer(self.embedding_config, self.pinecone_config)
#         await self.vectorizer.initialize()
        
#         self.query_processor = GeminiQueryProcessor(
#             gemini_api_key=os.getenv("GEMINI_API_KEY"),
#             pinecone_api_key=os.getenv("PINECONE_API_KEY"),
#             pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
#             index_name="hackrx-insurance-docs"
#         )
        
#         print("Integrated pipeline initialized successfully!")
    
#     async def process_document(self, document_url: str) -> dict:
#         """Process a document and create vector embeddings"""
#         try:
#             results = await complete_pipeline_with_free_vectorization(
#                 document_url,
#                 self.embedding_config,
#                 self.pinecone_config
#             )
#             return results
#         except Exception as e:
#             print(f"Error processing document: {e}")
#             return {"error": str(e)}
    
#     async def process_query(self, query: str, domain_context: str = "insurance") -> dict:
#         """Process a natural language query using advanced query processing"""
#         try:
#             result = await self.query_processor.process_query(query, domain_context)
#             return {
#                 "query": query,
#                 "answer": result.answer,
#                 "confidence": result.confidence,
#                 "reasoning": result.reasoning,
#                 "sources": result.sources,
#                 "query_id": result.query_id
#             }
#         except Exception as e:
#             print(f"Error processing query: {e}")
#             return {"error": str(e)}
    
#     async def process_queries_batch(self, queries: list[str], domain_context: str = "insurance") -> list[dict]:
#         """Process multiple queries in batch"""
#         results = []
#         for query in queries:
#             result = await self.process_query(query, domain_context)
#             results.append(result)
#         return results
    
#     async def test_integrated_pipeline(self):
#         """Test the complete integrated pipeline"""
#         print("Testing integrated pipeline...")
        
#         # Test document processing
#         document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        
#         print("Processing document...")
#         doc_results = await self.process_document(document_url)
#         print(f"Document processing results: {doc_results}")
        
#         # Test query processing
#         test_queries = [
#             "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
#             "What is the waiting period for pre-existing diseases (PED) to be covered?",
#             "Does this policy cover maternity expenses, and what are the conditions?",
#             "What is the waiting period for cataract surgery?",
#             "Are the medical expenses for an organ donor covered under this policy?"
#         ]
        
#         print("Processing queries...")
#         query_results = await self.process_queries_batch(test_queries)
        
#         for i, result in enumerate(query_results):
#             print(f"\n--- Query {i+1}: {result.get('query', 'N/A')} ---")
#             print(f"Answer: {result.get('answer', 'N/A')}")
#             print(f"Confidence: {result.get('confidence', 0)}")
#             print(f"Reasoning: {result.get('reasoning', 'N/A')}")
        
#         return {
#             "document_results": doc_results,
#             "query_results": query_results
#         }

# # Standalone functions for easy integration
# async def process_document_with_query_support(document_url: str) -> dict:
#     """Process document and prepare for query processing"""
#     pipeline = IntegratedPipeline()
#     await pipeline.initialize()
#     return await pipeline.process_document(document_url)

# async def query_processed_documents(query: str, domain_context: str = "insurance") -> dict:
#     """Query documents that have been processed and stored in Pinecone"""
#     pipeline = IntegratedPipeline()
#     await pipeline.initialize()
#     return await pipeline.process_query(query, domain_context)

# # Main execution for testing
# async def main():
#     """Main execution function"""
#     pipeline = IntegratedPipeline()
#     await pipeline.initialize()
#     results = await pipeline.test_integrated_pipeline()
#     return results

# if __name__ == "__main__":
#     asyncio.run(main())
