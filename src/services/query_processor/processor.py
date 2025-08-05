import os
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Structured query result with confidence and sources"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning: str
    query_id: str

@dataclass
class RetrievedContext:
    """Retrieved document context with metadata"""
    content: str
    score: float
    metadata: Dict[str, Any]
    document_id: str
    page_number: int = None

class GeminiQueryProcessor:
    """
    Query Processing Engine using Google Gemini for LLM operations
    and Pinecone for vector search
    """
    
    def __init__(self, 
                 gemini_api_key: str,
                 pinecone_api_key: str,
                 pinecone_environment: str,
                 index_name: str,
                 embedding_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Initialize the query processor
        
        Args:
            gemini_api_key: Google Gemini API key
            pinecone_api_key: Pinecone API key
            pinecone_environment: Pinecone environment
            index_name: Pinecone index name
            embedding_model: Sentence transformer model for embeddings
        """
        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(index_name)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        logger.info("Query processor initialized successfully")
    
    def understand_query(self, query: str, domain_context: str = "insurance") -> Dict[str, Any]:
        """
        Use Gemini to understand and structure the natural language query
        
        Args:
            query: Natural language query
            domain_context: Domain context (insurance, legal, HR, compliance)
            
        Returns:
            Structured query understanding
        """
        prompt = f"""
        You are an expert in {domain_context} document analysis. Analyze the following query and extract key information.

        Query: "{query}"

        Please provide a structured analysis in JSON format with the following fields:
        1. "intent": The main intent of the query (e.g., "coverage_check", "eligibility", "claim_process", "conditions")
        2. "entities": Key entities mentioned (e.g., medical procedures, policy terms, dates, amounts)
        3. "query_type": Type of question ("yes_no", "factual", "conditional", "comparison")
        4. "keywords": Important keywords for semantic search
        5. "domain_specific_terms": Domain-specific terminology identified
        6. "complexity_level": Scale 1-5 (1=simple factual, 5=complex conditional reasoning)

        Return only valid JSON without any markdown formatting.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            query_understanding = json.loads(response.text)
            
            logger.info(f"Query understood: {query_understanding['intent']}")
            return query_understanding
            
        except Exception as e:
            logger.error(f"Error understanding query: {e}")
            # Fallback simple understanding
            return {
                "intent": "general_inquiry",
                "entities": [],
                "query_type": "factual",
                "keywords": query.split(),
                "domain_specific_terms": [],
                "complexity_level": 3
            }
    
    def semantic_search(self, query: str, top_k: int = 10, 
                       filter_metadata: Dict[str, Any] = None) -> List[RetrievedContext]:
        """
        Perform semantic search using Pinecone vector database
        
        Args:
            query: Search query
            top_k: Number of top results to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of retrieved contexts with scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            retrieved_contexts = []
            for match in search_results['matches']:
                context = RetrievedContext(
                    content=match['metadata'].get('text', ''),
                    score=match['score'],
                    metadata=match['metadata'],
                    document_id=match['metadata'].get('document_id', ''),
                    page_number=match['metadata'].get('page_number')
                )
                retrieved_contexts.append(context)
            
            logger.info(f"Retrieved {len(retrieved_contexts)} contexts")
            return retrieved_contexts
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def rank_contexts(self, query: str, contexts: List[RetrievedContext], 
                     query_understanding: Dict[str, Any]) -> List[RetrievedContext]:
        """
        Re-rank retrieved contexts based on relevance and query understanding
        
        Args:
            query: Original query
            contexts: Retrieved contexts
            query_understanding: Structured query analysis
            
        Returns:
            Re-ranked contexts
        """
        # Calculate relevance scores based on multiple factors
        for context in contexts:
            relevance_score = context.score  # Base similarity score
            
            # Boost score if context contains query entities
            entities = query_understanding.get('entities', [])
            for entity in entities:
                # Handle both string and dict entity formats
                entity_text = entity
                if isinstance(entity, dict):
                    entity_text = entity.get('text', str(entity))
                elif not isinstance(entity, str):
                    entity_text = str(entity)
                
                if entity_text.lower() in context.content.lower():
                    relevance_score *= 1.1
            
            # Boost score for domain-specific terms
            domain_terms = query_understanding.get('domain_specific_terms', [])
            for term in domain_terms:
                # Handle both string and dict term formats
                term_text = term
                if isinstance(term, dict):
                    term_text = term.get('text', str(term))
                elif not isinstance(term, str):
                    term_text = str(term)
                
                if term_text.lower() in context.content.lower():
                    relevance_score *= 1.05
            
            context.score = relevance_score
        
        # Sort by updated relevance score
        ranked_contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        
        logger.info("Contexts re-ranked based on query understanding")
        return ranked_contexts
    
    def generate_answer(self, query: str, contexts: List[RetrievedContext], 
                       query_understanding: Dict[str, Any]) -> QueryResult:
        """
        Generate structured answer using Gemini with retrieved contexts
        
        Args:
            query: Original query
            contexts: Top ranked contexts
            query_understanding: Structured query analysis
            
        Returns:
            Structured query result with reasoning
        """
        # Prepare context for Gemini
        context_text = "\n\n".join([
            f"Context {i+1} (Score: {ctx.score:.3f}):\n{ctx.content}"
            for i, ctx in enumerate(contexts[:5])  # Use top 5 contexts
        ])
        
        prompt = f"""
        You are an expert document analyst. Based on the provided contexts, answer the user's query with precision and provide clear reasoning.

        Query: "{query}"
        Query Type: {query_understanding.get('query_type', 'factual')}
        Intent: {query_understanding.get('intent', 'general')}

        Relevant Document Contexts:
        {context_text}

        Please provide a response in the following JSON format:
        {{
            "answer": "Direct answer to the query",
            "confidence": 0.85,
            "reasoning": "Step-by-step explanation of how you arrived at the answer",
            "sources_used": ["Context 1", "Context 2"],
            "decision_factors": ["factor1", "factor2"],
            "conditions_or_limitations": "Any conditions or limitations to the answer"
        }}

        Guidelines:
        1. Be precise and factual
        2. If information is insufficient, clearly state limitations
        3. For yes/no questions, provide clear binary answers with conditions
        4. Confidence should be 0.0-1.0 based on context quality and completeness
        5. Always explain your reasoning process

        Return only valid JSON without markdown formatting.
        """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            answer_data = json.loads(response.text)
            
            # Create structured result
            result = QueryResult(
                answer=answer_data.get('answer', 'Unable to determine answer'),
                confidence=answer_data.get('confidence', 0.5),
                sources=[{
                    'content': ctx.content[:200] + '...',
                    'score': ctx.score,
                    'document_id': ctx.document_id,
                    'page_number': ctx.page_number
                } for ctx in contexts[:3]],
                reasoning=answer_data.get('reasoning', 'No reasoning provided'),
                query_id=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            logger.info(f"Answer generated with confidence: {result.confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback response
            return QueryResult(
                answer="Unable to process query due to technical error",
                confidence=0.0,
                sources=[],
                reasoning="Technical error occurred during processing",
                query_id=f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def process_query(self, query: str, domain_context: str = "insurance", 
                     top_k: int = 10) -> QueryResult:
        """
        Complete query processing pipeline
        
        Args:
            query: Natural language query
            domain_context: Domain context
            top_k: Number of contexts to retrieve
            
        Returns:
            Complete query result
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Understand the query
        query_understanding = self.understand_query(query, domain_context)
        
        # Step 2: Semantic search
        retrieved_contexts = self.semantic_search(query, top_k)
        
        # Step 3: Rank contexts
        ranked_contexts = self.rank_contexts(query, retrieved_contexts, query_understanding)
        
        # Step 4: Generate answer
        result = self.generate_answer(query, ranked_contexts, query_understanding)
        
        logger.info(f"Query processed successfully with confidence: {result.confidence}")
        return result
    
    def batch_process_queries(self, queries: List[str], 
                             domain_context: str = "insurance") -> List[QueryResult]:
        """
        Process multiple queries in batch
        
        Args:
            queries: List of queries to process
            domain_context: Domain context
            
        Returns:
            List of query results
        """
        results = []
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            result = self.process_query(query, domain_context)
            results.append(result)
        
        return results

# Usage Example and Testing
if __name__ == "__main__":
    # Initialize processor
    processor = GeminiQueryProcessor(
        gemini_api_key="your_gemini_api_key",
        pinecone_api_key="your_pinecone_api_key",
        pinecone_environment="your_pinecone_env",
        index_name="your_index_name"
    )
    
    # Test queries
    test_queries = [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover knee surgery, and what are the conditions?"
    ]
    
    # Process queries
    results = processor.batch_process_queries(test_queries)
    
    # Display results
    for i, result in enumerate(results):
        print(f"\n--- Query {i+1} Results ---")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Sources: {len(result.sources)} contexts used")