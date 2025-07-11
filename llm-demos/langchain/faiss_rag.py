"""
Advanced FAISS-based Retrieval-Augmented Generation (RAG) System.

This module provides an enterprise-grade RAG implementation with proper error handling,
configuration management, caching, and performance optimization.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
from langchain.chat_models import init_chat_model
from langchain_community.docstore import InMemoryDocstore
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "text-embedding-3-large"
    dimension: int = 3072  # Known dimension for text-embedding-3-large
    batch_size: int = 100
    
    def __post_init__(self):
        """Validate configuration."""
        if self.dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")


@dataclass
class TextSplitterConfig:
    """Configuration for text splitting."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", " ", ""])
    
    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("Chunk overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")


@dataclass
class LLMConfig:
    """Configuration for language model."""
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.1
    max_tokens: Optional[int] = 1000


@dataclass
class RAGConfig:
    """Main configuration for RAG system."""
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    text_splitter_config: TextSplitterConfig = field(default_factory=TextSplitterConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    vector_store_path: str = "faiss_store"
    similarity_threshold: float = 0.7
    max_results: int = 5
    cache_embeddings: bool = True


class DocumentProcessor:
    """Advanced document processor with error handling and optimization."""
    
    def __init__(self, config: TextSplitterConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
            is_separator_regex=False
        )
    
    def load_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load and process a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of processed documents
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF processing fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        if not file_path.suffix.lower() == '.pdf':
            raise ValueError(f"File must be a PDF: {file_path}")
        
        try:
            logger.info(f"Loading PDF: {file_path}")
            loader = PyPDFLoader(
                file_path=str(file_path),
                pages_delimiter="\n\f"
            )
            
            docs = list(loader.lazy_load())
            
            if not docs:
                raise ValueError(f"No content extracted from PDF: {file_path}")
            
            logger.info(f"Successfully loaded {len(docs)} pages from PDF")
            return docs
            
        except Exception as e:
            logger.error(f"Failed to load PDF {file_path}: {e}")
            raise ValueError(f"PDF processing failed: {e}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        if not documents:
            logger.warning("No documents provided for splitting")
            return []
        
        try:
            start_time = time.time()
            all_splits = self.text_splitter.split_documents(documents)
            processing_time = time.time() - start_time
            
            logger.info(f"Split {len(documents)} documents into {len(all_splits)} chunks in {processing_time:.2f}s")
            return all_splits
            
        except Exception as e:
            logger.error(f"Document splitting failed: {e}")
            raise
    
    def get_document_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Get statistics about the documents."""
        if not documents:
            return {"total_docs": 0, "total_chars": 0, "avg_chars": 0}
        
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chars = total_chars / len(documents)
        
        return {
            "total_docs": len(documents),
            "total_chars": total_chars,
            "avg_chars": round(avg_chars, 2),
            "sources": list(set(doc.metadata.get('source', 'Unknown') for doc in documents))
        }


class VectorStoreManager:
    """Advanced vector store manager with caching and optimization."""
    
    def __init__(self, config: EmbeddingConfig, store_path: str):
        self.config = config
        self.store_path = Path(store_path)
        self.embeddings = self._initialize_embeddings()
        self.vector_store: Optional[FAISS] = None
    
    def _initialize_embeddings(self) -> OpenAIEmbeddings:
        """Initialize the embedding model."""
        try:
            embeddings = OpenAIEmbeddings(
                model=self.config.model_name,
                chunk_size=self.config.batch_size
            )
            
            # Test embedding to verify configuration
            test_embedding = embeddings.embed_query("test")
            actual_dimension = len(test_embedding)
            
            if actual_dimension != self.config.dimension:
                logger.warning(f"Expected dimension {self.config.dimension}, got {actual_dimension}")
                self.config.dimension = actual_dimension
            
            logger.info(f"Embeddings initialized: {self.config.model_name} (dim: {actual_dimension})")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        try:
            start_time = time.time()
            logger.info(f"Creating vector store from {len(documents)} documents")
            
            # Create FAISS index
            index = faiss.IndexFlatL2(self.config.dimension)
            
            # Initialize vector store
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            
            # Add documents in batches for better performance
            batch_size = self.config.batch_size
            document_ids = []
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = vector_store.add_documents(batch)
                document_ids.extend(batch_ids)
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            processing_time = time.time() - start_time
            logger.info(f"Vector store created successfully in {processing_time:.2f}s")
            
            self.vector_store = vector_store
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def save_vector_store(self, vector_store: FAISS) -> None:
        """Save vector store to disk."""
        try:
            self.store_path.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(str(self.store_path))
            logger.info(f"Vector store saved to {self.store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load_vector_store(self) -> FAISS:
        """Load vector store from disk."""
        if not self.store_path.exists():
            raise FileNotFoundError(f"Vector store not found: {self.store_path}")
        
        try:
            vector_store = FAISS.load_local(
                str(self.store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {self.store_path}")
            self.vector_store = vector_store
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise


class AdvancedRAGSystem:
    """Enterprise-grade RAG system with comprehensive features."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.document_processor = DocumentProcessor(config.text_splitter_config)
        self.vector_manager = VectorStoreManager(
            config.embedding_config,
            config.vector_store_path
        )
        self.llm = self._initialize_llm()
        self.vector_store: Optional[FAISS] = None
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the language model."""
        try:
            llm = init_chat_model(
                self.config.llm_config.model_name,
                model_provider=self.config.llm_config.provider,
                temperature=self.config.llm_config.temperature,
                max_tokens=self.config.llm_config.max_tokens
            )
            logger.info(f"LLM initialized: {self.config.llm_config.model_name}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def ingest_documents(self, pdf_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """
        Ingest multiple PDF documents into the vector store.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        all_documents = []
        stats = {"successful_files": 0, "failed_files": 0, "total_chunks": 0}
        
        for pdf_path in pdf_paths:
            try:
                documents = self.document_processor.load_pdf(pdf_path)
                chunks = self.document_processor.split_documents(documents)
                all_documents.extend(chunks)
                stats["successful_files"] += 1
                logger.info(f"Successfully processed: {pdf_path}")
            except Exception as e:
                stats["failed_files"] += 1
                logger.error(f"Failed to process {pdf_path}: {e}")
        
        if all_documents:
            self.vector_store = self.vector_manager.create_vector_store(all_documents)
            self.vector_manager.save_vector_store(self.vector_store)
            stats["total_chunks"] = len(all_documents)
        
        total_time = time.time() - start_time
        stats["processing_time"] = round(total_time, 2)
        
        logger.info(f"Document ingestion completed: {stats}")
        return stats
    
    def load_existing_store(self) -> None:
        """Load an existing vector store."""
        self.vector_store = self.vector_manager.load_vector_store()
    
    def search_similar_documents(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call ingest_documents() or load_existing_store() first.")
        
        k = k or self.config.max_results
        score_threshold = score_threshold or self.config.similarity_threshold
        
        try:
            start_time = time.time()
            results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
            search_time = time.time() - start_time
            
            # Filter by score threshold
            filtered_results = [
                (doc, score) for doc, score in results
                if score >= score_threshold
            ]
            
            logger.info(f"Found {len(filtered_results)}/{len(results)} relevant documents in {search_time:.3f}s")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise
    
    def generate_answer(
        self,
        question: str,
        context_documents: List[Tuple[Document, float]],
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer using retrieved context.
        
        Args:
            question: User question
            context_documents: Retrieved documents with scores
            custom_prompt: Custom prompt template
            
        Returns:
            Dictionary with answer and metadata
        """
        if not context_documents:
            return {
                "answer": "I don't have enough relevant information to answer this question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Build context from documents
        context_parts = []
        sources = []
        
        for doc, score in context_documents:
            context_parts.append(f"Content: {doc.page_content}")
            source_info = {
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "score": round(score, 3)
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Use custom prompt or default
        prompt_template = custom_prompt or """
You are an expert assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question accurately and comprehensively.

Guidelines:
- Base your answer strictly on the provided context
- If you don't know the answer, clearly state that you don't have enough information
- Always cite your sources with specific page numbers when available
- Provide a confidence level for your answer
- Be concise but thorough

Question: {question}

Context:
{context}

Answer:"""
        
        try:
            start_time = time.time()
            prompt = prompt_template.format(question=question, context=context)
            response = self.llm.invoke(prompt)
            generation_time = time.time() - start_time
            
            # Calculate average confidence based on similarity scores
            avg_confidence = sum(score for _, score in context_documents) / len(context_documents)
            
            result = {
                "answer": response.content,
                "sources": sources,
                "confidence": round(avg_confidence, 3),
                "generation_time": round(generation_time, 3),
                "context_length": len(context)
            }
            
            logger.info(f"Answer generated in {generation_time:.3f}s with confidence {avg_confidence:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise
    
    def query(
        self,
        question: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG query: search + generate answer.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score
            custom_prompt: Custom prompt template
            
        Returns:
            Complete answer with metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Search for relevant documents
        context_documents = self.search_similar_documents(
            question, k=k, score_threshold=score_threshold
        )
        
        # Generate answer
        result = self.generate_answer(question, context_documents, custom_prompt)
        result["question"] = question
        result["retrieved_docs"] = len(context_documents)
        
        return result


def main():
    """Demonstration of the advanced RAG system."""
    try:
        # Configuration
        config = RAGConfig()
        
        # Initialize RAG system
        rag = AdvancedRAGSystem(config)
        
        # PDF files to process
        pdf_files = ["pdfs/brochure.pdf"]
        
        # Check if vector store exists
        if Path(config.vector_store_path).exists():
            logger.info("Loading existing vector store...")
            rag.load_existing_store()
        else:
            logger.info("Creating new vector store...")
            stats = rag.ingest_documents(pdf_files)
            print(f"📊 Ingestion Stats: {json.dumps(stats, indent=2)}")
        
        # Example queries
        queries = [
            "Does TRD Pro have moonroof?",
            "What are the key features of the vehicle?",
            "What is the price information mentioned?"
        ]
        
        for query in queries:
            print(f"\n{'='*60}")
            print(f"🔍 Query: {query}")
            print('='*60)
            
            result = rag.query(query)
            
            print(f"📝 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']}")
            print(f"📚 Sources: {len(result['sources'])} documents")
            print(f"⏱️ Generation Time: {result['generation_time']}s")
            
            if result['sources']:
                print("\n📖 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source']} (Page: {source['page']}, Score: {source['score']})")
        
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    main()