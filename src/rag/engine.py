from typing import AsyncIterator, List, Optional

from ..config.settings import settings
from ..core.interfaces import (
    DecomposedQuery,
    Document,
    LLMInterface,
    QueryDecomposerInterface,
    QueryResult,
    StreamingChunk,
    SubQuery,
    VectorStoreInterface,
)
from ..data.loader import DocumentLoader
from ..data.processor import DocumentProcessor
from ..decomposition.query_decomposer import (
    ContextAwareDecomposer,
    SmartQueryDecomposer,
)
from ..llm.github_llm import GitHubLLM
from ..vectorstore.chroma_store import ChromaVectorStore


class AdvancedRAGEngine:
    """RAG engine with query decomposition and deep reasoning"""

    def __init__(
        self,
        llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        query_decomposer: Optional[QueryDecomposerInterface] = None,
        use_context_aware_decomposer: bool = True,
    ):
        self.llm = llm or GitHubLLM()
        self.vector_store = vector_store or ChromaVectorStore()

        # Choose decomposer type
        if use_context_aware_decomposer:
            self.query_decomposer = query_decomposer or ContextAwareDecomposer(
                self.llm
            )
        else:
            self.query_decomposer = query_decomposer or SmartQueryDecomposer(
                self.llm
            )

        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()

    async def ingest_documents(self, directory_path: str) -> int:
        """Ingest documents from directory"""
        documents = await self.document_loader.load_from_directory(
            directory_path
        )
        processed_docs = await self.document_processor.process_documents(
            documents
        )
        doc_ids = await self.vector_store.add_documents(processed_docs)
        return len(doc_ids)

    async def query_with_decomposition(
        self, question: str, k: Optional[int] = None
    ) -> DecomposedQuery:
        """Query with automatic decomposition and reasoning"""
        k = k or settings.retrieval_k

        print(f"🔍 Original Question: {question}")

        # Step 1: Decompose the query
        sub_query_texts = await self.query_decomposer.decompose_query(question)

        if len(sub_query_texts) == 1:
            print("📝 No decomposition needed, using direct RAG")
            # Use regular RAG for simple queries
            result = await self.query(question, k)
            return DecomposedQuery(
                original_question=question,
                sub_queries=[
                    SubQuery(
                        question=question,
                        answer=result.response,
                        source_documents=result.source_documents,
                        metadata=result.metadata,
                    )
                ],
                final_answer=result.response,
                metadata={"decomposition_used": False},
            )

        print(f"🔧 Decomposed into {len(sub_query_texts)} sub-queries:")
        for i, sq in enumerate(sub_query_texts, 1):
            print(f"   {i}. {sq}")

        # Step 2: Answer each sub-query using RAG
        sub_queries = []
        all_source_docs = []

        for sub_query_text in sub_query_texts:
            print(f"\n🔍 Answering: {sub_query_text}")

            # Get relevant documents for this sub-query
            relevant_docs = await self.vector_store.similarity_search(
                sub_query_text, k=k
            )
            context = self._prepare_context(relevant_docs)

            # Generate answer for this sub-query
            prompt = self._create_sub_query_prompt(sub_query_text, context)
            answer = await self.llm.generate(prompt, temperature=0.7)

            sub_query = SubQuery(
                question=sub_query_text,
                answer=answer,
                source_documents=relevant_docs,
                metadata={
                    "num_sources": len(relevant_docs),
                    "context_length": len(context),
                },
            )
            sub_queries.append(sub_query)
            all_source_docs.extend(relevant_docs)

            print(f"✅ Answer: {answer[:100]}...")

        # Step 3: Synthesize final answer with reasoning
        print("\n🧠 Synthesizing final answer...")
        final_answer = await self._synthesize_final_answer(
            question, sub_queries
        )

        # Step 4: Add to conversation history if using context-aware decomposer
        if isinstance(self.query_decomposer, ContextAwareDecomposer):
            self.query_decomposer.add_to_history(question, final_answer)

        return DecomposedQuery(
            original_question=question,
            sub_queries=sub_queries,
            final_answer=final_answer,
            reasoning_steps=[f"Answered: {sq.question}" for sq in sub_queries],
            metadata={
                "decomposition_used": True,
                "num_sub_queries": len(sub_queries),
                "total_sources": len(
                    set(doc.doc_id for doc in all_source_docs if doc.doc_id)
                ),
            },
        )

    async def query_with_decomposition_stream(
        self, question: str, k: Optional[int] = None
    ) -> AsyncIterator[StreamingChunk]:
        """Streaming version of decomposed query"""
        k = k or settings.retrieval_k

        # First, do the decomposition and sub-query answering
        decomposed = await self.query_with_decomposition(question, k)

        # Stream the synthesis process
        synthesis_prompt = self._create_synthesis_prompt(
            question, decomposed.sub_queries
        )

        async for chunk in self.llm.generate_stream(synthesis_prompt):
            if chunk.content and not hasattr(self, "_decomp_metadata_sent"):
                chunk.metadata = {
                    "decomposition_used": True,
                    "num_sub_queries": len(decomposed.sub_queries),
                    "sub_queries": [
                        sq.question for sq in decomposed.sub_queries
                    ],
                    "reasoning_steps": decomposed.reasoning_steps,
                }
                self._decomp_metadata_sent = True
            yield chunk

        # Clean up flag
        if hasattr(self, "_decomp_metadata_sent"):
            delattr(self, "_decomp_metadata_sent")

    async def query(
        self, question: str, k: Optional[int] = None
    ) -> QueryResult:
        """Regular RAG query (backward compatibility)"""
        k = k or settings.retrieval_k
        relevant_docs = await self.vector_store.similarity_search(question, k=k)
        context = self._prepare_context(relevant_docs)
        prompt = self._create_prompt(question, context)
        response = await self.llm.generate(prompt)

        return QueryResult(
            response=response,
            source_documents=relevant_docs,
            metadata={"num_sources": len(relevant_docs), "query": question},
        )

    async def query_stream(
        self, question: str, k: Optional[int] = None
    ) -> AsyncIterator[StreamingChunk]:
        """Regular streaming RAG query (backward compatibility)"""
        k = k or settings.retrieval_k
        relevant_docs = await self.vector_store.similarity_search(question, k=k)
        context = self._prepare_context(relevant_docs)
        prompt = self._create_prompt(question, context)

        async for chunk in self.llm.generate_stream(prompt):
            if chunk.content and not hasattr(self, "_sources_sent"):
                chunk.metadata = {
                    "num_sources": len(relevant_docs),
                    "query": question,
                    "source_documents": relevant_docs,
                }
                self._sources_sent = True
            yield chunk

        if hasattr(self, "_sources_sent"):
            delattr(self, "_sources_sent")

    async def _synthesize_final_answer(
        self, original_question: str, sub_queries: List[SubQuery]
    ) -> str:
        """Synthesize final answer from sub-query results"""
        synthesis_prompt = self._create_synthesis_prompt(
            original_question, sub_queries
        )
        return await self.llm.generate(synthesis_prompt, temperature=0.5)

    def _create_synthesis_prompt(
        self, original_question: str, sub_queries: List[SubQuery]
    ) -> str:
        """Create prompt for synthesizing final answer"""
        qa_pairs = []
        for sq in sub_queries:
            qa_pairs.append(f"Q: {sq.question}\nA: {sq.answer}")

        qa_context = "\n\n".join(qa_pairs)

        return f"""You are an expert analyst tasked with providing a comprehensive answer to a complex question.

You have been provided with answers to several related sub-questions. Use these answers to construct a complete, well-reasoned response to the original question.

Original Question: {original_question}

Sub-Question Answers:
{qa_context}

Instructions:
- Synthesize the information from all sub-answers
- Provide a comprehensive response to the original question
- Show logical reasoning and connections between the sub-answers
- If there are contradictions, acknowledge and explain them
- Cite which sub-questions support your conclusions
- Be thorough but concise

Comprehensive Answer:"""

    def _create_sub_query_prompt(self, sub_question: str, context: str) -> str:
        """Create prompt for answering individual sub-queries"""
        return f"""You are an expert assistant answering a focused question based on provided context.

Context:
{context}

Question: {sub_question}

Instructions:
- Answer the question directly and specifically
- Base your answer on the provided context
- Be concise but complete
- If the context doesn't contain enough information, say so clearly
- Focus only on this specific question

Answer:"""

    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            similarity = doc.metadata.get("similarity_score", 0)
            context_parts.append(
                f"Source {i} (similarity: {similarity:.3f}) - {source}:\n{doc.content}\n"
            )
        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create regular RAG prompt"""
        return f"""You are an AI assistant that answers questions based on provided context.
Use the context below to answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Instructions:
- Answer based primarily on provided context
- If the context doesn't contain enough information, say so
- Cite sources when possible
- Be concise but thorough

Answer:"""


# Backward compatibility alias
RAGEngine = AdvancedRAGEngine
