import asyncio
import time
from datetime import datetime
from typing import AsyncIterator, List, Optional

from ..config.settings import settings
from ..core.interfaces import (
    Document,
    LLMInterface,
    ReflexionCycle,
    ReflexionDecision,
    ReflexionEvaluatorInterface,
    ReflexionMemory,
    StreamingChunk,
    VectorStoreInterface,
)
from ..data.loader import DocumentLoader
from ..data.processor import DocumentProcessor
from ..llm.github_llm import GitHubLLM
from ..memory.cache import ReflexionMemoryCache, create_query_hash
from ..reflexion.evaluator import SmartReflexionEvaluator
from ..vectorstore.chroma_store import ChromaVectorStore


class ReflexionRAGEngine:
    """Advanced RAG engine with dynamic reflexion loop architecture"""

    def __init__(
        self,
        generation_llm: Optional[LLMInterface] = None,
        evaluation_llm: Optional[LLMInterface] = None,
        summary_llm: Optional[LLMInterface] = None,
        vector_store: Optional[VectorStoreInterface] = None,
        reflexion_evaluator: Optional[ReflexionEvaluatorInterface] = None,
    ):
        # Initialize different LLMs for different purposes
        self.generation_llm = generation_llm or GitHubLLM(
            model_override=settings.llm_model,
            temperature_override=settings.llm_temperature,
            max_tokens_override=settings.llm_max_tokens,
        )

        self.evaluation_llm = evaluation_llm or GitHubLLM(
            model_override=settings.evaluation_model,
            temperature_override=settings.evaluation_temperature,
            max_tokens_override=settings.evaluation_max_tokens,
        )

        self.summary_llm = summary_llm or GitHubLLM(
            model_override=settings.summary_model,
            temperature_override=settings.summary_temperature,
            max_tokens_override=settings.summary_max_tokens,
        )

        self.vector_store = vector_store or ChromaVectorStore()
        self.reflexion_evaluator = (
            reflexion_evaluator or SmartReflexionEvaluator(self.evaluation_llm)
        )

        # Memory cache for reflexion loops
        self.memory_cache = (
            ReflexionMemoryCache() if settings.enable_memory_cache else None
        )

        # Document processing
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

    async def query_with_reflexion_stream(
        self, question: str
    ) -> AsyncIterator[StreamingChunk]:
        """Main reflexion loop query with streaming"""

        if not settings.enable_reflexion_loop:
            # Fallback to simple RAG
            async for chunk in self.simple_query_stream(question):
                yield chunk
            return

        print(f"🔄 Starting Reflexion Loop for: {question}")

        # Check memory cache first
        query_hash = create_query_hash(question)
        cached_memory = None
        if self.memory_cache:
            cached_memory = self.memory_cache.get(query_hash)
            if cached_memory:
                print("💾 Found cached reflexion result")
                async for chunk in self._stream_cached_result(cached_memory):
                    yield chunk
                return

        # Initialize reflexion memory
        reflexion_memory = ReflexionMemory(original_query=question)
        start_time = time.time()
        try:
            # Reflexion loop
            cycle_number = 1
            current_query = question

            while cycle_number <= settings.max_reflexion_cycles:
                print(f"\n🔄 Reflexion Cycle {cycle_number}")
                print(f"🔍 Query: {current_query}")

                cycle_start = time.time()

                # Step 1: Retrieve documents
                k = (
                    settings.initial_retrieval_k
                    if cycle_number == 1
                    else settings.reflexion_retrieval_k
                )
                retrieved_docs = await self.vector_store.similarity_search(
                    current_query, k=k
                )
                print(f"📚 Retrieved {len(retrieved_docs)} documents")

                # Step 2: Generate partial answer
                context = self._prepare_context(retrieved_docs)
                generation_prompt = self._create_generation_prompt(
                    current_query, context, cycle_number
                )

                partial_answer_chunks = []
                async for chunk in self.generation_llm.generate_stream(
                    generation_prompt
                ):
                    partial_answer_chunks.append(chunk.content)
                    # Stream intermediate answers as markdown blocks
                    if chunk.content:
                        yield StreamingChunk(
                            content=chunk.content,
                            metadata={
                                "cycle_number": cycle_number,
                                "is_partial": True,
                                "reflexion_mode": True,
                            },
                        )

                partial_answer = "".join(partial_answer_chunks)
                print(f"✅ Generated answer ({len(partial_answer)} chars)")

                # Step 3: Self-evaluation
                print("🤔 Evaluating response quality...")
                evaluation = await self.reflexion_evaluator.evaluate_response(
                    question, partial_answer, retrieved_docs, cycle_number
                )

                print(f"📊 Confidence: {evaluation.confidence_score:.2f}")
                print(f"🎯 Decision: {evaluation.decision.value}")
                print(f"💭 Reasoning: {evaluation.reasoning}")

                # Create reflexion cycle
                cycle = ReflexionCycle(
                    cycle_number=cycle_number,
                    query=current_query,
                    retrieved_docs=retrieved_docs,
                    partial_answer=partial_answer,
                    evaluation=evaluation,
                    timestamp=datetime.now(),
                    processing_time=time.time() - cycle_start,
                )
                reflexion_memory.add_cycle(cycle)

                # Step 4: Decision tree
                if evaluation.decision == ReflexionDecision.COMPLETE:
                    print("✅ Response is complete!")
                    reflexion_memory.final_answer = partial_answer
                    break

                elif evaluation.decision == ReflexionDecision.INSUFFICIENT_DATA:
                    print("❌ Insufficient data in knowledge base")
                    reflexion_memory.final_answer = (
                        self._create_insufficient_data_response(
                            question, reflexion_memory.get_all_partial_answers()
                        )
                    )
                    break

                elif (
                    evaluation.confidence_score >= settings.confidence_threshold
                ):
                    print("✅ Confidence threshold reached!")
                    reflexion_memory.final_answer = partial_answer
                    break

                elif cycle_number >= settings.max_reflexion_cycles:
                    print("🔄 Max cycles reached, synthesizing final answer...")
                    break

                else:
                    # Continue with follow-up queries
                    if evaluation.follow_up_queries:
                        current_query = evaluation.follow_up_queries[0]
                        print(f"🔄 Following up with: {current_query}")
                    else:
                        # Generate follow-up queries
                        follow_ups = await self.reflexion_evaluator.generate_follow_up_queries(
                            question, partial_answer, evaluation.missing_aspects
                        )
                        if follow_ups:
                            current_query = follow_ups[0]
                            print(f"🔄 Generated follow-up: {current_query}")
                        else:
                            print("❌ No follow-up queries generated, stopping")
                            reflexion_memory.final_answer = partial_answer
                            break

                cycle_number += 1
                await asyncio.sleep(0.1)  # Brief pause between cycles

            # Final synthesis if needed
            if (
                not reflexion_memory.final_answer
                and len(reflexion_memory.cycles) > 1
            ):
                print("\n🧠 Synthesizing final comprehensive answer...")
                reflexion_memory.final_answer = (
                    await self._synthesize_final_answer(
                        question, reflexion_memory
                    )
                )

            # Stream final answer
            final_answer = (
                reflexion_memory.final_answer
                or "Unable to generate a complete answer."
            )

            # Add processing metadata
            reflexion_memory.total_processing_time = time.time() - start_time

            # Cache the result
            if self.memory_cache:
                self.memory_cache.put(query_hash, reflexion_memory)

            # Stream final answer with metadata
            yield StreamingChunk(
                content=f"\n\n## 🎯 Final Comprehensive Answer\n\n{final_answer}",
                is_complete=True,
                metadata={
                    "reflexion_complete": True,
                    "total_cycles": len(reflexion_memory.cycles),
                    "total_processing_time": reflexion_memory.total_processing_time,
                    "total_documents": reflexion_memory.total_documents_retrieved,
                    "final_confidence": reflexion_memory.cycles[
                        -1
                    ].evaluation.confidence_score
                    if reflexion_memory.cycles
                    else 0.0,
                    "memory_cached": self.memory_cache is not None,
                },
            )

        except Exception as e:
            print(f"❌ Reflexion loop failed: {e}")
            # Fallback to simple query
            async for chunk in self.simple_query_stream(question):
                yield chunk

    async def simple_query_stream(
        self, question: str
    ) -> AsyncIterator[StreamingChunk]:
        """Simple RAG query without reflexion (fallback)"""
        print("📝 Using simple RAG mode")

        retrieved_docs = await self.vector_store.similarity_search(
            question, k=settings.retrieval_k
        )
        context = self._prepare_context(retrieved_docs)
        prompt = self._create_simple_prompt(question, context)

        async for chunk in self.generation_llm.generate_stream(prompt):
            if chunk.content:
                yield StreamingChunk(
                    content=chunk.content,
                    metadata={
                        "simple_mode": True,
                        "num_sources": len(retrieved_docs),
                    },
                )

    async def _stream_cached_result(
        self, memory: ReflexionMemory
    ) -> AsyncIterator[StreamingChunk]:
        """Stream cached reflexion result"""

        # Stream each cycle's partial answer
        for i, cycle in enumerate(memory.cycles, 1):
            yield StreamingChunk(
                content=f"\n## 🔄 Cycle {i} (Cached)\n\n{cycle.partial_answer}\n",
                metadata={
                    "cycle_number": i,
                    "is_cached": True,
                    "confidence": cycle.evaluation.confidence_score,
                },
            )
            await asyncio.sleep(0.1)  # Simulate streaming delay

        # Stream final answer
        yield StreamingChunk(
            content=f"\n## 🎯 Final Answer (Cached)\n\n{memory.final_answer}",
            is_complete=True,
            metadata={
                "cached_result": True,
                "total_cycles": len(memory.cycles),
                "total_processing_time": memory.total_processing_time,
            },
        )

    async def _synthesize_final_answer(
        self, question: str, memory: ReflexionMemory
    ) -> str:
        """Synthesize final answer from all reflexion cycles"""

        partial_answers = memory.get_all_partial_answers()
        all_docs = memory.get_all_retrieved_docs()

        synthesis_prompt = self._create_synthesis_prompt(
            question, partial_answers, all_docs, memory.cycles
        )

        answer_chunks = []
        async for chunk in self.summary_llm.generate_stream(synthesis_prompt):
            answer_chunks.append(chunk.content)

        return "".join(answer_chunks)

    def _create_generation_prompt(
        self, query: str, context: str, cycle_number: int
    ) -> str:
        """Create prompt for answer generation"""

        cycle_instruction = ""
        if cycle_number == 1:
            cycle_instruction = "This is the initial response. Provide a comprehensive answer based on the available context."
        else:
            cycle_instruction = f"This is cycle {cycle_number} of a reflexion loop. Focus on addressing specific aspects that may have been missed in previous attempts."

        return f"""You are an expert AI assistant providing detailed, accurate answers based on retrieved context.

{cycle_instruction}

Question: {query}

Context from Knowledge Base:
{context}

Instructions:
- Provide a comprehensive, well-structured answer
- Base your response on the provided context
- Be specific and detailed where possible
- If information is incomplete, clearly state what's missing
- Use clear, professional language
- Organize information logically

Answer:"""

    def _create_synthesis_prompt(
        self,
        question: str,
        partial_answers: List[str],
        all_docs: List[Document],
        cycles: List[ReflexionCycle],
    ) -> str:
        """Create prompt for final answer synthesis"""

        # Combine all partial answers
        answers_text = "\n\n".join(
            [
                f"Cycle {i + 1} Answer:\n{answer}"
                for i, answer in enumerate(partial_answers)
            ]
        )

        # Get evaluation insights
        evaluation_insights = []
        for cycle in cycles:
            eval_info = f"Cycle {cycle.cycle_number}: Confidence {cycle.evaluation.confidence_score:.2f}"
            if cycle.evaluation.missing_aspects:
                eval_info += (
                    f", Missing: {', '.join(cycle.evaluation.missing_aspects)}"
                )
            evaluation_insights.append(eval_info)

        insights_text = "\n".join(evaluation_insights)

        return f"""You are an expert analyst tasked with creating a comprehensive final answer by synthesizing multiple research cycles.

Original Question: {question}

Research Cycles and Answers:
{answers_text}

Evaluation Insights:
{insights_text}

Total Documents Analyzed: {len(all_docs)}

Instructions:
- Synthesize all partial answers into one comprehensive response
- Resolve any contradictions by explaining different perspectives
- Highlight the most confident and well-supported information
- Organize the final answer logically with clear structure
- Include relevant details from all cycles
- Acknowledge any remaining uncertainties
- Provide a complete, authoritative answer to the original question

Final Comprehensive Answer:"""

    def _create_simple_prompt(self, question: str, context: str) -> str:
        """Create simple RAG prompt (fallback)"""
        return f"""You are an AI assistant that answers questions based on provided context.

Question: {question}

Context:
{context}

Instructions:
- Answer based on the provided context
- Be comprehensive and accurate
- If context is insufficient, state what information is missing

Answer:"""

    def _create_insufficient_data_response(
        self, question: str, partial_answers: List[str]
    ) -> str:
        """Create response when insufficient data is available"""

        if partial_answers:
            combined = "\n\n".join(partial_answers)
            return f"""Based on the available information in the knowledge base, I can provide the following partial answer:

{combined}

However, the knowledge base appears to have insufficient information to fully answer your question: "{question}"

To get a complete answer, you may need to:
- Add more relevant documents to the knowledge base
- Consult additional sources
- Rephrase your question to focus on available information"""

        return f"""I apologize, but the knowledge base does not contain sufficient information to answer your question: "{question}"

Please consider:
- Adding relevant documents to the knowledge base
- Checking if your question relates to the available content
- Rephrasing your question to match available information"""

    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from retrieved documents"""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            similarity = doc.metadata.get("similarity_score", 0)
            context_parts.append(
                f"Source {i} (similarity: {similarity:.3f}) - {source}:\n{doc.content}\n"
            )
        return "\n".join(context_parts)

    def get_memory_stats(self) -> dict:
        """Get memory cache statistics"""
        if self.memory_cache:
            return self.memory_cache.get_stats()
        return {"cache_disabled": True}

    async def clear_memory_cache(self) -> bool:
        """Clear memory cache"""
        if self.memory_cache:
            self.memory_cache.clear()
            return True
        return False
