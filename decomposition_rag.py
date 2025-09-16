import asyncio
import sys

from src.rag.engine import AdvancedRAGEngine


def safe_metadata_get(result, key, default=None):
    """Safe access to metadata"""
    try:
        return (
            getattr(result, "metadata", {}).get(key, default)
            if result.metadata
            else default
        )
    except (AttributeError, TypeError):
        return default


async def test_decomposition():
    """Test the RAG with query decomposition"""
    rag = AdvancedRAGEngine(use_context_aware_decomposer=True)
    print("🔄 Ingesting documents...")
    num_docs = await rag.ingest_documents("./docs")
    print(f"✅ Ingested {num_docs} document chunks\n")

    # Test queries that benefit from decomposition
    test_queries = [
        "What is the scope of this project, what is there in the vision, and how do they aim to achieve it?",
        "Can you explain some very key points of the project, and can you even explain them in detail, what is the project about and how do we find a use for this?",
    ]

    for query in test_queries:
        print("=" * 80)
        print(f"🤔 Testing Query: {query}")
        print("=" * 80)

        # Test decomposed query
        result = await rag.query_with_decomposition(query)

        print("\n📊 Results:")

        print(
            f"   - Decomposition used: {safe_metadata_get(result, 'decomposition_used', False)}"
        )
        print(
            f"   - Number of sub-queries: {safe_metadata_get(result, 'num_sub_queries', 0)}"
        )
        print(f"   - Total sources: {safe_metadata_get(result, 'total_sources', 0)}")

        print("\n🎯 Final Answer:")
        print(result.final_answer)

        print("\n🔍 Sub-queries and Answers:")
        for i, sq in enumerate(result.sub_queries, 1):
            print(f"   {i}. Q: {sq.question}")
            print(f"      A: {sq.answer[:150]}...")

        print("\n" + "=" * 80 + "\n")


async def interactive_decomposition_chat():
    """Interactive chat with decomposition"""
    rag = AdvancedRAGEngine(use_context_aware_decomposer=True)
    print("Loading Documents...")

    num_docs = await rag.ingest_documents("./docs")
    print(f"✅ Loaded {num_docs} chunks\n")

    print("💬 Advanced RAG Chat with Query Decomposition")
    print("Ask complex questions to see decomposition in action!")
    print("(type 'exit' to quit, 'stream' for streaming mode)")
    print("=" * 60)

    streaming_mode = True

    while True:
        try:
            question = input("\n🤔 Your question: ").strip()

            if question.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye!")
                break

            if question.lower() == "stream":
                streaming_mode = not streaming_mode
                print(f"🔄 Streaming mode: {'ON' if streaming_mode else 'OFF'}")
                continue

            if not question:
                continue

            if streaming_mode:
                print("🤖 Assistant: ", end="", flush=True)
                async for chunk in rag.query_with_decomposition_stream(question):
                    if chunk.content:
                        print(chunk.content, end="", flush=True)
                print()  # New line
            else:
                result = await rag.query_with_decomposition(question)
                print(f"🤖 Assistant: {result.final_answer}")

                if safe_metadata_get(result, "decomposition_used"):
                    print(
                        f"\n📝 (Decomposed into {len(result.sub_queries)} sub-questions)"
                    )

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_decomposition_chat())
    else:
        asyncio.run(test_decomposition())
