import asyncio
import sys

from src.rag.engine import RAGEngine


async def test_streaming():
    rag = RAGEngine()
    print("Ingesting documents...")
    num_docs = await rag.ingest_documents("./docs")
    print(f"Ingested {num_docs} document chunks\n")

    # Test a regular query here
    print("Regular Query:")
    result = await rag.query("What is the main topic discussed in the documents?")
    print(f"Response: {result.response}\n")
    print(f"Sources used: {result.metadata['num_sources']}\n")

    # Test Streaming queries
    print("Streaming Query:")
    print("Response: ", end="", flush=True)

    full_response = ""
    usage_info = None
    async for chunk in rag.query_stream("Summarize the key points from the documents"):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_response += chunk.content

        if chunk.usage_info:
            usage_info = chunk.usage_info

    print("\n")

    if usage_info:
        print(f"Token usage: {usage_info}")

    print(f"\nFull response length: {len(full_response)} characters")


async def interactive_streaming_chat():
    """Interactive streaming chat with your RAG system"""
    rag = RAGEngine()
    print("Loading documents ...")
    num_docs = await rag.ingest_documents("./docs")
    print(f"Loaded {num_docs} chunks\n")

    print("💬 Interactive RAG Chat (type 'exit' to quit)")
    print("=" * 50)

    while True:
        try:
            question = input("\n your question: ").strip()

            if question.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            if not question:
                continue
            print("🤖 Assistant: ", end="", flush=True)

            async for chunk in rag.query_stream(question):
                if chunk.content:
                    print(chunk.content, end="", flush=True)

            print()

        except KeyboardInterrupt:
            print("\n Goodbye!....")
            break
        except Exception as e:
            print(f"\n Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_streaming_chat())
    else:
        asyncio.run(test_streaming())
