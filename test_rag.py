import asyncio

from src.rag.engine import RAGEngine


async def main():
    rag = RAGEngine()

    print("Ingesting documents ....")
    num_docs = await rag.ingest_documents("./docs")
    print(f"Ingested {num_docs} document  chunks")

    # Query the system
    print("\nQuerying the system ....")
    result = await rag.query("What is the main topic discussed in the documents?")

    print(f"Response:{result.response}")
    print(f"Sources used: {result.metadata['num_sources']}")


if __name__ == "__main__":
    asyncio.run(main())
