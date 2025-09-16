import argparse
import asyncio
import shutil

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from src.config.settings import settings
from src.rag.engine import AdvancedRAGEngine

# Initialize Rich Console for beautiful TUI
console = Console()


async def ingest_documents(rag: AdvancedRAGEngine, directory: str):
    """Ingests documents from a specified directory."""
    with console.status("[bold green]Ingesting documents...", spinner="dots"):
        try:
            num_docs = await rag.ingest_documents(directory)
            console.print(
                Panel(
                    f"[bold green]✅ Success![/bold green]\nIngested {num_docs} document chunks from '{directory}'.",
                    title="Ingestion Complete",
                    border_style="green",
                )
            )
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]❌ Error during ingestion:[/bold red]\n{e}",
                    title="Ingestion Failed",
                    border_style="red",
                )
            )


def delete_chroma_db():
    """Deletes the ChromaDB persistence directory."""
    db_path = settings.chroma_persist_directory
    confirm = Prompt.ask(
        f"Are you sure you want to permanently delete the database at '{db_path}'?",
        choices=["yes", "no"],
        default="no",
    )
    if confirm.lower() == "yes":
        try:
            shutil.rmtree(db_path)
            console.print(
                Panel(
                    f"[bold green]✅ Success![/bold green]\nDatabase at '{db_path}' has been deleted.",
                    title="Database Deleted",
                    border_style="green",
                )
            )
        except FileNotFoundError:
            console.print(
                Panel(
                    f"[bold yellow]⚠️ Warning![/bold yellow]\nDatabase directory '{db_path}' not found. Nothing to delete.",
                    title="Deletion Skipped",
                    border_style="yellow",
                )
            )
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]❌ Error deleting database:[/bold red]\n{e}",
                    title="Deletion Failed",
                    border_style="red",
                )
            )
    else:
        console.print("[yellow]Database deletion cancelled.[/yellow]")


async def run_tests():
    """Runs a suite of automated tests for the RAG engine."""
    console.print(
        Panel(
            "🚀 Starting RAG Engine Tests 🚀",
            title="[bold cyan]Automated Testing[/bold cyan]",
            border_style="cyan",
        )
    )
    rag = AdvancedRAGEngine(use_context_aware_decomposer=True)

    console.print("\n[bold]Step 1: Ingesting test documents...[/bold]")
    await ingest_documents(rag, "./docs")

    console.print("\n[bold]Step 2: Testing Simple RAG Query...[/bold]")
    simple_query = "What is the main topic of the documents?"
    try:
        with console.status("[bold green]Running simple query...", spinner="dots"):
            result = await rag.query(simple_query)
        table = Table(title="Simple RAG Test Result")
        table.add_column("Query", style="cyan")
        table.add_column("Response", style="magenta")
        table.add_column("Sources", style="green")
        table.add_row(simple_query, result.response, str(len(result.source_documents)))
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]❌ Simple RAG test failed:[/bold red] {e}")

    console.print("\n[bold]Step 3: Testing Decomposed RAG Query...[/bold]")
    complex_query = "Compare the project's vision with its implementation details and explain the key differences."
    try:
        with console.status("[bold green]Running decomposed query...", spinner="dots"):
            result = await rag.query_with_decomposition(complex_query)

        table = Table(title="Decomposed RAG Test Result")
        table.add_column("Original Query", style="cyan")
        table.add_column("Final Answer", style="magenta")
        table.add_column("Sub-Queries", style="green")

        sub_queries_str = "\n".join(
            [f"{i + 1}. {sq.question}" for i, sq in enumerate(result.sub_queries)]
        )
        table.add_row(complex_query, result.final_answer, sub_queries_str)
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]❌ Decomposed RAG test failed:[/bold red] {e}")

    console.print(
        Panel(
            "✅ All tests completed!",
            title="[bold green]Test Summary[/bold green]",
            border_style="green",
        )
    )


# Corrected interactive_chat function
async def interactive_chat(decomposition: bool, no_stream: bool):
    """Starts an interactive chat session with the RAG engine."""
    mode = "Decomposition" if decomposition else "Simple"
    streaming = not no_stream
    title = f"💬 Advanced RAG Chat ({mode} Mode | Streaming: {'ON' if streaming else 'OFF'}) 💬"
    console.print(Panel(title, title_align="center", border_style="bold blue"))
    console.print("Ask questions about your documents. Type 'exit' or 'quit' to end.")

    # Ingest documents once at the start of the chat session
    rag = AdvancedRAGEngine(use_context_aware_decomposer=decomposition)
    with console.status("[bold green]Loading documents...", spinner="dots"):
        await rag.ingest_documents("./docs")
    console.print("[green]✅ Documents loaded.[/green]")

    while True:
        try:
            question = Prompt.ask("\n[bold yellow]You[/bold yellow]")
            if question.lower() in ["exit", "quit"]:
                console.print("[bold blue]👋 Goodbye![/bold blue]")
                break

            # Use console.status for the thinking message
            with console.status(
                "[bold cyan]Assistant is thinking...[/bold cyan]",
                spinner="dots",
            ):
                if not streaming:
                    # Non-streaming logic remains inside the status
                    if decomposition:
                        result = await rag.query_with_decomposition(question)
                        console.print(
                            f"[bold cyan]Assistant:[/bold cyan] {result.final_answer}"
                        )
                    else:
                        result = await rag.query(question)
                        console.print(
                            f"[bold cyan]Assistant:[/bold cyan] {result.response}"
                        )
                    continue  # Skip to the next loop iteration

                # For streaming, prepare the generator inside the status block
                if decomposition:
                    stream_generator = rag.query_with_decomposition_stream(question)
                else:
                    stream_generator = rag.query_stream(question)

            # Streaming happens *after* the status block has closed
            if streaming:
                console.print("[bold cyan]Assistant:[/bold cyan] ", end="")
                full_response = ""
                async for chunk in stream_generator:
                    if chunk.content:
                        console.print(chunk.content, end="")
                        full_response += chunk.content
                console.print()  # Print a final newline

        except KeyboardInterrupt:
            console.print("\n[bold blue]👋 Goodbye![/bold blue]")
            break
        except Exception as e:
            console.print(
                Panel(
                    f"[bold red]❌ An error occurred:[/bold red]\n{e}",
                    title="Error",
                    border_style="red",
                )
            )


async def main():
    parser = argparse.ArgumentParser(
        description="A professional CLI for the RAG Engine with GitHub Models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    parser_ingest = subparsers.add_parser(
        "ingest", help="Ingest documents into the vector store."
    )
    parser_ingest.add_argument(
        "directory",
        type=str,
        default="./docs",
        nargs="?",
        help="The directory containing documents to ingest (default: ./docs).",
    )

    # Delete-db command
    subparsers.add_parser("delete-db", help="Delete the ChromaDB database directory.")

    # Test command
    subparsers.add_parser("test", help="Run automated tests for the RAG engine.")

    # Chat command (default)
    parser_chat = subparsers.add_parser(
        "chat", help="Start an interactive chat session (default)."
    )
    parser_chat.add_argument(
        "-d",
        "--decomposition",
        action="store_true",
        help="Run in query decomposition mode for complex questions.",
    )
    parser_chat.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and wait for the full response.",
    )

    args = parser.parse_args()

    rag = AdvancedRAGEngine()

    if args.command == "ingest":
        await ingest_documents(rag, args.directory)
    elif args.command == "delete-db":
        delete_chroma_db()
    elif args.command == "test":
        await run_tests()
    elif args.command == "chat":
        await interactive_chat(args.decomposition, args.no_stream)


if __name__ == "__main__":
    # To handle cases where no command is provided, default to chat
    import sys

    if len(sys.argv) == 1:
        sys.argv.append("chat")

    asyncio.run(main())
