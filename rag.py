import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Prompt

from src.rag.engine import AdvancedRAGEngine

app = typer.Typer(help="Interactive RAG Chat with Query Decomposition")
console = Console()


class InteractiveRAGChat:
    def __init__(self, docs_path: str):
        self.rag = AdvancedRAGEngine(use_context_aware_decomposer=True)
        self.docs_path = docs_path
        self.is_initialized = False

    async def initialize(self):
        """Initialize the RAG system with documents"""
        if self.is_initialized:
            return

        with console.status("[bold cyan]Loading documents...", spinner="dots"):
            try:
                num_docs = await self.rag.ingest_documents(self.docs_path)
                self.is_initialized = True
                console.print(
                    f"[green]✅ Loaded {num_docs} document chunks successfully![/green]"
                )
            except Exception as e:
                console.print(f"[red]❌ Failed to load documents: {e}[/red]")
                raise

    async def process_query_with_thinking(self, question: str):
        """Process query with perplexity-style thinking indicators"""

        # Step 1: Show decomposition thinking
        with console.status(
            "[bold yellow]🤔 Analyzing your question...", spinner="dots"
        ):
            await asyncio.sleep(1)
            sub_query_texts = await self.rag.query_decomposer.decompose_query(
                question
            )

        if len(sub_query_texts) == 1:
            console.print(
                "[dim]💭 Simple question detected, no decomposition needed[/dim]"
            )
            await self.stream_simple_answer(question)
        else:
            await self.process_decomposed_query(question, sub_query_texts)

    async def process_decomposed_query(self, question: str, sub_queries: list):
        """Process decomposed query with progress tracking"""

        # Show decomposition results
        console.print(
            f"[bold blue]🔧 Breaking down into {len(sub_queries)} focused questions:[/bold blue]"
        )
        for i, sq in enumerate(sub_queries, 1):
            console.print(f"   [dim]{i}.[/dim] {sq}")

        console.print()

        # Create progress bar for sub-queries
        progress_columns = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        )

        with Progress(*progress_columns) as progress:
            task = progress.add_task(
                "[cyan]Researching answers...", total=len(sub_queries)
            )

            sub_query_results = []

            for i, sub_query in enumerate(sub_queries):
                progress.update(
                    task,
                    description=f"[cyan]🔍 Researching: {sub_query[:50]}...",
                )

                # Get answer for this sub-query
                relevant_docs = await self.rag.vector_store.similarity_search(
                    sub_query, k=5
                )
                context = self.rag._prepare_context(relevant_docs)
                prompt = self.rag._create_sub_query_prompt(sub_query, context)

                answer_chunks = []
                async for chunk in self.rag.llm.generate_stream(
                    prompt, temperature=0.7
                ):
                    answer_chunks.append(chunk.content)

                answer = "".join(answer_chunks)
                sub_query_results.append(
                    {
                        "question": sub_query,
                        "answer": answer,
                        "sources": len(relevant_docs),
                    }
                )

                progress.advance(task)
                await asyncio.sleep(0.5)  # Small delay for better UX

        # Show research results
        console.print("\n[bold green]📚 Research Complete![/bold green]")
        for i, result in enumerate(sub_query_results, 1):
            console.print(
                f"[dim]{i}. {result['question'][:60]}... ({result['sources']} sources)[/dim]"
            )

        # Synthesize final answer
        console.print(
            "\n[bold magenta]🧠 Synthesizing comprehensive answer...[/bold magenta]"
        )
        await self.stream_synthesized_answer(question, sub_query_results)

    async def stream_synthesized_answer(self, question: str, sub_results: list):
        """Stream the final synthesized answer"""

        # Create synthesis prompt
        qa_pairs = []
        for result in sub_results:
            qa_pairs.append(f"Q: {result['question']}\nA: {result['answer']}")

        qa_context = "\n\n".join(qa_pairs)
        synthesis_prompt = f"""You are an expert analyst providing a comprehensive answer.

Original Question: {question}

Research Results:
{qa_context}

Instructions:
- Synthesize information from all research
- Provide a comprehensive, well-structured response
- Show logical connections between findings
- Be thorough but clear and engaging

Comprehensive Answer:"""

        # Stream the answer with rich formatting
        console.print("\n" + "=" * 60)
        console.print("[bold cyan]🎯 Final Answer:[/bold cyan]\n")

        response_text = ""
        async for chunk in self.rag.llm.generate_stream(synthesis_prompt):
            if chunk.content:
                console.print(chunk.content, end="", highlight=False)
                response_text += chunk.content

        console.print("\n" + "=" * 60)

        # Show metadata
        total_sources = sum(r["sources"] for r in sub_results)
        console.print(
            f"[dim]📊 Used {len(sub_results)} research steps • {total_sources} total sources[/dim]"
        )

    async def stream_simple_answer(self, question: str):
        """Stream answer for simple questions"""

        with console.status(
            "[bold yellow]🔍 Searching knowledge base...", spinner="dots"
        ):
            relevant_docs = await self.rag.vector_store.similarity_search(
                question, k=5
            )
            context = self.rag._prepare_context(relevant_docs)
            prompt = self.rag._create_prompt(question, context)

        console.print("\n" + "=" * 60)
        console.print("[bold cyan]🎯 Answer:[/bold cyan]\n")

        async for chunk in self.rag.llm.generate_stream(prompt):
            if chunk.content:
                console.print(chunk.content, end="", highlight=False)

        console.print("\n" + "=" * 60)
        console.print(f"[dim]📊 Used {len(relevant_docs)} sources[/dim]")

    async def chat_loop(self):
        """Main interactive chat loop"""

        # Welcome message
        welcome_panel = Panel.fit(
            "[bold cyan]🤖 Interactive RAG Assistant[/bold cyan]\n\n"
            "Ask complex questions and I'll break them down for comprehensive answers!\n"
            "[dim]Type 'exit' to quit, 'help' for commands[/dim]",
            border_style="cyan",
        )
        console.print(welcome_panel)
        console.print()

        while True:
            try:
                # Get user input with rich prompt
                question = Prompt.ask(
                    "[bold blue]💬 Your Question",
                    default="",
                    show_default=False,
                ).strip()

                if not question:
                    continue

                if question.lower() in ["exit", "quit", "bye"]:
                    console.print("[bold red]👋 Goodbye![/bold red]")
                    break

                if question.lower() == "help":
                    self.show_help()
                    continue

                if question.lower() == "clear":
                    console.clear()
                    continue

                # Process the question
                console.print()
                await self.process_query_with_thinking(question)
                console.print("\n")

            except KeyboardInterrupt:
                console.print("\n[bold red]👋 Goodbye![/bold red]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def show_help(self):
        """Show help information"""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

• [bold]help[/bold] - Show this help message
• [bold]clear[/bold] - Clear the screen
• [bold]exit/quit/bye[/bold] - Exit the chat

[bold cyan]Tips:[/bold cyan]

• Ask complex questions for automatic decomposition
• Questions with multiple parts work best
• Try: "What are the benefits and risks of X?"
• Try: "Compare A and B, and explain the differences"
"""
        console.print(Panel(help_text, border_style="blue"))


@app.command()
def chat(
    docs_path: str = typer.Argument(
        "./docs", help="Path to documents directory"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose output"
    ),
):
    """Start interactive RAG chat session"""

    # Check if docs path exists
    if not Path(docs_path).exists():
        console.print(f"[red]❌ Documents path '{docs_path}' not found![/red]")
        console.print(
            "[yellow]💡 Create a 'docs' folder and add some text files to get started.[/yellow]"
        )
        raise typer.Exit(1)

    async def main():
        chat_instance = InteractiveRAGChat(docs_path)

        try:
            await chat_instance.initialize()
            await chat_instance.chat_loop()
        except Exception as e:
            if verbose:
                console.print_exception()
            else:
                console.print(f"[red]❌ Error: {e}[/red]")
            raise typer.Exit(1)

    # Run the async chat
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]👋 Goodbye![/bold red]")


@app.command()
def demo():
    """Run a demo with sample questions"""

    demo_questions = [
        "What are the benefits and risks of artificial intelligence?",
        "Compare machine learning and deep learning approaches",
        "Explain the causes and effects of climate change",
        "What are the advantages and disadvantages of renewable energy?",
    ]

    console.print("[bold cyan]🎬 Demo Mode - Sample Questions:[/bold cyan]\n")

    for i, question in enumerate(demo_questions, 1):
        console.print(f"{i}. {question}")

    console.print(
        "\n[dim]Copy any question above and use it in chat mode![/dim]"
    )
    console.print(
        "[yellow]💡 Run: python test_interactive_rag.py chat[/yellow]"
    )


if __name__ == "__main__":
    app()
