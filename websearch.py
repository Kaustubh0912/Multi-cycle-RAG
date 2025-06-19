"""
Google Custom Search + Crawl4AI Content Extractor

Version: 2.1.0 - Fixed for Crawl4AI v0.6.x
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, cast

# Third-party imports
import requests

# CORRECTED Crawl4AI imports for v0.6.x
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

# For async generator handling

load_dotenv()
# Fixed logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
console = Console()

# Constants
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", 30))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 10))
MIN_CONTENT_LENGTH = int(os.getenv("MIN_CONTENT_LENGTH", 200))
MAX_TITLE_LENGTH = int(os.getenv("MAX_TITLE_LENGTH", 80))


class ExtractionStrategy(Enum):
    """Content extraction strategies"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"


class ContentStatus(Enum):
    """Content extraction status"""

    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    LOW_QUALITY = "low_quality"


@dataclass
class SearchResult:
    """Data class for search results"""

    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class ExtractedContent:
    """Data class for extracted content"""

    url: str
    title: str
    content: str
    status: ContentStatus
    strategy: Optional[ExtractionStrategy] = None
    word_count: int = 0
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Calculate word count after initialization"""
        if self.content and self.status == ContentStatus.SUCCESS:
            self.word_count = len(self.content.split())


@dataclass
class ContentSelectors:
    """CSS selectors for different content extraction strategies"""

    primary: str = "article, .article, .post, .content, main, .main-content, .entry-content"
    secondary: str = ".post-content, .article-body, .blog-post, .story-body"
    fallback: str = 'p, .text, .description, .summary, div[class*="content"]'


class GoogleSearchExtractor:
    """Google Custom Search API integration with enhanced error handling"""

    def __init__(self, api_key: str, cse_id: str) -> None:
        """Initialize the Google Search extractor"""
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()

    def search(
        self, query: str, num_results: int = 5
    ) -> Optional[Dict[str, Any]]:
        """Perform Google Custom Search and return results"""
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, MAX_SEARCH_RESULTS),
        }

        try:
            response = self.session.get(
                self.base_url, params=params, timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()

            data = response.json()
            logger.info(
                "Search successful: %d results found",
                len(data.get("items", [])),
            )
            return data

        except requests.exceptions.RequestException as e:
            logger.error("Search API Error: %s", str(e))
            console.print(f"[red]Search API Error: {str(e)}[/red]")
            return None
        except Exception as e:
            logger.error("Unexpected search error: %s", str(e))
            console.print(f"[red]Unexpected search error: {str(e)}[/red]")
            return None

    def extract_search_results(
        self, search_data: Dict[str, Any]
    ) -> List[SearchResult]:
        """Extract SearchResult objects from search data"""
        results = []
        if "items" in search_data:
            for i, item in enumerate(search_data["items"], 1):
                result = SearchResult(
                    title=item.get("title", "N/A"),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", "N/A"),
                    rank=i,
                )
                results.append(result)
        return results

    def extract_urls(self, search_data: Dict[str, Any]) -> List[str]:
        """Extract URLs from search results"""
        urls = []
        if "items" in search_data:
            for item in search_data["items"]:
                if "link" in item:
                    urls.append(item["link"])
        return urls

    def display_search_results(
        self, search_results: List[SearchResult]
    ) -> None:
        """Display search results in a formatted table"""
        if not search_results:
            console.print("[yellow]No search results found[/yellow]")
            return

        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Rank", style="dim", width=5)
        table.add_column("Title", style="cyan", width=40)
        table.add_column("URL", style="green", width=50)
        table.add_column("Snippet", style="white", width=60)

        for result in search_results:
            table.add_row(
                str(result.rank),
                result.title[:40],
                result.url[:50],
                result.snippet[:60],
            )

        console.print(table)

    def __del__(self) -> None:
        """Clean up session on deletion"""
        if hasattr(self, "session"):
            self.session.close()


class SmartContentExtractor:
    """Enhanced Crawl4AI integration with intelligent content filtering"""

    def __init__(self) -> None:
        """Initialize the content extractor"""
        self.selectors = ContentSelectors()

    async def extract_content(self, urls: List[str]) -> List[ExtractedContent]:
        """Extract clean, contextual content from URLs using multiple strategies"""
        results: List[ExtractedContent] = []

        # Create browser config for better stability
        browser_config = BrowserConfig(
            headless=True, browser_type="chromium", verbose=False
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Extracting content...", total=len(urls)
                )

                for url in urls:
                    try:
                        progress.update(
                            task, description=f"Processing: {url[:50]}..."
                        )

                        # Try extraction strategies in order
                        content = await self._extract_with_fallback(
                            crawler, url
                        )
                        results.append(content)

                    except Exception as e:
                        logger.error(
                            "Error extracting content from %s: %s", url, str(e)
                        )
                        error_content = ExtractedContent(
                            url=url,
                            title="Exception occurred",
                            content="",
                            status=ContentStatus.ERROR,
                            error_message=str(e),
                        )
                        results.append(error_content)

                    progress.advance(task)

        return results

    async def _extract_with_fallback(
        self, crawler: AsyncWebCrawler, url: str
    ) -> ExtractedContent:
        """Try multiple extraction strategies with fallback"""
        strategies = [
            ExtractionStrategy.PRIMARY,
            ExtractionStrategy.SECONDARY,
            ExtractionStrategy.FALLBACK,
        ]

        for strategy in strategies:
            try:
                content = await self._extract_with_strategy(
                    crawler, url, strategy
                )

                if (
                    content.status == ContentStatus.SUCCESS
                    and self._is_quality_content(content.content)
                ):
                    return content

            except Exception as e:
                logger.warning(
                    "Strategy %s failed for %s: %s", strategy.value, url, str(e)
                )
                continue

        # If all strategies fail, return the last attempt
        return ExtractedContent(
            url=url,
            title="All strategies failed",
            content="Unable to extract meaningful content",
            status=ContentStatus.FAILED,
            error_message="All extraction strategies failed",
        )

    async def _extract_with_strategy(
        self, crawler: AsyncWebCrawler, url: str, strategy: ExtractionStrategy
    ) -> ExtractedContent:
        """Extract content using a specific strategy - FIXED VERSION"""
        config = self._get_config_for_strategy(strategy)

        try:
            # Get the result and handle type inference issues
            crawl_result = await crawler.arun(url=url, config=config)

            # Cast to Any to bypass type checker issues, then validate at runtime
            result: Any = cast(Any, crawl_result)

            # Robust runtime validation instead of relying on type checker
            success = getattr(result, "success", False)
            markdown = getattr(result, "markdown", None)

            if success and markdown:
                # Extract markdown content safely with fallbacks
                markdown_content = ""

                if hasattr(markdown, "fit_markdown") and markdown.fit_markdown:
                    markdown_content = str(markdown.fit_markdown)
                elif (
                    hasattr(markdown, "raw_markdown") and markdown.raw_markdown
                ):
                    markdown_content = str(markdown.raw_markdown)
                elif markdown:
                    markdown_content = str(markdown)

                if markdown_content:
                    cleaned_content = self._clean_content(markdown_content)

                    return ExtractedContent(
                        url=url,
                        title=self._extract_title_from_markdown(
                            cleaned_content
                        ),
                        content=cleaned_content,
                        status=ContentStatus.SUCCESS,
                        strategy=strategy,
                    )

            # Handle failure cases
            error_msg = getattr(
                result, "error_message", "Failed to extract content"
            )
            return ExtractedContent(
                url=url,
                title="Failed to extract",
                content="",
                status=ContentStatus.ERROR,
                strategy=strategy,
                error_message=error_msg,
            )

        except Exception as e:
            return ExtractedContent(
                url=url,
                title="Exception occurred",
                content="",
                status=ContentStatus.ERROR,
                strategy=strategy,
                error_message=f"Extraction failed: {str(e)}",
            )

    def _get_config_for_strategy(
        self, strategy: ExtractionStrategy
    ) -> CrawlerRunConfig:
        """Get crawler configuration for specific strategy - COMPLETELY FIXED"""

        if strategy == ExtractionStrategy.PRIMARY:
            return CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.55,
                        threshold_type="fixed",
                        min_word_threshold=75,
                    )
                ),
                css_selector=self.selectors.primary,
                excluded_tags=[
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "form",
                    "script",
                    "style",
                    "menu",
                    "sidebar",
                ],
                word_count_threshold=25,
                exclude_external_links=True,
                exclude_social_media_links=True,
                remove_overlay_elements=True,
            )
        elif strategy == ExtractionStrategy.SECONDARY:
            return CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.4,
                        threshold_type="fixed",
                        min_word_threshold=50,
                    )
                ),
                css_selector=self.selectors.secondary,
                excluded_tags=[
                    "nav",
                    "header",
                    "footer",
                    "aside",
                    "form",
                    "script",
                    "style",
                ],
                word_count_threshold=20,
            )
        else:  # FALLBACK
            return CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.3,
                        threshold_type="fixed",
                        min_word_threshold=30,
                    )
                ),
                css_selector=self.selectors.fallback,
                excluded_tags=["nav", "header", "footer", "aside", "form"],
                word_count_threshold=15,
            )

    def _clean_content(self, content: str) -> str:
        """Clean extracted content from navigation and UI elements"""
        if not content:
            return content

        # Navigation patterns to remove
        nav_patterns = [
            r"Skip to main content.*?\n",
            r"Open menu.*?\n",
            r"Log [Ii]n.*?\n",
            r"Sign [Uu]p.*?\n",
            r"Subscribe.*?\n",
            r"Follow us.*?\n",
            r"Get App.*?\n",
            r"Download.*?\n",
            r"\[.*?\]\(#.*?\)",  # Skip links
            r"^\s*\*\s*&nbsp;\s*$",  # Empty navigation items
            r"^\s*\*\s*$",  # Empty bullet points
        ]

        cleaned = content
        for pattern in nav_patterns:
            cleaned = re.sub(
                pattern, "", cleaned, flags=re.MULTILINE | re.IGNORECASE
            )

        # Remove excessive whitespace
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        cleaned = re.sub(r"^\s+|\s+$", "", cleaned, flags=re.MULTILINE)

        # Filter navigation lines
        lines = cleaned.split("\n")
        filtered_lines = []

        nav_indicators = [
            "menu",
            "navigation",
            "log in",
            "sign up",
            "subscribe",
            "follow",
            "get app",
        ]

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                filtered_lines.append(line)
                continue

            # Skip obvious navigation lines
            is_nav = (
                any(indicator in line_lower for indicator in nav_indicators)
                and len(line.strip()) < 50
            )

            if not is_nav:
                filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    def _is_quality_content(self, content: str) -> bool:
        """Check if extracted content meets quality standards"""
        if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
            return False

        # Check for navigation indicators
        nav_indicators = [
            "skip to main content",
            "open menu",
            "log in",
            "sign up",
            "subscribe",
            "follow us",
        ]
        content_lower = content.lower()
        nav_count = sum(
            1 for indicator in nav_indicators if indicator in content_lower
        )

        if nav_count > 3:  # Too many navigation elements
            return False

        # Check word-to-link ratio
        words = len(content.split())
        links = content.count("[")  # Markdown links
        if words > 0 and links / words > 0.3:  # Too many links
            return False

        # Check for meaningful sentences
        sentences = content.split(".")
        meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]

        if len(meaningful_sentences) < 3:  # Too few meaningful sentences
            return False

        return True

    def _extract_title_from_markdown(self, markdown: str) -> str:
        """Extract title from markdown content"""
        if not markdown:
            return "No title found"

        lines = markdown.split("\n")

        # Look for markdown headers first
        for line in lines[:15]:
            line = line.strip()
            if line.startswith("# ") and len(line) > 3:
                title = line[2:].strip()
                if self._is_valid_title(title):
                    return title[:MAX_TITLE_LENGTH]

        # Look for any meaningful heading
        for line in lines[:20]:
            line = line.strip()
            if (
                line
                and not line.startswith("[")
                and 10 < len(line) < 100
                and self._is_valid_title(line)
            ):
                return line[:MAX_TITLE_LENGTH]

        return "No title found"

    def _is_valid_title(self, title: str) -> bool:
        """Check if a title is valid (not navigation)"""
        nav_keywords = ["menu", "skip", "log in", "sign up", "subscribe"]
        return not any(nav in title.lower() for nav in nav_keywords)


class ContentExtractorApp:
    """Main application class with enhanced functionality"""

    def __init__(self) -> None:
        """Initialize the application"""
        if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
            raise ValueError(
                "GOOGLE_API_KEY and GOOGLE_CSE_ID must be set in environment variables"
            )

        self.search_client = GoogleSearchExtractor(
            GOOGLE_API_KEY, GOOGLE_CSE_ID
        )
        self.crawl_client = SmartContentExtractor()

    async def process_query(self, query: str, num_results: int = 5) -> None:
        """Process a search query and extract clean content"""
        console.print(f"\n[bold blue]🔍 Searching for: '{query}'[/bold blue]")

        # Step 1: Perform Google Custom Search
        search_data = self.search_client.search(query, num_results)
        if not search_data:
            return

        # Step 2: Extract and display search results
        search_results = self.search_client.extract_search_results(search_data)
        console.print(
            f"\n[bold green]📋 Found {len(search_results)} results:[/bold green]"
        )
        self.search_client.display_search_results(search_results)

        # Step 3: Extract URLs
        urls = self.search_client.extract_urls(search_data)
        if not urls:
            console.print("[yellow]No URLs found to process[/yellow]")
            return

        # Step 4: Extract content using enhanced Crawl4AI
        console.print(
            f"\n[bold blue]🕷️ Extracting clean content from {len(urls)} URLs...[/bold blue]"
        )
        extracted_content = await self.crawl_client.extract_content(urls)

        # Step 5: Display results
        self._display_extracted_content(extracted_content)

    def _display_extracted_content(
        self, content_list: List[ExtractedContent]
    ) -> None:
        """Display extracted content in a readable format"""
        console.print(
            "\n[bold green]📄 Content Extraction Results:[/bold green]"
        )

        successful_extractions = [
            c for c in content_list if c.status == ContentStatus.SUCCESS
        ]
        console.print(
            f"[dim]Successfully extracted: {len(successful_extractions)}/{len(content_list)} URLs[/dim]"
        )

        for i, content in enumerate(content_list, 1):
            self._display_single_content(content, i)

    def _display_single_content(
        self, content: ExtractedContent, index: int
    ) -> None:
        """Display a single content extraction result"""
        console.print(f"\n{'=' * 80}")
        console.print(
            f"[bold cyan]Result #{index}: {content.title}[/bold cyan]"
        )
        console.print(f"[dim]URL: {content.url}[/dim]")
        console.print(f"[dim]Status: {content.status.value}[/dim]")

        if content.strategy:
            console.print(f"[dim]Strategy: {content.strategy.value}[/dim]")
        if content.word_count > 0:
            console.print(f"[dim]Word Count: {content.word_count}[/dim]")

        console.print(f"{'=' * 80}")

        if content.status == ContentStatus.SUCCESS:
            self._display_successful_content(content, index)
        else:
            error_msg = (
                content.error_message or content.content or "Unknown error"
            )
            console.print(f"[red]{error_msg}[/red]")

    def _display_successful_content(
        self, content: ExtractedContent, index: int
    ) -> None:
        """Display successful content extraction with save option"""
        # Display content preview
        preview = content.content[:800]
        if len(content.content) > 800:
            preview += "\n... [Content truncated for display]"

        console.print(f"[white]{preview}[/white]")

        # Show content stats
        console.print(
            f"\n[dim]Content stats: {content.word_count} words, "
            f"{len(content.content)} characters[/dim]"
        )

        # Option to save full content
        save_option = Prompt.ask(
            f"\n[yellow]Save full content for Result #{index}? (y/n)[/yellow]",
            default="n",
        )

        if save_option.lower() == "y":
            self._save_content_to_file(content, index)

    def _save_content_to_file(
        self, content: ExtractedContent, index: int
    ) -> None:
        """Save content to a markdown file"""
        filename = self._generate_safe_filename(content.title, index)

        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {content.title}\n\n")
                f.write(f"**Source:** {content.url}\n")
                f.write(
                    f"**Extraction Strategy:** {content.strategy.value if content.strategy else 'unknown'}\n"
                )
                f.write(f"**Word Count:** {content.word_count}\n")
                f.write(f"**Status:** {content.status.value}\n\n")
                f.write("---\n\n")
                f.write(content.content)

            console.print(f"[green]✅ Content saved to: {filename}[/green]")
            logger.info("Content saved to file: %s", filename)

        except Exception as e:
            console.print(f"[red]❌ Error saving file: {str(e)}[/red]")
            logger.error("Error saving file %s: %s", filename, str(e))

    def _generate_safe_filename(self, title: str, index: int) -> str:
        """Generate a safe filename from title"""
        # Clean title for filename
        safe_title = re.sub(r"[^\w\s-]", "", title)
        safe_title = re.sub(r"[-\s]+", "_", safe_title)
        safe_title = safe_title[:30]  # Limit length

        filename = f"content_{index}_{safe_title}.md"
        return filename

    async def run_interactive(self) -> None:
        """Run the application in interactive mode"""
        self._display_welcome_message()

        while True:
            try:
                user_input = Prompt.ask(
                    "[bold green]Enter search query[/bold green]"
                ).strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("[yellow]👋 Goodbye![/yellow]")
                    break

                if user_input.lower() == "help":
                    self._show_help()
                    continue

                if not user_input:
                    console.print(
                        "[yellow]Please enter a search query[/yellow]"
                    )
                    continue

                # Get number of results
                num_results = self._get_num_results()
                await self.process_query(user_input, num_results)

            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {str(e)}[/red]")
                logger.error("Unexpected error in interactive mode: %s", str(e))

    def _display_welcome_message(self) -> None:
        """Display welcome message"""
        console.print(
            "[bold blue]🚀 Enhanced Google Search + Crawl4AI Content Extractor[/bold blue]"
        )
        console.print(
            "[dim]Extract clean, LLM-ready markdown content from search results[/dim]"
        )
        console.print(
            "[dim]Features: Smart content filtering, navigation removal, quality validation[/dim]"
        )
        console.print(
            "[dim]Commands: Enter search query, 'help', or 'exit'[/dim]\n"
        )

    def _get_num_results(self) -> int:
        """Get number of results from user input"""
        num_results_str = Prompt.ask(
            "[cyan]Number of results to process (1-10)[/cyan]",
            default="5",
        )

        try:
            num_results = int(num_results_str)
            return max(1, min(num_results, MAX_SEARCH_RESULTS))
        except ValueError:
            console.print("[yellow]Invalid number, using default (5)[/yellow]")
            return 5

    def _show_help(self) -> None:
        """Display help information"""
        help_text = """
[bold blue]📖 Help - Enhanced Content Extractor[/bold blue]

[bold]Features:[/bold]
• Smart content filtering with multiple extraction strategies
• Automatic navigation and UI element removal
• Content quality validation
• Clean markdown output optimized for LLMs
• Professional error handling and recovery

[bold]Usage:[/bold]
• Enter any search query to find and extract content
• Specify number of results (1-10) to process
• Content is automatically cleaned and filtered

[bold]Content Quality Features:[/bold]
• Removes navigation menus, headers, footers
• Filters out low-quality content blocks
• Validates content meaningfulness
• Multiple extraction strategies for best results

[bold]Commands:[/bold]
• [cyan]help[/cyan] - Show this help message
• [cyan]exit[/cyan] - Quit the application

[bold]Example queries:[/bold]
• "Python web scraping best practices"
• "machine learning tutorials 2025"
• "async programming patterns"
        """
        console.print(help_text)


async def main() -> None:
    """Main entry point with proper error handling"""
    try:
        app = ContentExtractorApp()
        await app.run_interactive()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Application interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Critical application error: {str(e)}[/red]")
        console.print("[dim]Please check your dependencies and try again[/dim]")
        logger.critical("Critical application error: %s", str(e))


if __name__ == "__main__":
    asyncio.run(main())
