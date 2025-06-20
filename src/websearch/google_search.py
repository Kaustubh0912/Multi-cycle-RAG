import asyncio
import re
from dataclasses import dataclass
from enum import Enum
from typing import List

import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from ..config.settings import settings
from ..core.exceptions import (
    ContentExtractionException,
    WebSearchAPIException,
)
from ..core.interfaces import (
    WebSearchInterface,
    WebSearchResult,
    WebSearchStatus,
)
from ..utils.logging import logger


class ExtractionStrategy(Enum):
    """Content extraction strategies"""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    FALLBACK = "fallback"


@dataclass
class SearchResultData:
    """Raw search result data from Google API"""

    title: str
    url: str
    snippet: str
    rank: int


@dataclass
class ContentSelectors:
    """CSS selectors for different content extraction strategies"""

    primary: str = (
        "article, .article, .post, .content, main, .main-content, .entry-content"
    )
    secondary: str = ".post-content, .article-body, .blog-post, .story-body"
    fallback: str = 'p, .text, .description, .summary, div[class*="content"]'


class GoogleWebSearch(WebSearchInterface):
    """Google Custom Search API integration with content extraction"""

    def __init__(self):
        """Initialize Google web search"""
        self.api_key = settings.google_api_key
        self.cse_id = settings.google_cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.session = requests.Session()
        self.selectors = ContentSelectors()

        # Validate configuration
        if not self.api_key or not self.cse_id:
            logger.warning("Google Search API credentials not configured")

    async def is_available(self) -> bool:
        """Check if web search service is available"""
        return bool(self.api_key and self.cse_id)

    async def search_and_extract(
        self, query: str, num_results: int = 5
    ) -> List[WebSearchResult]:
        """Search web and extract content from results"""
        if not await self.is_available():
            logger.warning("Web search not available - missing API credentials")
            return []

        try:
            # Perform Google search
            search_results = await self._perform_search(query, num_results)
            if not search_results:
                return []

            # Extract content from the search results
            web_results = await self._extract_content_from_results(search_results)
            logger.info(
                f"Web search completed: {len(web_results)} results extracted",
                query=query,
                successful_extractions=len(
                    [r for r in web_results if r.status == WebSearchStatus.SUCCESS]
                ),
            )

            return web_results

        except Exception as e:
            logger.error(f"Web search failed: {e}", query=query)
            return []

    async def _perform_search(
        self, query: str, num_results: int
    ) -> List[SearchResultData]:
        """Perform Google Custom Search"""
        params = {
            "key": self.api_key,
            "cx": self.cse_id,
            "q": query,
            "num": min(num_results, settings.web_search_results_count),
        }

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=settings.web_search_timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Check for API errors
            if "error" in data:
                raise WebSearchAPIException(f"Google API Error: {data['error']}")

            items = data.get("items", [])
            logger.info(f"Google search returned {len(items)} results", query=query)

            # Convert to SearchResultData objects
            search_results = []
            for i, item in enumerate(items, 1):
                search_results.append(
                    SearchResultData(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        rank=i,
                    )
                )
            return search_results

        except requests.exceptions.RequestException as e:
            raise WebSearchAPIException(f"Search API request failed: {e}")
        except Exception as e:
            raise WebSearchAPIException(f"Search failed: {e}")

    async def _extract_content_from_results(
        self, search_results: List[SearchResultData]
    ) -> List[WebSearchResult]:
        """Extract content from search result URLs"""
        web_results = []

        if not settings.web_search_enable_content_extraction:
            # Return results with just snippets if content extraction is disabled
            for result in search_results:
                web_results.append(
                    WebSearchResult(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                        content=result.snippet,  # Use snippet as content
                        rank=result.rank,
                        status=WebSearchStatus.SUCCESS,
                        word_count=len(result.snippet.split()),
                        extraction_strategy="snippet_only",
                    )
                )
            return web_results

        # Extract content using Crawl4AI v0.6.x
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs concurrently but with limits
            semaphore = asyncio.Semaphore(3)  # Limit concurrent extractions

            tasks = [
                self._extract_single_url(crawler, result, semaphore)
                for result in search_results
            ]

            web_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and convert to proper results
            final_results = []
            for i, result in enumerate(web_results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Content extraction failed for {search_results[i].url}: {result}"
                    )
                    # Create fallback result with snippet
                    final_results.append(
                        WebSearchResult(
                            url=search_results[i].url,
                            title=search_results[i].title,
                            snippet=search_results[i].snippet,
                            content=search_results[i].snippet,
                            rank=search_results[i].rank,
                            status=WebSearchStatus.FAILED,
                            error_message=str(result),
                        )
                    )
                else:
                    final_results.append(result)

            return final_results

    async def _extract_single_url(
        self,
        crawler: AsyncWebCrawler,
        search_result: SearchResultData,
        semaphore: asyncio.Semaphore,
    ) -> WebSearchResult:
        """Extract content from a single URL"""
        async with semaphore:
            try:
                # Configure crawler for this URL using v0.6.x API
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS,
                    # Updated: content_filter is now part of markdown_generator
                    markdown_generator=DefaultMarkdownGenerator(
                        content_filter=PruningContentFilter(
                            threshold=0.48,
                            threshold_type="fixed",
                            min_word_threshold=settings.web_search_min_content_length,
                        )
                    ),
                    page_timeout=settings.web_search_timeout,
                    check_robots_txt=True,
                    verbose=False,
                )

                # Crawl the URL - Updated API call
                result = await crawler.arun(url=search_result.url, config=run_config)

                # Updated: Check result success properly
                if not result:
                    raise ContentExtractionException("Crawling failed")

                # Extract content using different strategies
                content = await self._extract_content_with_strategies(result)

                if (
                    not content
                    or len(content.strip()) < settings.web_search_min_content_length
                ):
                    # Use snippet as fallback
                    content = search_result.snippet
                    status = WebSearchStatus.LOW_QUALITY
                    strategy = "snippet_fallback"
                else:
                    status = WebSearchStatus.SUCCESS
                    strategy = "content_extraction"

                # Clean and truncate title
                title = self._clean_title(search_result.title)

                return WebSearchResult(
                    url=search_result.url,
                    title=title,
                    snippet=search_result.snippet,
                    content=content,
                    rank=search_result.rank,
                    status=status,
                    word_count=len(content.split()),
                    extraction_strategy=strategy,
                )

            except Exception as e:
                logger.error(f"Content extraction failed for {search_result.url}: {e}")
                return WebSearchResult(
                    url=search_result.url,
                    title=search_result.title,
                    snippet=search_result.snippet,
                    content=search_result.snippet,  # Fallback to snippet
                    rank=search_result.rank,
                    status=WebSearchStatus.ERROR,
                    error_message=str(e),
                )

    async def _extract_content_with_strategies(self, crawl_result) -> str:
        """Extract content using multiple strategies"""

        # Updated: Access markdown content properly in v0.6.x
        if hasattr(crawl_result, "markdown") and crawl_result.markdown:
            # Check if markdown has fit_markdown attribute (new structure)
            if hasattr(crawl_result.markdown, "fit_markdown"):
                content = crawl_result.markdown.fit_markdown.strip()
            else:
                content = str(crawl_result.markdown).strip()

            if len(content) >= settings.web_search_min_content_length:
                return content

        # Try to get cleaned HTML content
        if hasattr(crawl_result, "cleaned_html") and crawl_result.cleaned_html:
            content = self._html_to_text(crawl_result.cleaned_html)
            if len(content) >= settings.web_search_min_content_length:
                return content

        # Try raw HTML as last resort
        if hasattr(crawl_result, "html") and crawl_result.html:
            content = self._html_to_text(crawl_result.html)
            if len(content) >= settings.web_search_min_content_length:
                return content

        return ""

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text"""
        # Remove script and style elements
        html = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        html = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )

        # Remove HTML tags
        html = re.sub(r"<[^>]+>", "", html)

        # Clean up whitespace
        html = re.sub(r"\s+", " ", html)
        html = html.strip()

        return html

    def _clean_title(self, title: str) -> str:
        """Clean and truncate title"""
        # Remove common title suffixes
        title = re.sub(r"\s*[-|]\s*.*$", "", title)

        # Truncate if too long
        if len(title) > settings.web_search_max_title_length:
            title = (
                title[: settings.web_search_max_title_length].rsplit(" ", 1)[0] + "..."
            )

        return title.strip()
