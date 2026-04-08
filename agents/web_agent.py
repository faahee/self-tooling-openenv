"""Web Agent — Chrome-based web search, page fetching, and web scraping.

Uses Selenium with headless Chrome for full JavaScript rendering,
Google search, and robust web scraping.
"""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import re
import urllib.parse
from functools import partial
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)


class WebAgent:
    """Chrome-powered web agent for searching, fetching, and scraping.

    Capabilities:
    - Google search via headless Chrome
    - Page fetch with full JS rendering
    - Scrape tables, links, headings, prices, emails, phone numbers, images
    - Save scraped data to file
    """

    TIMEOUT = 20
    MAX_RESULTS = 5
    MAX_PAGE_CHARS = 4000

    def __init__(self, config: dict) -> None:
        self.config = config
        self._driver: webdriver.Chrome | None = None

    def _get_driver(self) -> webdriver.Chrome:
        """Create or return a reusable headless Chrome driver."""
        if self._driver is not None:
            try:
                self._driver.current_url  # Check if still alive
                return self._driver
            except Exception:
                self._driver = None

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        # Suppress logging noise
        options.add_argument("--log-level=3")
        options.add_experimental_option("excludeSwitches", ["enable-logging"])

        service = Service(ChromeDriverManager().install())
        self._driver = webdriver.Chrome(service=service, options=options)
        self._driver.set_page_load_timeout(self.TIMEOUT)
        logger.info("Chrome headless driver initialized.")
        return self._driver

    def _fetch_html(self, url: str, wait_for: str | None = None) -> str:
        """Fetch a page's HTML using Chrome (runs in thread)."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        driver = self._get_driver()
        driver.get(url)
        if wait_for:
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for))
                )
            except Exception:
                pass  # Continue with whatever loaded
        return driver.page_source

    async def _get(self, url: str, wait_for: str | None = None) -> str:
        """Async wrapper: fetch a URL via Chrome and return raw HTML."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._fetch_html, url, wait_for))

    # ── SEARCH ────────────────────────────────────────────────────────────────

    async def search(self, query: str) -> str:
        """Web search — tries DuckDuckGo HTML first, falls back to Google via Chrome."""
        # Try DuckDuckGo HTML (lightweight, no JS needed, rarely blocked)
        ddg_result = await self._search_ddg(query)
        if ddg_result:
            return ddg_result

        # Fallback: Google via headless Chrome
        return await self._search_google(query)

    async def _search_ddg(self, query: str) -> str | None:
        """DuckDuckGo HTML search — lightweight and reliable."""
        articles = await self._search_ddg_urls(query)
        if not articles:
            return None
        results: list[str] = []
        for i, art in enumerate(articles):
            line = f"{i + 1}. **{art['title']}**"
            if art["snippet"]:
                line += f"\n   {art['snippet']}"
            if art["url"]:
                line += f"\n   {art['url']}"
            results.append(line)
        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        return None

    async def _search_ddg_urls(self, query: str) -> list[dict]:
        """Search DDG and return list of {title, url, snippet} dicts with real URLs."""
        import httpx
        api_url = "https://html.duckduckgo.com/html/"
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.post(
                    api_url,
                    data={"q": query},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        ),
                    },
                )
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("DuckDuckGo search failed: %s", exc)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[dict] = []

        for result_div in soup.select("div.result")[:self.MAX_RESULTS]:
            title_tag = result_div.select_one("a.result__a")
            snippet_tag = result_div.select_one("a.result__snippet")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            href = title_tag.get("href", "")
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            real_url = self._extract_ddg_url(href)
            if not real_url:
                continue
            results.append({"title": title, "url": real_url, "snippet": snippet})
        return results

    @staticmethod
    def _extract_ddg_url(href: str) -> str | None:
        """Extract actual URL from DDG redirect/tracking link."""
        if not href:
            return None
        # DDG ad/tracking links
        if "duckduckgo.com/y.js" in href:
            return None
        # DDG redirect: //duckduckgo.com/l/?uddg=<encoded_url>
        if "duckduckgo.com" in href:
            m = re.search(r'uddg=([^&]+)', href)
            if m:
                return urllib.parse.unquote(m.group(1))
            return None
        if href.startswith("http"):
            return href
        if href.startswith("//"):
            return "https:" + href
        return None

    # ── DEEP SEARCH (search + read articles) ──────────────────────────────────

    async def search_and_read(self, query: str, max_articles: int = 3) -> str:
        """Deep search: find articles via DDG, fetch and read actual content."""
        articles = await self._search_ddg_urls(query)
        if not articles:
            return await self.search(query)

        results: list[str] = []
        for art in articles:
            if len(results) >= max_articles:
                break
            content = await self._fetch_article_text(art["url"])
            if content and len(content) > 100:
                results.append(
                    f"## {art['title']}\n"
                    f"Source: {art['url']}\n\n"
                    f"{content}"
                )

        if not results:
            # All fetches failed — fall back to snippet-only search
            return await self.search(query)

        return (
            f"Research results for '{query}' "
            f"({len(results)} articles read):\n\n"
            + "\n\n---\n\n".join(results)
        )

    async def _fetch_article_text(self, url: str, max_chars: int = 3000) -> str:
        """Fetch a single article via httpx and extract readable paragraph text."""
        import httpx
        try:
            async with httpx.AsyncClient(
                timeout=12, follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                },
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
        except Exception as exc:
            logger.debug("Article fetch failed for %s: %s", url, exc)
            return ""

        soup = BeautifulSoup(resp.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "form", "meta", "link", "svg"]):
            tag.decompose()

        main = soup.find("article") or soup.find("main") or soup.find("body")
        if not main:
            return ""

        # Prefer substantial paragraphs for quality content
        paragraphs = main.find_all("p")
        text = "\n".join(
            p.get_text(strip=True)
            for p in paragraphs
            if len(p.get_text(strip=True)) > 40
        )
        if not text:
            text = main.get_text(separator="\n", strip=True)

        # Clean up blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        if len(text) > max_chars:
            text = text[:max_chars] + "\n... (truncated)"
        return text

    async def _search_google(self, query: str) -> str:
        """Google search via headless Chrome (fallback)."""
        search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}&hl=en"
        try:
            html = await self._get(search_url, wait_for="#search")
        except Exception as exc:
            return f"Search failed: {exc}"

        soup = BeautifulSoup(html, "lxml")
        results: list[str] = []

        # Google organic results
        for i, div in enumerate(soup.select("div.g")[:self.MAX_RESULTS]):
            title_tag = div.select_one("h3")
            link_tag = div.select_one("a[href]")
            snippet_tag = div.select_one("div[data-sncf], div.VwiC3b, span.aCOpRe")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            href = link_tag["href"] if link_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            line = f"{i + 1}. **{title}**"
            if snippet:
                line += f"\n   {snippet}"
            if href:
                line += f"\n   {href}"
            results.append(line)

        if results:
            return f"Search results for '{query}':\n\n" + "\n\n".join(results)
        return f"No results found for '{query}'."

    # ── FETCH ─────────────────────────────────────────────────────────────────

    async def fetch(self, url: str) -> str:
        """Fetch a page using Chrome and return clean readable text."""
        try:
            html = await self._get(url)
        except Exception as exc:
            return f"Could not fetch {url}: {exc}"

        soup = BeautifulSoup(html, "lxml")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                         "noscript", "iframe", "form", "meta", "link"]):
            tag.decompose()

        # Get title
        title = soup.title.get_text(strip=True) if soup.title else ""

        # Extract main content — prefer article/main tags
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text("\n", strip=True)

        # Clean up blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)

        if len(text) > self.MAX_PAGE_CHARS:
            text = text[:self.MAX_PAGE_CHARS] + "\n... (truncated)"

        header = f"# {title}\n{url}\n\n" if title else f"{url}\n\n"
        return header + text if text else f"No readable content at {url}."

    # ── SCRAPE ────────────────────────────────────────────────────────────────

    async def scrape(
        self,
        url: str,
        mode: str = "text",
        save_to: str | None = None,
        css_selector: str | None = None,
        paginate: bool = False,
        max_pages: int = 3,
    ) -> str:
        """Scrape structured data from a web page.

        Args:
            url: Target URL.
            mode: What to extract — "text", "tables", "links", "headings",
                  "emails", "phones", "prices", "images", "all".
            save_to: Optional file path to save results (csv/json/txt).
            css_selector: Optional CSS selector to narrow extraction.
            paginate: Follow "next page" links automatically.
            max_pages: Max pages to scrape when paginating.

        Returns:
            Scraped data as formatted string.
        """
        all_results: list[str] = []
        current_url = url
        pages_done = 0

        while current_url and pages_done < max_pages:
            try:
                html = await self._get(current_url)
            except Exception as exc:
                return f"Scrape failed at {current_url}: {exc}"

            soup = BeautifulSoup(html, "lxml")

            # Narrow to CSS selector if provided
            root: Tag = soup
            if css_selector:
                found = soup.select(css_selector)
                if not found:
                    return f"CSS selector '{css_selector}' matched nothing on {current_url}."
                # Wrap matches in a fake div for uniform processing
                root = BeautifulSoup(
                    "<div>" + "".join(str(f) for f in found) + "</div>", "lxml"
                ).find("div")

            page_result = self._extract(root, mode, current_url)
            all_results.append(page_result)
            pages_done += 1

            if not paginate:
                break

            # Find next page link
            next_link = soup.select_one("a[rel='next'], a.next, .pagination .next a, a:contains('Next')")
            if next_link and next_link.get("href"):
                href = next_link["href"]
                current_url = href if href.startswith("http") else urllib.parse.urljoin(current_url, href)
            else:
                break

        combined = "\n\n---\n\n".join(all_results)

        # Save to file if requested
        if save_to and combined:
            saved = self._save(combined, save_to, mode, url)
            combined += f"\n\n✓ Saved to: {saved}"

        return combined

    def _extract(self, soup: Tag, mode: str, url: str) -> str:
        """Extract data from a BeautifulSoup element by mode."""
        if mode == "tables":
            return self._extract_tables(soup, url)
        elif mode == "links":
            return self._extract_links(soup, url)
        elif mode == "headings":
            return self._extract_headings(soup)
        elif mode == "emails":
            return self._extract_pattern(soup, r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "emails")
        elif mode == "phones":
            return self._extract_pattern(soup, r"[\+\(]?[0-9][0-9\s\-\(\)\.]{7,}[0-9]", "phone numbers")
        elif mode == "prices":
            return self._extract_pattern(soup, r"[$€£¥₹]\s?[\d,]+(?:\.\d{2})?", "prices")
        elif mode == "images":
            return self._extract_images(soup, url)
        elif mode == "all":
            parts = []
            for m in ("headings", "tables", "links", "emails"):
                r = self._extract(soup, m, url)
                if r and "Nothing found" not in r:
                    parts.append(r)
            return "\n\n".join(parts) if parts else "Nothing found."
        else:  # "text" default
            return self._extract_text(soup)

    def _extract_text(self, soup: Tag) -> str:
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [l for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        if len(text) > self.MAX_PAGE_CHARS:
            text = text[:self.MAX_PAGE_CHARS] + "\n..."
        return text or "No text content found."

    def _extract_tables(self, soup: Tag, url: str) -> str:
        tables = soup.find_all("table")
        if not tables:
            return "No tables found on the page."

        parts: list[str] = []
        for i, table in enumerate(tables[:10], 1):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if not rows:
                continue

            # Format as aligned table
            col_widths = [max(len(str(r[c])) for r in rows if c < len(r)) for c in range(max(len(r) for r in rows))]
            lines = []
            for j, row in enumerate(rows):
                padded = [str(cell).ljust(col_widths[k]) for k, cell in enumerate(row) if k < len(col_widths)]
                lines.append(" | ".join(padded))
                if j == 0:
                    lines.append("-+-".join("-" * w for w in col_widths))
            parts.append(f"Table {i}:\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "Tables found but no data extracted."

    def _extract_links(self, soup: Tag, base_url: str) -> str:
        seen = set()
        links: list[str] = []
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(strip=True)
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            full = href if href.startswith("http") else urllib.parse.urljoin(base_url, href)
            if full not in seen:
                seen.add(full)
                links.append(f"  {text or '(no text)'}  ->  {full}")
        if links:
            return f"Links ({len(links)} found):\n" + "\n".join(links[:50])
        return "No links found."

    def _extract_headings(self, soup: Tag) -> str:
        headings: list[str] = []
        for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            text = tag.get_text(strip=True)
            if text:
                level = int(tag.name[1])
                headings.append("  " * (level - 1) + f"[{tag.name.upper()}] {text}")
        return ("Page structure:\n" + "\n".join(headings)) if headings else "No headings found."

    def _extract_pattern(self, soup: Tag, pattern: str, label: str) -> str:
        text = soup.get_text(" ", strip=True)
        matches = list(dict.fromkeys(re.findall(pattern, text)))
        if matches:
            return f"Found {len(matches)} {label}:\n" + "\n".join(f"  {m}" for m in matches[:50])
        return f"No {label} found."

    def _extract_images(self, soup: Tag, base_url: str) -> str:
        imgs: list[str] = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            alt = img.get("alt", "")
            full = src if src.startswith("http") else urllib.parse.urljoin(base_url, src)
            imgs.append(f"  {alt or '(no alt)'}  ->  {full}")
        if imgs:
            return f"Images ({len(imgs)} found):\n" + "\n".join(imgs[:30])
        return "No images found."

    def _save(self, content: str, path: str, mode: str, url: str) -> str:
        """Save scraped content to a file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()

        try:
            if ext == ".json":
                data = {"url": url, "mode": mode, "content": content}
                p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            elif ext == ".csv" and mode == "tables":
                # Best-effort CSV from table text
                reader = csv.reader(io.StringIO(content))
                p.write_text(content, encoding="utf-8")
            else:
                p.write_text(content, encoding="utf-8")
        except Exception as exc:
            logger.error("Save failed: %s", exc)
            return f"(save failed: {exc})"

        return str(p.resolve())

    # ── SITE STATUS CHECK ─────────────────────────────────────────────────────

    async def _check_site_status(self, domain: str) -> str:
        """Check if a website is up by sending an HTTP HEAD request."""
        import httpx
        target_url = f"https://{domain}" if not domain.startswith("http") else domain
        try:
            async with httpx.AsyncClient(
                timeout=10, follow_redirects=True,
                headers={"User-Agent": "JARVIS/1.0"},
            ) as client:
                resp = await client.head(target_url)
                status = resp.status_code
                if 200 <= status < 400:
                    return f"✅ **{domain}** is UP (HTTP {status} OK)."
                else:
                    return f"⚠️ **{domain}** returned HTTP {status}."
        except Exception as exc:
            return f"❌ **{domain}** appears to be DOWN or unreachable.\nError: {exc}"

    # ── NATURAL LANGUAGE HANDLER ──────────────────────────────────────────────

    async def handle(self, user_input: str, context: dict) -> str:
        """Route to search, fetch, or scrape based on user input."""
        nl = user_input.lower().strip()

        # Detect scraping intent
        scrape_keywords = ("scrape", "extract", "crawl", "harvest", "collect",
                           "get all", "grab all", "pull data", "get data from",
                           "get links", "get emails", "get tables", "get prices")
        is_scrape = any(kw in nl for kw in scrape_keywords)

        # Extract URL
        url_match = re.search(r"https?://\S+", user_input)
        url = url_match.group(0).rstrip(".,\"'") if url_match else None

        # Also match plain domains
        if not url:
            domain_match = re.search(
                r"\b((?:www\.)?[\w-]+\.(?:com|org|net|io|dev|ai|co|gov|edu)(?:/\S*)?)\b", nl
            )
            if domain_match:
                url = domain_match.group(1)

        if is_scrape and url:
            # Determine scrape mode from user input
            mode = "text"
            if any(w in nl for w in ("table", "tables", "grid")):
                mode = "tables"
            elif any(w in nl for w in ("link", "links", "url", "urls")):
                mode = "links"
            elif any(w in nl for w in ("email", "emails", "mail")):
                mode = "emails"
            elif any(w in nl for w in ("price", "prices", "cost")):
                mode = "prices"
            elif any(w in nl for w in ("heading", "headings", "structure", "outline")):
                mode = "headings"
            elif any(w in nl for w in ("image", "images", "photo", "photos")):
                mode = "images"
            elif any(w in nl for w in ("phone", "phones", "number", "contact")):
                mode = "phones"
            elif any(w in nl for w in ("all", "everything")):
                mode = "all"

            # Detect save-to file
            save_match = re.search(r"(?:save|export|write|store)\s+(?:to\s+)?([^\s]+\.(?:csv|json|txt))", nl)
            save_to = save_match.group(1) if save_match else None

            # Detect CSS selector
            selector_match = re.search(r'selector[:\s]+["\']?([^"\']+)["\']?', nl)
            css_selector = selector_match.group(1).strip() if selector_match else None

            # Detect pagination
            paginate = any(w in nl for w in ("all pages", "paginate", "next page", "multiple pages"))

            return await self.scrape(
                url=url,
                mode=mode,
                save_to=save_to,
                css_selector=css_selector,
                paginate=paginate,
            )

        # Site status check ("is X up", "is X down", "check if X is online")
        status_match = re.search(
            r"(?:is\s+)([\w.-]+\.(?:com|org|net|io|dev|ai|co|gov|edu))\s+(?:up|down|online|alive|working|running|accessible)",
            nl
        )
        if not status_match:
            status_match = re.search(
                r"(?:check if|check whether|ping)\s+([\w.-]+\.(?:com|org|net|io|dev|ai|co|gov|edu))",
                nl
            )
        if status_match:
            target = status_match.group(1)
            return await self._check_site_status(target)

        # Direct URL fetch (non-scraping)
        if url and not is_scrape:
            return await self.fetch(url)

        # Scrape intent without URL — fall back to search
        if is_scrape and not url:
            # Extract the topic from scrape command and search for it instead
            scrape_topic = re.sub(
                r"(?:scrape|extract|crawl|harvest|collect|get all|grab all|pull data|get data from)\s*",
                "", nl
            ).strip()
            if scrape_topic:
                return await self.search(scrape_topic)

        # Deep research — fetch and read actual article content
        research_match = re.search(
            r"(?:research the web for|research online for|deep search for|"
            r"read articles about|find and read about|research about|research)\s+(.+)",
            nl
        )
        if research_match:
            rquery = research_match.group(1).strip().rstrip("?.")
            return await self.search_and_read(rquery)

        # Web search — extract the core query from natural language
        query = user_input.strip()
        patterns = [
            r"(?:search the web for|search online for|web search for)\s+(.+)",
            r"(?:search for|google|look up|find info(?:rmation)? (?:on|about))\s+(.+)",
            r"(?:news about|latest news (?:on|about)|recent news (?:on|about)|breaking news (?:on|about))\s+(.+)",
            r"(?:what(?:'s| is| are| happened)(?: the| to| with| in)?)\s+(.+)",
            r"(?:tell me about|explain|who is|who are)\s+(.+)",
            r"(?:latest|recent|current)\s+(.+)",
            r"(?:search)\s+(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, nl)
            if m:
                query = m.group(1).strip().rstrip("?.")
                break

        # For news queries, append "latest" to get current results (but not if already temporal)
        news_triggers = ("news", "war", "attack", "election", "crisis", "conflict",
                         "score", "price", "weather", "happened", "update")
        already_temporal = ("latest", "today", "now", "current", "recent", "this week", "this month")
        if (any(t in nl for t in news_triggers)
                and not any(t in query.lower() for t in already_temporal)):
            query = f"latest {query}"

        return await self.search(query)

    def close(self) -> None:
        """Shut down the Chrome driver if running."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
            logger.info("Chrome driver closed.")
