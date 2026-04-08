"""API Agent — Real-time data fetcher using free, keyless public APIs.

Routes natural-language queries to the correct fetcher and returns
structured dicts with source attribution and timestamps.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any

import httpx

from core.llm_core import TaskType

logger = logging.getLogger(__name__)

# ── WMO Weather Interpretation Codes (WW) ────────────────────────────────────
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# ── Coin-name aliases ─────────────────────────────────────────────────────────
_COIN_ALIASES: dict[str, str] = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "cardano": "cardano",
    "ada": "cardano",
    "xrp": "ripple",
    "ripple": "ripple",
    "bnb": "binancecoin",
    "binance": "binancecoin",
}

# ── Known API descriptions for listing ────────────────────────────────────────
_KNOWN_APIS: dict[str, str] = {
    "weather": "Current weather via Open-Meteo (free, no key)",
    "crypto": "Crypto prices via CoinGecko (free)",
    "currency": "Exchange rates via Frankfurter (free)",
    "news": "Headlines via BBC RSS (free)",
    "wikipedia": "Summaries via Wikipedia REST API (free)",
    "ip": "Public IP info via ipinfo.io (free)",
    "joke": "Random joke via official-joke-api (free)",
    "github": "GitHub repo info via GitHub API (free)",
}


class APIAgent:
    """Routes natural-language queries to free public APIs.

    Every response is a structured dict with ``source`` and ``fetched_at``
    fields so the LLM never has to guess where data came from.
    """

    def __init__(self, config: dict, llm_core: Any, ui_layer: Any) -> None:
        """Initialise the API agent.

        Args:
            config: Full JARVIS configuration dictionary.
            llm_core: LLM core instance (used only for fallback).
            ui_layer: UI layer instance.
        """
        self.config = config
        self.llm = llm_core
        self.ui = ui_layer
        self._custom_apis: dict[str, dict[str, str]] = {}
        self._client = httpx.AsyncClient(timeout=10.0, follow_redirects=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  Public entry point (signature expected by orchestrator)
    # ══════════════════════════════════════════════════════════════════════════

    async def handle(self, user_input: str, context: dict) -> str:
        """Route a user query to the right real-time fetcher.

        Args:
            user_input: Natural-language query.
            context: Context dict from the orchestrator.

        Returns:
            Formatted string with source attribution.
        """
        low = user_input.lower()

        # ── Meta commands (register / list) ───────────────────────────────
        if any(w in low for w in ("register api", "add api", "save api", "define api")):
            return self._register_api(user_input)
        if any(w in low for w in ("list api", "show api", "what apis")):
            return self._list_apis()

        # ── Route to real-time fetchers ───────────────────────────────────
        result = await self.handle_realtime_query(user_input)
        fields_filter = _detect_weather_filter(low)
        return _format_result(result, fields_filter)

    # ══════════════════════════════════════════════════════════════════════════
    #  Master router
    # ══════════════════════════════════════════════════════════════════════════

    async def handle_realtime_query(self, query: str) -> dict[str, Any]:
        """Detect query type by keywords and route to the correct fetcher.

        Args:
            query: Raw user query string.

        Returns:
            Structured dict with data, ``source``, and ``fetched_at``.
        """
        low = query.lower()

        # Priority 1 — IP address (most specific, fewest false positives)
        if "my ip" in low or "public ip" in low or "ip address" in low:
            return await self.get_my_ip()

        # Priority 2 — Weather
        if any(w in low for w in ("weather", "temperature", "forecast", "rain", "humidity", "wind")):
            city = _extract_location(query)
            return await self.get_weather(city)

        # Priority 3 — Crypto
        if any(w in low for w in ("bitcoin", "crypto", "ethereum", "btc", "eth",
                                   "coin", "solana", "doge", "dogecoin", "xrp", "bnb")):
            return await self.get_crypto_price(query)

        # Priority 4 — Currency
        if any(w in low for w in ("currency", "exchange rate", "usd", "inr", "eur", "convert")):
            return await self.get_exchange_rate(query)

        # Priority 5 — News
        if any(w in low for w in ("news", "headline", "latest news", "top stories")):
            return await self.get_news_headlines()

        # Priority 6 — Joke
        if "joke" in low:
            return await self.get_joke()

        # Priority 7 — GitHub
        if "github" in low and ("repo" in low or "/" in query):
            return await self._github(query)

        # Priority 8 — Wikipedia (LAST — "what is" and "who is" overlap with everything)
        if any(w in low for w in ("wikipedia", "wiki")):
            topic = _extract_topic(query)
            return await self.get_wikipedia_summary(topic)

        # Direct URL
        url_match = re.search(r'https?://[^\s]+', query)
        if url_match:
            return await self._call_url(url_match.group(0), query)

        # Custom registered APIs
        for name, api in self._custom_apis.items():
            if name in low:
                return await self._call_url(api.get("url", ""), query)

        # LLM fallback
        return await self._llm_api_call(query)

    # ══════════════════════════════════════════════════════════════════════════
    #  Fetcher functions — every one returns a dict, never raises
    # ══════════════════════════════════════════════════════════════════════════

    async def get_weather(self, city: str) -> dict[str, Any]:
        """Fetch current weather for *city* via Open-Meteo (free, no key).

        Args:
            city: City name (e.g. ``"Chennai"``).

        Returns:
            Dict with temperature, humidity, wind, condition, source, fetched_at.
        """
        source = "Open-Meteo"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            # Step 1 — geocode the city name
            geo_resp = await self._client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en"},
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            results = geo_data.get("results")
            if not results:
                return {"error": f"City '{city}' not found", "source": source, "fetched_at": now}

            lat = results[0]["latitude"]
            lon = results[0]["longitude"]
            resolved_name = results[0].get("name", city)
            country = results[0].get("country", "")

            # Step 2 — fetch current weather
            weather_resp = await self._client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,apparent_temperature,"
                        "relative_humidity_2m,weather_code,"
                        "wind_speed_10m,precipitation"
                    ),
                },
            )
            weather_resp.raise_for_status()
            current = weather_resp.json().get("current", {})

            weather_code = current.get("weather_code", -1)
            condition = _WMO_CODES.get(weather_code, f"Unknown (code {weather_code})")

            return {
                "city": f"{resolved_name}, {country}",
                "temperature": f"{current.get('temperature_2m')}°C",
                "feels_like": f"{current.get('apparent_temperature')}°C",
                "humidity": f"{current.get('relative_humidity_2m')}%",
                "wind_speed": f"{current.get('wind_speed_10m')} km/h",
                "precipitation": f"{current.get('precipitation')} mm",
                "condition": condition,
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("Weather fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_crypto_price(self, query: str) -> dict[str, Any]:
        """Fetch cryptocurrency price via CoinGecko (free, no key).

        Args:
            query: User query containing a coin name/ticker.

        Returns:
            Dict with coin, usd, inr, source, fetched_at.
        """
        source = "CoinGecko"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            coin_id = "bitcoin"
            low = query.lower()
            for alias, cid in _COIN_ALIASES.items():
                if alias in low:
                    coin_id = cid
                    break

            resp = await self._client.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd,inr"},
            )
            resp.raise_for_status()
            data = resp.json()
            prices = data.get(coin_id, {})
            return {
                "coin": coin_id.title(),
                "usd": prices.get("usd", "N/A"),
                "inr": prices.get("inr", "N/A"),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("Crypto fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_exchange_rate(self, query: str) -> dict[str, Any]:
        """Fetch currency exchange rates via Frankfurter (free, no key).

        Args:
            query: User query (e.g. ``"USD to INR"``).

        Returns:
            Dict with base, target, rate, source, fetched_at.
        """
        source = "Frankfurter (ECB)"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            base, target = _extract_currencies(query)
            resp = await self._client.get(
                "https://api.frankfurter.app/latest",
                params={"from": base, "to": target},
            )
            resp.raise_for_status()
            data = resp.json()
            rate = data.get("rates", {}).get(target, "N/A")
            return {
                "base": base,
                "target": target,
                "rate": rate,
                "date": data.get("date", ""),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("Exchange rate fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_news_headlines(self) -> dict[str, Any]:
        """Fetch top news headlines from BBC RSS (free, no key).

        Returns:
            Dict with headlines list, source, fetched_at.
        """
        source = "BBC News RSS"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            resp = await self._client.get("https://feeds.bbci.co.uk/news/rss.xml")
            resp.raise_for_status()
            root = ET.fromstring(resp.text)
            items = root.findall(".//item")[:10]
            headlines: list[dict[str, str]] = []
            for item in items:
                title_el = item.find("title")
                desc_el = item.find("description")
                headlines.append({
                    "title": title_el.text if title_el is not None else "",
                    "description": desc_el.text if desc_el is not None else "",
                })
            return {"headlines": headlines, "source": source, "fetched_at": now}
        except Exception as e:
            logger.error("News fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_wikipedia_summary(self, topic: str) -> dict[str, Any]:
        """Fetch a Wikipedia summary via the REST API (free, no key).

        Args:
            topic: The topic to look up.

        Returns:
            Dict with title, summary, url, source, fetched_at.
        """
        source = "Wikipedia"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            safe_topic = topic.strip().replace(" ", "_")
            resp = await self._client.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}",
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "title": data.get("title", topic),
                "summary": data.get("extract", "No summary available."),
                "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("Wikipedia fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_my_ip(self) -> dict[str, Any]:
        """Fetch public IP information via ipinfo.io (free, no key).

        Returns:
            Dict with ip, city, region, country, isp, source, fetched_at.
        """
        source = "ipinfo.io"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            resp = await self._client.get("https://ipinfo.io/json")
            resp.raise_for_status()
            data = resp.json()
            return {
                "ip": data.get("ip"),
                "city": data.get("city"),
                "region": data.get("region"),
                "country": data.get("country"),
                "isp": data.get("org"),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("IP fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    async def get_joke(self) -> dict[str, Any]:
        """Fetch a random joke via official-joke-api (free, no key).

        Returns:
            Dict with setup, punchline, source, fetched_at.
        """
        source = "official-joke-api"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            resp = await self._client.get(
                "https://official-joke-api.appspot.com/random_joke"
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "setup": data.get("setup", ""),
                "punchline": data.get("punchline", ""),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("Joke fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    # ── GitHub (kept from original) ───────────────────────────────────────────

    async def _github(self, user_input: str) -> dict[str, Any]:
        """Fetch GitHub repository info (free, no key).

        Args:
            user_input: Query containing ``owner/repo``.

        Returns:
            Dict with repo details, source, fetched_at.
        """
        source = "GitHub API"
        now = datetime.now().isoformat(timespec="seconds")
        m = re.search(r'(\w[\w-]*)/(\w[\w-]*)', user_input)
        if not m:
            return {
                "error": "Could not parse owner/repo. Try: 'github info for torvalds/linux'",
                "source": source,
                "fetched_at": now,
            }
        owner, repo = m.group(1), m.group(2)
        try:
            resp = await self._client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={"Accept": "application/vnd.github.v3+json"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "repo": data.get("full_name"),
                "description": data.get("description"),
                "stars": data.get("stargazers_count"),
                "forks": data.get("forks_count"),
                "language": data.get("language"),
                "open_issues": data.get("open_issues_count"),
                "url": data.get("html_url"),
                "source": source,
                "fetched_at": now,
            }
        except Exception as e:
            logger.error("GitHub fetch failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    # ── Direct URL call ───────────────────────────────────────────────────────

    async def _call_url(self, url: str, user_input: str) -> dict[str, Any]:
        """Call an arbitrary URL and return the response.

        Args:
            url: The URL to call.
            user_input: Original user query (used to detect POST).

        Returns:
            Dict with response body, source, fetched_at.
        """
        source = url
        now = datetime.now().isoformat(timespec="seconds")
        low = user_input.lower()
        method = "POST" if "post" in low else "GET"

        body: dict | None = None
        body_match = re.search(r'(?:body|data|payload)[:\s]+(\{[^}]+\})', user_input, re.I)
        if body_match:
            try:
                body = json.loads(body_match.group(1))
            except Exception:
                body = None

        headers: dict[str, str] = {}
        token_match = re.search(r'(?:auth|bearer|token)[:\s]+([A-Za-z0-9_.\-]+)', user_input, re.I)
        if token_match:
            headers["Authorization"] = f"Bearer {token_match.group(1)}"

        try:
            if method == "POST":
                resp = await self._client.post(url, json=body, headers=headers)
            else:
                resp = await self._client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "json" in content_type:
                data = resp.json()
                return {"response": data, "source": source, "fetched_at": now}
            return {"response": resp.text[:2000], "source": source, "fetched_at": now}
        except Exception as e:
            logger.error("URL call failed: %s", e)
            return {"error": str(e), "source": source, "fetched_at": now}

    # ── Custom API registry ───────────────────────────────────────────────────

    def _register_api(self, user_input: str) -> str:
        """Register a named API endpoint.

        Args:
            user_input: Command like ``register api myapi url https://...``.

        Returns:
            Confirmation string.
        """
        name_match = re.search(r'api\s+(\w+)', user_input, re.I)
        url_match = re.search(r'https?://[^\s]+', user_input)
        if not name_match or not url_match:
            return "Usage: register api <name> url <url>"
        name = name_match.group(1).lower()
        url = url_match.group(0)
        self._custom_apis[name] = {"url": url}
        return f"API '{name}' registered at {url}"

    def _list_apis(self) -> str:
        """List all known and custom APIs.

        Returns:
            Formatted list string.
        """
        lines: list[str] = ["Available APIs:"]
        for name, desc in _KNOWN_APIS.items():
            lines.append(f"  {name}: {desc}")
        for name, info in self._custom_apis.items():
            lines.append(f"  {name}: {info.get('url', '')}")
        return "\n".join(lines)

    # ── LLM fallback ─────────────────────────────────────────────────────────

    async def _llm_api_call(self, user_input: str) -> dict[str, Any]:
        """Fall back to LLM when no known API matches.

        Args:
            user_input: The original user query.

        Returns:
            Dict with LLM suggestion, source, fetched_at.
        """
        source = "LLM suggestion (no live data)"
        now = datetime.now().isoformat(timespec="seconds")
        try:
            prompt = (
                f"The user wants to make an API call:\n{user_input}\n\n"
                f"Known free APIs: {list(_KNOWN_APIS.keys())}\n\n"
                "Suggest which API to call or give the curl command. Be concise."
            )
            response = await self.llm.generate(prompt, task_type=TaskType.FORMAT_RESULT)
            return {"suggestion": response, "source": source, "fetched_at": now}
        except Exception as e:
            return {"error": str(e), "source": source, "fetched_at": now}


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ══════════════════════════════════════════════════════════════════════════════


def _detect_weather_filter(low: str) -> list[str] | None:
    """Detect which weather fields the user cares about.

    Returns a list of field keys to include, or None to show everything.
    Only activates for narrow queries — full 'weather' queries always get all fields.

    Args:
        low: Lowercase user input.

    Returns:
        List of field keys to show, or None for full weather block.
    """
    # Full weather block — don't filter
    full_triggers = ("weather", "forecast", "all", "full", "everything", "conditions")
    if any(w in low for w in full_triggers):
        return None

    # Temperature only
    temp_triggers = ("temperature", "how hot", "how cold", "degrees", "celsius", "fahrenheit", "temp")
    if any(w in low for w in temp_triggers):
        return ["city", "temperature", "feels_like", "condition"]

    # Rain / precipitation only
    rain_triggers = ("rain", "precipitation", "raining", "drizzle", "shower", "wet", "umbrella")
    if any(w in low for w in rain_triggers):
        return ["city", "precipitation", "condition"]

    # Humidity only
    if "humidity" in low:
        return ["city", "humidity", "condition"]

    # Wind only
    if "wind" in low:
        return ["city", "wind_speed", "condition"]

    return None


def _format_result(result: dict[str, Any], fields_filter: list[str] | None = None) -> str:
    """Format any result dict into a human-readable string with source line.

    Args:
        result: Dict returned by any fetcher function.
        fields_filter: Optional list of field keys to show (weather only). None = all.

    Returns:
        Multi-line display string ending with source attribution.
    """
    source = result.get("source", "Unknown")
    fetched = result.get("fetched_at", "N/A")

    if "error" in result:
        return f"⚠️ Error: {result['error']}\n📡 Source: {source} | Fetched: {fetched}"

    lines: list[str] = []
    skip = {"source", "fetched_at"}

    # Special-case formatters
    if "headlines" in result:
        lines.append("📰 Top Headlines:")
        for i, h in enumerate(result["headlines"][:10], 1):
            lines.append(f"  {i}. {h.get('title', '')}")
            if h.get("description"):
                lines.append(f"     {h['description'][:120]}")
    elif "setup" in result and "punchline" in result:
        lines.append(f"😄 {result['setup']}")
        lines.append(f"   {result['punchline']}")
    elif "suggestion" in result:
        lines.append(str(result["suggestion"]))
    elif "response" in result:
        val = result["response"]
        if isinstance(val, dict):
            lines.append(json.dumps(val, indent=2)[:2000])
        else:
            lines.append(str(val)[:2000])
    else:
        # Apply field filter for weather-type results
        keys_to_show = fields_filter if (fields_filter and "temperature" in result) else None
        for key, value in result.items():
            if key in skip:
                continue
            if keys_to_show and key not in keys_to_show:
                continue
            label = key.replace("_", " ").title()
            lines.append(f"{label}: {value}")

    lines.append(f"\n📡 Source: {source} | Fetched: {fetched}")
    return "\n".join(lines)


def _extract_location(query: str) -> str:
    """Extract a city name from a weather query.

    Args:
        query: Natural-language weather query.

    Returns:
        City name string (defaults to ``"London"``).
    """
    # Try "in/for/at/of <city>" — word boundary prevents matching "at" inside "what"
    m = re.search(r'\b(?:in|for|at|of)\s+([a-zA-Z][a-zA-Z ]{0,30}?)(?:\s*[\?.,!]|\s*$)', query, re.I)
    if m:
        candidate = m.group(1).strip()
        # Filter out noise words that aren't cities
        _NOISE = {"today", "now", "the", "this", "my", "our", "current", "right",
                  "weather", "temperature", "forecast", "morning", "evening",
                  "afternoon", "night", "tomorrow", "yesterday",
                  "is", "are", "was", "will", "be", "it", "its",
                  "in", "for", "at", "of", "on", "and", "or",
                  "what", "how", "whats", "hows", "can", "you", "me",
                  "tell", "show", "get", "check", "please"}
        words = [w for w in candidate.split() if w.lower() not in _NOISE]
        if words:
            return " ".join(words)

    # Fallback — find capitalized words that aren't common English
    _COMMON = {"what", "how", "is", "the", "weather", "today", "tell", "me",
               "in", "for", "at", "of", "show", "get", "check", "current",
               "now", "right", "please", "can", "you", "whats", "hows",
               "temperature", "forecast", "rain", "humidity", "wind"}
    words = query.split()
    for w in reversed(words):
        clean = w.strip("?,!.")
        if clean and clean[0].isupper() and clean.lower() not in _COMMON and len(clean) > 2:
            return clean

    # Last resort — take the last word that isn't a noise word
    for w in reversed(words):
        clean = w.strip("?,!.")
        if clean.lower() not in _COMMON and clean.isalpha() and len(clean) > 2:
            return clean

    return "Chennai"


def _extract_topic(query: str) -> str:
    """Extract a Wikipedia topic from a query.

    Args:
        query: Natural-language query like ``"who is Alan Turing"``.

    Returns:
        Topic string.
    """
    for prefix in ("who is ", "what is ", "tell me about ", "wikipedia ", "wiki "):
        idx = query.lower().find(prefix)
        if idx != -1:
            return query[idx + len(prefix):].strip().rstrip("?.")
    # Fallback — drop common leading words
    cleaned = re.sub(r'^(search|look up|find)\s+', '', query, flags=re.I)
    return cleaned.strip().rstrip("?.")


def _extract_currencies(query: str) -> tuple[str, str]:
    """Extract base and target currency codes from a query.

    Args:
        query: Query like ``"USD to INR"`` or ``"convert euros to dollars"``.

    Returns:
        Tuple of (base, target) ISO-4217 codes. Defaults to ``("USD", "INR")``.
    """
    _CURRENCY_NAMES: dict[str, str] = {
        "dollar": "USD", "dollars": "USD", "usd": "USD",
        "rupee": "INR", "rupees": "INR", "inr": "INR",
        "euro": "EUR", "euros": "EUR", "eur": "EUR",
        "pound": "GBP", "pounds": "GBP", "gbp": "GBP",
        "yen": "JPY", "jpy": "JPY",
        "yuan": "CNY", "cny": "CNY", "rmb": "CNY",
        "aud": "AUD", "cad": "CAD", "chf": "CHF",
    }
    low = query.lower()
    # Try "X to Y" pattern
    m = re.search(r'(\w+)\s+to\s+(\w+)', low)
    if m:
        base = _CURRENCY_NAMES.get(m.group(1), m.group(1).upper())
        target = _CURRENCY_NAMES.get(m.group(2), m.group(2).upper())
        return base, target
    # Collect any mentioned currencies
    found: list[str] = []
    for word in low.split():
        if word in _CURRENCY_NAMES:
            found.append(_CURRENCY_NAMES[word])
    if len(found) >= 2:
        return found[0], found[1]
    if len(found) == 1:
        return "USD", found[0]
    return "USD", "INR"
