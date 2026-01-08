"""News tool handler."""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp
    from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

API_TIMEOUT = 15

# Valid categories for TheNewsAPI
VALID_CATEGORIES = frozenset([
    "general", "science", "sports", "business", "health",
    "entertainment", "tech", "politics", "food", "travel"
])

# Map common aliases to valid categories
CATEGORY_MAP = {
    "technology": "tech",
    "world": "general",
    "nation": "politics",
}

# Categories that should be treated as topic searches
TOPIC_CATEGORIES = frozenset([
    "fashion", "gaming", "crypto", "cryptocurrency", "auto",
    "automotive", "cars", "real estate", "weather", "local"
])


async def get_news(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    hass_timezone,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get news from TheNewsAPI.com.

    Args:
        arguments: Tool arguments (category, topic, max_results)
        session: aiohttp session
        api_key: TheNewsAPI key
        hass_timezone: Home Assistant timezone
        track_api_call: Callback to track API usage

    Returns:
        News data dict
    """
    if not api_key:
        return {"error": "TheNewsAPI key not configured. Add it in Settings → PolyVoice → API Keys."}

    category = arguments.get("category", "")
    topic = arguments.get("topic", "")
    max_results = min(arguments.get("max_results", 5), 25)

    try:
        base_url = "https://api.thenewsapi.com/v1/news"
        fetch_limit = 25

        if topic:
            encoded_topic = urllib.parse.quote(topic)
            url = f"{base_url}/all?api_token={api_key}&search={encoded_topic}&language=en&limit={fetch_limit}"
            display_topic = topic
        elif category:
            mapped_category = CATEGORY_MAP.get(category.lower(), category.lower())

            if mapped_category in VALID_CATEGORIES:
                url = f"{base_url}/top?api_token={api_key}&locale=us&language=en&categories={mapped_category}&limit={fetch_limit}"
                display_topic = f"{category.title()} News"
            elif mapped_category in TOPIC_CATEGORIES or mapped_category not in VALID_CATEGORIES:
                encoded_topic = urllib.parse.quote(mapped_category)
                url = f"{base_url}/all?api_token={api_key}&search={encoded_topic}&language=en&limit={fetch_limit}"
                display_topic = f"{category.title()} News"
            else:
                url = f"{base_url}/top?api_token={api_key}&locale=us&language=en&limit={fetch_limit}"
                display_topic = "Top Headlines"
        else:
            url = f"{base_url}/top?api_token={api_key}&locale=us&language=en&limit={fetch_limit}"
            display_topic = "Top Headlines"

        _LOGGER.info("Fetching news from TheNewsAPI: %s", display_topic)

        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }

        track_api_call("news")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("data", [])

                    if not articles:
                        return {"message": f"No news found for '{display_topic}'"}

                    headlines = []
                    seen_titles = set()
                    blocked_sources = {"yahoo.com", "finance.yahoo.com"}
                    blocked_count = 0

                    for article in articles:
                        if len(headlines) >= max_results:
                            break

                        source_name = article.get("source", "Unknown")
                        title = article.get("title", "No title")

                        if source_name in blocked_sources:
                            blocked_count += 1
                            continue

                        if title in seen_titles:
                            continue
                        seen_titles.add(title)

                        description = article.get("description", "")
                        snippet = article.get("snippet", "")
                        published_at = article.get("published_at", "")

                        summary = description if description else snippet
                        if summary and len(summary) > 200:
                            summary = summary[:200] + "..."

                        date_text = ""
                        if published_at:
                            try:
                                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                                dt_local = dt.astimezone(hass_timezone)
                                date_text = dt_local.strftime("%B %d at %I:%M %p")
                            except (ValueError, KeyError, TypeError, AttributeError):
                                date_text = published_at

                        headlines.append({
                            "headline": title,
                            "summary": summary,
                            "source": source_name,
                            "published": date_text
                        })

                    result = {
                        "topic": display_topic,
                        "article_count": len(headlines),
                        "articles": headlines
                    }

                    _LOGGER.info("TheNewsAPI: %d returned, %d blocked, %d passed", len(articles), blocked_count, len(headlines))
                    return result

                elif response.status == 401:
                    return {"error": "Invalid TheNewsAPI key. Check your API token."}
                elif response.status == 402:
                    return {"error": "TheNewsAPI usage limit reached."}
                elif response.status == 429:
                    return {"error": "TheNewsAPI rate limit exceeded. Try again later."}
                else:
                    error_data = await response.text()
                    _LOGGER.error("TheNewsAPI error: %s - %s", response.status, error_data)
                    return {"error": f"Failed to fetch news: HTTP {response.status}"}

    except Exception as err:
        _LOGGER.error("Error getting news: %s", err, exc_info=True)
        return {"error": f"Failed to get news: {str(err)}"}
