"""Wikipedia and age calculation tool handlers."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from ..utils.helpers import get_nested

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

API_TIMEOUT = 15


async def calculate_age(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    track_api_call: callable,
) -> dict[str, Any]:
    """Calculate a person's age from Wikidata birthdate.

    Args:
        arguments: Tool arguments (person_name)
        session: aiohttp session
        track_api_call: Callback to track API usage

    Returns:
        Age data dict
    """
    person_name = arguments.get("person_name", "").strip()

    if not person_name:
        return {"error": "No person name provided"}

    try:
        track_api_call("wikipedia")

        # Step 1: Search Wikidata for the person
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={person_name}&language=en&format=json&limit=1"

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(search_url) as search_response:
                if search_response.status != 200:
                    return {"error": "Failed to search Wikidata"}
                search_data = await search_response.json()

        search_results = search_data.get("search", [])
        if not search_results:
            return {"error": f"Could not find '{person_name}' on Wikidata"}

        entity_id = search_results[0].get("id")
        entity_label = search_results[0].get("label", person_name)
        entity_description = search_results[0].get("description", "")

        # Step 2: Get entity details including birthdate
        entity_url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={entity_id}&props=claims&format=json"

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(entity_url) as entity_response:
                if entity_response.status != 200:
                    return {"error": "Failed to get entity details"}
                entity_data = await entity_response.json()

        entities = entity_data.get("entities", {})
        entity = entities.get(entity_id, {})
        claims = entity.get("claims", {})

        # P569 = date of birth in Wikidata
        birth_claims = claims.get("P569", [])
        if not birth_claims:
            return {"error": f"No birthdate found for {entity_label}"}

        birth_claim = birth_claims[0]
        time_value = get_nested(birth_claim, "mainsnak", "datavalue", "value", "time", default="")

        if not time_value:
            return {"error": f"Could not parse birthdate for {entity_label}"}

        # Parse Wikidata time format: "+1984-12-30T00:00:00Z"
        try:
            birth_date_str = time_value.lstrip("+").split("T")[0]
            birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        except ValueError:
            return {"error": f"Invalid birthdate format for {entity_label}"}

        # Calculate age
        today = datetime.now()
        age = today.year - birth_date.year

        # Adjust if birthday hasn't occurred yet this year
        if (today.month, today.day) < (birth_date.month, birth_date.day):
            age -= 1

        # Check for death date (P570)
        death_claims = claims.get("P570", [])
        is_deceased = bool(death_claims)

        result = {
            "name": entity_label,
            "age": age,
            "birth_date": birth_date.strftime("%B %d, %Y"),
            "description": entity_description,
        }

        if is_deceased:
            death_time = get_nested(death_claims[0], "mainsnak", "datavalue", "value", "time", default="")
            if death_time:
                try:
                    death_date_str = death_time.lstrip("+").split("T")[0]
                    death_date = datetime.strptime(death_date_str, "%Y-%m-%d")
                    result["death_date"] = death_date.strftime("%B %d, %Y")
                    result["age_at_death"] = age
                    result["is_deceased"] = True
                except ValueError:
                    pass

        _LOGGER.info("Age lookup: %s is %d years old (born %s)", entity_label, age, result["birth_date"])
        return result

    except Exception as err:
        _LOGGER.error("Age calculation error: %s", err, exc_info=True)
        return {"error": f"Failed to calculate age: {str(err)}"}


async def get_wikipedia_summary(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    track_api_call: callable,
) -> dict[str, Any]:
    """Get Wikipedia summary for a topic.

    Args:
        arguments: Tool arguments (topic)
        session: aiohttp session
        track_api_call: Callback to track API usage

    Returns:
        Wikipedia summary dict
    """
    topic = arguments.get("topic", "").strip()

    if not topic:
        return {"error": "No topic provided"}

    try:
        track_api_call("wikipedia")

        # Use Wikipedia REST API for summary
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(url) as response:
                if response.status == 404:
                    # Try search instead
                    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={topic}&format=json&srlimit=1"
                    async with session.get(search_url) as search_resp:
                        if search_resp.status == 200:
                            search_data = await search_resp.json()
                            results = search_data.get("query", {}).get("search", [])
                            if results:
                                new_topic = results[0].get("title", "")
                                if new_topic:
                                    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{new_topic.replace(' ', '_')}"
                                    async with session.get(url) as retry_resp:
                                        if retry_resp.status == 200:
                                            data = await retry_resp.json()
                                        else:
                                            return {"error": f"Could not find information about '{topic}'"}
                                else:
                                    return {"error": f"Could not find information about '{topic}'"}
                            else:
                                return {"error": f"Could not find information about '{topic}'"}
                        else:
                            return {"error": f"Could not find information about '{topic}'"}
                elif response.status != 200:
                    return {"error": f"Wikipedia API error: {response.status}"}
                else:
                    data = await response.json()

        title = data.get("title", topic)
        extract = data.get("extract", "No summary available")
        page_url = get_nested(data, "content_urls", "desktop", "page", default="")
        description = data.get("description", "")

        result = {
            "title": title,
            "summary": extract,
            "description": description,
        }

        if page_url:
            result["url"] = page_url

        _LOGGER.info("Wikipedia lookup: %s", title)
        return result

    except Exception as err:
        _LOGGER.error("Wikipedia lookup error: %s", err, exc_info=True)
        return {"error": f"Failed to get Wikipedia summary: {str(err)}"}
