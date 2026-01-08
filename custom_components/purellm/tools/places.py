"""Places and restaurant tool handlers."""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any, TYPE_CHECKING

from ..utils.helpers import calculate_distance_miles

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

API_TIMEOUT = 15


async def find_nearby_places(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Find nearby places using Google Places API.

    Args:
        arguments: Tool arguments (query, max_results)
        session: aiohttp session
        api_key: Google Places API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Places data dict
    """
    query = arguments.get("query", "")
    max_results = min(arguments.get("max_results", 5), 20)

    if not api_key:
        return {"error": "Google Places API key not configured. Add it in Settings → PolyVoice → API Keys."}

    if not query:
        return {"error": "No search query provided"}

    try:
        url = "https://places.googleapis.com/v1/places:searchText"

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.rating,places.currentOpeningHours"
        }

        body = {
            "textQuery": query,
            "locationBias": {
                "circle": {
                    "center": {"latitude": latitude, "longitude": longitude},
                    "radius": 10000.0
                }
            },
            "maxResultCount": max_results,
            "rankPreference": "DISTANCE"
        }

        track_api_call("places")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.post(url, json=body, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    _LOGGER.error("Google Places HTTP error: %s - %s", response.status, error_text)
                    return {"error": f"Google Places API error: {response.status}"}

                data = await response.json()

        places = data.get("places", [])

        if not places:
            return {"results": f"No results found for '{query}' near you."}

        results = []

        for idx, place in enumerate(places[:max_results], 1):
            place_name = place.get("displayName", {}).get("text", "Unknown")
            address = place.get("formattedAddress", "Address not available")
            rating = place.get("rating", "No rating")

            is_open = "Unknown hours"
            if "currentOpeningHours" in place:
                opening_hours = place.get("currentOpeningHours", {})
                is_open = "Open now" if opening_hours.get("openNow") else "Closed"

            place_lat = place.get("location", {}).get("latitude")
            place_lng = place.get("location", {}).get("longitude")

            distance_str = ""
            if place_lat and place_lng:
                miles = calculate_distance_miles(latitude, longitude, place_lat, place_lng)
                distance_str = f" ({miles:.1f} miles away)"

            result_text = f"{idx}. {place_name} - {address}{distance_str}. Rating: {rating}/5. {is_open}."
            results.append(result_text)

        response_text = f"Found {len(results)} places for '{query}' near you (sorted by distance):\n\n" + "\n".join(results)

        return {"results": response_text}

    except Exception as err:
        _LOGGER.error("Error calling Google Places API: %s", err, exc_info=True)
        return {"error": f"Failed to search for places: {str(err)}"}


async def get_restaurant_recommendations(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get restaurant recommendations from Yelp API.

    Args:
        arguments: Tool arguments (query, max_results)
        session: aiohttp session
        api_key: Yelp API key
        latitude: Search center latitude
        longitude: Search center longitude
        track_api_call: Callback to track API usage

    Returns:
        Restaurant data dict
    """
    query = arguments.get("query", "")
    max_results = min(arguments.get("max_results", 5), 10)

    if not query:
        return {"error": "No restaurant/food type specified"}

    if not api_key:
        return {"error": "Yelp API key not configured. Add it in Settings → PolyVoice → API Keys."}

    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.yelp.com/v3/businesses/search?term={encoded_query}&latitude={latitude}&longitude={longitude}&limit={max_results}&sort_by=rating"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }

        _LOGGER.info("Searching Yelp for: %s", query)

        track_api_call("restaurants")

        async with asyncio.timeout(API_TIMEOUT):
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    businesses = data.get("businesses", [])

                    if not businesses:
                        return {"message": f"No restaurants found for '{query}'"}

                    results = []
                    for biz in businesses:
                        result = {
                            "name": biz.get("name"),
                            "rating": biz.get("rating"),
                            "review_count": biz.get("review_count"),
                            "price": biz.get("price", "N/A"),
                            "address": ", ".join(biz.get("location", {}).get("display_address", [])),
                        }

                        categories = [cat.get("title") for cat in biz.get("categories", [])]
                        if categories:
                            result["cuisine"] = ", ".join(categories[:2])

                        distance_meters = biz.get("distance", 0)
                        distance_miles = distance_meters / 1609.34
                        result["distance"] = f"{distance_miles:.1f} miles"

                        if not biz.get("is_closed", True):
                            result["status"] = "Open now"

                        results.append(result)

                    _LOGGER.info("Yelp found %d restaurants", len(results))
                    return {
                        "query": query,
                        "count": len(results),
                        "restaurants": results
                    }
                else:
                    _LOGGER.error("Yelp API error: %s", response.status)
                    return {"error": f"Yelp API returned status {response.status}"}

    except Exception as err:
        _LOGGER.error("Error searching Yelp: %s", err, exc_info=True)
        return {"error": f"Failed to search restaurants: {str(err)}"}
