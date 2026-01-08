"""Weather tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp

_LOGGER = logging.getLogger(__name__)

# API timeout in seconds
API_TIMEOUT = 15


async def get_weather_forecast(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    api_key: str,
    latitude: float,
    longitude: float,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get weather forecast from OpenWeatherMap.

    Args:
        arguments: Tool arguments (location, forecast_type)
        session: aiohttp session
        api_key: OpenWeatherMap API key
        latitude: Default latitude
        longitude: Default longitude
        track_api_call: Callback to track API usage

    Returns:
        Weather data dict
    """
    forecast_type = arguments.get("forecast_type", "both")
    location_query = arguments.get("location", "").strip()

    if not api_key:
        return {"error": "OpenWeatherMap API key not configured. Add it in Settings → PolyVoice → API Keys."}

    location_name = None

    # If user specified a location, geocode it
    if location_query:
        try:
            geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_query}&limit=1&appid={api_key}"
            async with session.get(geo_url) as geo_response:
                if geo_response.status == 200:
                    geo_data = await geo_response.json()
                    if geo_data and len(geo_data) > 0:
                        latitude = geo_data[0]["lat"]
                        longitude = geo_data[0]["lon"]
                        location_name = geo_data[0].get("name", location_query)
                        if geo_data[0].get("state"):
                            location_name += f", {geo_data[0]['state']}"
                        if geo_data[0].get("country"):
                            location_name += f", {geo_data[0]['country']}"
                        _LOGGER.info("Geocoded '%s' to %s (%s, %s)", location_query, location_name, latitude, longitude)
                    else:
                        return {"error": f"Could not find location: {location_query}"}
                else:
                    return {"error": f"Geocoding failed for: {location_query}"}
        except Exception as geo_err:
            _LOGGER.error("Geocoding error: %s", geo_err)
            return {"error": f"Could not geocode location: {location_query}"}

    try:
        result = {}
        track_api_call("weather")

        async with asyncio.timeout(API_TIMEOUT):
            # PARALLEL fetch: current weather AND forecast simultaneously
            current_url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"
            forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={latitude}&lon={longitude}&appid={api_key}&units=imperial"

            # Fire both requests at once
            current_task = session.get(current_url)
            forecast_task = session.get(forecast_url)

            async with current_task as current_response, forecast_task as forecast_response:
                # Process current weather
                if current_response.status == 200:
                    data = await current_response.json()

                    result["current"] = {
                        "temperature": round(data["main"]["temp"]),
                        "feels_like": round(data["main"]["feels_like"]),
                        "humidity": data["main"]["humidity"],
                        "conditions": data["weather"][0]["description"].title(),
                        "wind_speed": round(data["wind"]["speed"]),
                        "location": location_name or data["name"]
                    }

                    # Add rain if present
                    if "rain" in data:
                        result["current"]["rain_1h"] = data["rain"].get("1h", 0)

                    _LOGGER.info("Current weather: %s", result["current"])
                else:
                    _LOGGER.error("Weather API error: %s", current_response.status)
                    return {"error": "Could not get current weather"}

                # Process forecast data
                if forecast_response.status == 200:
                    data = await forecast_response.json()

                    # Get NEXT HOUR rain chance from first forecast entry
                    next_hour_rain = 0
                    if data["list"] and len(data["list"]) > 0:
                        next_hour_rain = round(data["list"][0].get("pop", 0) * 100)
                    result["current"]["rain_chance_next_hour"] = next_hour_rain

                    # Calculate AVERAGE rain chance for next 8 hours
                    rain_chances_8hr = []
                    for i, item in enumerate(data["list"][:3]):
                        rain_chances_8hr.append(item.get("pop", 0) * 100)
                    if rain_chances_8hr:
                        result["current"]["avg_rain_chance_8hr"] = round(sum(rain_chances_8hr) / len(rain_chances_8hr))
                    else:
                        result["current"]["avg_rain_chance_8hr"] = 0

                    # Get TODAY's date for extracting today's high/low
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    today_temps = []

                    # Group by day for weekly forecast
                    daily_forecasts = {}
                    for item in data["list"]:
                        dt = datetime.strptime(item["dt_txt"], "%Y-%m-%d %H:%M:%S")
                        day_key = dt.strftime("%A")
                        item_date = dt.strftime("%Y-%m-%d")

                        # Collect today's temps
                        if item_date == today_str:
                            today_temps.append(item["main"]["temp_max"])
                            today_temps.append(item["main"]["temp_min"])

                        if day_key not in daily_forecasts:
                            daily_forecasts[day_key] = {
                                "date": dt.strftime("%B %d"),
                                "high": item["main"]["temp_max"],
                                "low": item["main"]["temp_min"],
                                "conditions": item["weather"][0]["description"].title(),
                                "rain_chance": item.get("pop", 0) * 100
                            }
                        else:
                            daily_forecasts[day_key]["high"] = max(
                                daily_forecasts[day_key]["high"],
                                item["main"]["temp_max"]
                            )
                            daily_forecasts[day_key]["low"] = min(
                                daily_forecasts[day_key]["low"],
                                item["main"]["temp_min"]
                            )
                            # Take noon conditions if available
                            if dt.hour == 12:
                                daily_forecasts[day_key]["conditions"] = item["weather"][0]["description"].title()
                                daily_forecasts[day_key]["rain_chance"] = item.get("pop", 0) * 100

                    # ADD TODAY'S HIGH/LOW TO CURRENT
                    current_day = datetime.now().strftime("%A")
                    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%A")

                    if current_day in daily_forecasts:
                        result["current"]["todays_high"] = round(daily_forecasts[current_day]["high"])
                        if tomorrow in daily_forecasts:
                            result["current"]["todays_low"] = round(daily_forecasts[tomorrow]["low"])
                        else:
                            result["current"]["todays_low"] = round(daily_forecasts[current_day]["low"])
                    elif today_temps:
                        result["current"]["todays_high"] = round(max(today_temps))
                        result["current"]["todays_low"] = round(min(today_temps))
                    else:
                        first_day = list(daily_forecasts.values())[0] if daily_forecasts else None
                        if first_day:
                            result["current"]["todays_high"] = round(first_day["high"])
                            result["current"]["todays_low"] = round(first_day["low"])

                    # Format weekly forecast (only if requested)
                    if forecast_type in ["weekly", "both"]:
                        forecast_list = []
                        for day, forecast in list(daily_forecasts.items())[:5]:
                            forecast_list.append({
                                "day": day,
                                "date": forecast["date"],
                                "high": round(forecast["high"]),
                                "low": round(forecast["low"]),
                                "conditions": forecast["conditions"],
                                "rain_chance": round(forecast["rain_chance"])
                            })

                        result["forecast"] = forecast_list
                        _LOGGER.info("Weather forecast: %d days", len(forecast_list))

        if not result:
            return {"error": "No weather data retrieved"}

        return result

    except Exception as err:
        _LOGGER.error("Error getting weather: %s", err, exc_info=True)
        return {"error": f"Failed to get weather: {str(err)}"}
