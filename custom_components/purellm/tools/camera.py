"""Camera tool handlers."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Default camera friendly names
DEFAULT_CAMERA_FRIENDLY_NAMES = {
    "porch": "Front Porch",
    "driveway": "Driveway",
    "garage": "Garage",
    "backyard": "Backyard",
    "kitchen": "Kitchen",
    "living_room": "Living Room",
    "front_door": "Front Door",
}


async def check_camera(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    camera_friendly_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Check a camera with AI vision analysis.

    Args:
        arguments: Tool arguments (location, query)
        hass: Home Assistant instance
        camera_friendly_names: Custom camera name mappings

    Returns:
        Camera analysis dict
    """
    location = arguments.get("location", "").lower().strip()
    query = arguments.get("query", "")

    if not location:
        return {"error": "No camera location specified"}

    friendly_names = camera_friendly_names or DEFAULT_CAMERA_FRIENDLY_NAMES
    friendly_name = friendly_names.get(location, location.replace("_", " ").title())

    try:
        service_data = {
            "camera": location,
            "duration": 3,
        }
        if query:
            service_data["user_query"] = query

        result = await hass.services.async_call(
            "ha_video_vision",
            "analyze_camera",
            service_data,
            blocking=True,
            return_response=True,
        )

        if not result or not result.get("success"):
            error_msg = result.get('error', 'Unknown error') if result else 'Service unavailable'
            return {
                "location": friendly_name,
                "status": "unavailable",
                "error": f"Could not access {friendly_name} camera: {error_msg}"
            }

        analysis = result.get("description", "Unable to analyze camera feed")

        return {
            "location": friendly_name,
            "status": "checked",
            "description": analysis
        }

    except Exception as err:
        _LOGGER.error("Error checking camera %s: %s", location, err, exc_info=True)
        return {
            "location": friendly_name,
            "status": "error",
            "error": f"Failed to check {friendly_name} camera: {str(err)}"
        }


async def quick_camera_check(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    camera_friendly_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Fast camera check - just person detection + one sentence.

    Args:
        arguments: Tool arguments (location)
        hass: Home Assistant instance
        camera_friendly_names: Custom camera name mappings

    Returns:
        Brief camera check dict
    """
    location = arguments.get("location", "").lower().strip()

    if not location:
        return {"error": "No camera location specified"}

    friendly_names = camera_friendly_names or DEFAULT_CAMERA_FRIENDLY_NAMES
    friendly_name = friendly_names.get(location, location.replace("_", " ").title())

    try:
        result = await hass.services.async_call(
            "ha_video_vision",
            "analyze_camera",
            {"camera": location, "duration": 2},
            blocking=True,
            return_response=True,
        )

        if not result or not result.get("success"):
            return {"location": friendly_name, "error": "Camera unavailable"}

        analysis = result.get("description", "")
        brief = analysis.split('.')[0] + '.' if analysis else "No activity."

        return {
            "location": friendly_name,
            "brief": brief
        }

    except Exception as err:
        _LOGGER.error("Error quick-checking camera %s: %s", location, err)
        return {"location": friendly_name, "error": "Check failed"}
