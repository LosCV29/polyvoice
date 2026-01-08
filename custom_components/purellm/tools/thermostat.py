"""Thermostat tool handler."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def control_thermostat(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    thermostat_entity: str,
    temp_step: int,
    min_temp: int,
    max_temp: int,
    format_temp_func: callable,
) -> dict[str, Any]:
    """Control or check the thermostat.

    Args:
        arguments: Tool arguments (action, temperature)
        hass: Home Assistant instance
        thermostat_entity: Climate entity ID
        temp_step: Temperature step for raise/lower
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature
        format_temp_func: Function to format temperature display

    Returns:
        Thermostat control result dict
    """
    action = arguments.get("action", "").lower()
    temp_arg = arguments.get("temperature")

    if action not in ["raise", "lower", "set", "check"]:
        return {"error": "Invalid action. Use 'raise', 'lower', 'set', or 'check'"}

    try:
        thermostat = hass.states.get(thermostat_entity)
        if not thermostat:
            return {"error": "Thermostat not found"}

        current_target = thermostat.attributes.get("temperature")
        current_temp = thermostat.attributes.get("current_temperature")
        hvac_mode = thermostat.attributes.get("hvac_mode", thermostat.state)

        if current_target is None:
            current_target = 72

        # Handle check action
        if action == "check":
            response_text = (
                f"The thermostat is set to {hvac_mode} with a target temperature of "
                f"{format_temp_func(current_target)}. The current temperature in the home is "
                f"{format_temp_func(current_temp)}."
            )
            return {"response_text": response_text}

        # Calculate new temperature
        if action == "set":
            if temp_arg is None:
                return {"error": "Please specify a temperature to set"}
            try:
                temp_value = float(temp_arg)
                if not (-50 <= temp_value <= 150):
                    return {"error": "Temperature must be between -50 and 150"}
                new_temp = int(temp_value)
            except (ValueError, TypeError):
                return {"error": "Invalid temperature value"}
        elif action == "raise":
            new_temp = int(current_target + temp_step)
        else:  # lower
            new_temp = int(current_target - temp_step)

        # Clamp to allowed range
        new_temp = max(min_temp, min(max_temp, new_temp))

        _LOGGER.info("Thermostat control: action=%s, current=%s, new=%s", action, current_target, new_temp)

        await hass.services.async_call(
            "climate",
            "set_temperature",
            {"entity_id": thermostat_entity, "temperature": new_temp},
            blocking=True
        )

        # Build response
        if action == "set":
            response_text = f"I've set the thermostat to {format_temp_func(new_temp)}."
        elif action == "raise":
            response_text = f"I've raised the thermostat to {format_temp_func(new_temp)}."
        else:
            response_text = f"I've lowered the thermostat to {format_temp_func(new_temp)}."

        return {"response_text": response_text}

    except Exception as err:
        _LOGGER.error("Error controlling thermostat: %s", err, exc_info=True)
        return {"error": f"Failed to control thermostat: {str(err)}"}
