"""Device control, status, and history tool handlers."""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

from ..utils.fuzzy_matching import find_entity_by_name
from ..utils.helpers import format_human_readable_state, get_friendly_name

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def check_device_status(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    device_aliases: dict[str, str],
    user_query: str = "",
    temp_format_func: callable = None,
) -> dict[str, Any]:
    """Check the current status of any device.

    Args:
        arguments: Tool arguments (device)
        hass: Home Assistant instance
        device_aliases: Custom device name -> entity_id mapping
        user_query: Original user query for better extraction
        temp_format_func: Function to format temperature values

    Returns:
        Device status dict
    """
    device = arguments.get("device", "").strip()

    # Extract device name from original query using patterns
    extracted_device = None
    original_query = user_query.lower()

    patterns = [
        r"(?:what(?:'s| is) the |status of (?:the )?|check (?:the )?|is (?:the )?)([a-z ]+?)(?:\s+(?:status|open|closed|locked|unlocked|on|off|state)|\?|$)",
        r"(?:is |are )(?:the )?([a-z ]+?)(?:\s+(?:open|closed|locked|unlocked|on|off)|\?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, original_query)
        if match:
            extracted_device = match.group(1).strip()
            for suffix in [' status', ' state', ' currently', ' right now']:
                if extracted_device.endswith(suffix):
                    extracted_device = extracted_device[:-len(suffix)].strip()
            break

    if extracted_device and len(extracted_device) > len(device):
        _LOGGER.info("Device extraction: LLM said '%s', extracted '%s'", device, extracted_device)
        device = extracted_device

    if not device:
        return {"error": "No device specified. Please specify a device name like 'front door', 'garage', etc."}

    entity_id, friendly_name = find_entity_by_name(hass, device, device_aliases)

    if not entity_id:
        return {"error": f"Could not find a device matching '{device}'. Try using the exact name as shown in Home Assistant."}

    state = hass.states.get(entity_id)
    if not state:
        return {"error": f"Entity '{entity_id}' not found"}

    domain = entity_id.split(".")[0]
    current_state = state.state

    # Handle climate domain specially (thermostat)
    if domain == "climate":
        target_temp = state.attributes.get("temperature")
        current_temp = state.attributes.get("current_temperature")
        hvac_mode = state.attributes.get("hvac_mode", current_state)

        status_parts = []
        if target_temp:
            formatted_temp = temp_format_func(target_temp) if temp_format_func else f"{target_temp}°"
            status_parts.append(f"set to {formatted_temp}")
        if current_temp:
            formatted_temp = temp_format_func(current_temp) if temp_format_func else f"{current_temp}°"
            status_parts.append(f"currently {formatted_temp}")
        if hvac_mode:
            status_parts.append(f"mode: {hvac_mode}")

        status = ", ".join(status_parts) if status_parts else current_state.upper()

        _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)

        return {
            "device": friendly_name,
            "status": status,
            "target_temperature": target_temp,
            "current_temperature": current_temp,
            "hvac_mode": hvac_mode,
            "entity_id": entity_id
        }

    # Handle sensors with units
    if domain == "sensor":
        unit = state.attributes.get("unit_of_measurement", "")
        if unit:
            status = f"{current_state} {unit}"
        else:
            status = current_state

        _LOGGER.info("Device status check: %s -> %s (%s) status=%s", device, friendly_name, entity_id, status)

        return {
            "device": friendly_name,
            "status": status,
            "entity_id": entity_id
        }

    # Use helper for other domains
    status = format_human_readable_state(entity_id, current_state)

    _LOGGER.info("Device status check: %s -> %s (%s) domain=%s raw_state=%s status=%s",
                 device, friendly_name, entity_id, domain, current_state, status)

    return {
        "device": friendly_name,
        "status": status,
        "entity_id": entity_id
    }


async def get_device_history(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    device_aliases: dict[str, str],
    hass_timezone,
    user_query: str = "",
) -> dict[str, Any]:
    """Get historical state changes from HA Recorder.

    Args:
        arguments: Tool arguments (device, days_back, date)
        hass: Home Assistant instance
        device_aliases: Custom device name -> entity_id mapping
        hass_timezone: Home Assistant timezone
        user_query: Original user query for better extraction

    Returns:
        Device history dict
    """
    from homeassistant.util import dt as dt_util

    device = arguments.get("device", "").strip()
    days_back = min(arguments.get("days_back", 1), 10)
    specific_date = arguments.get("date", "")

    # Extract device name from original query
    original_query = user_query.lower()
    patterns = [
        r"(?:history (?:of |for )?(?:the )?|when (?:was |did )(?:the )?|how many times (?:was |did )(?:the )?)([a-z ]+?)(?:\s+(?:open|closed|locked|unlocked|today|yesterday|last|this)|\?|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, original_query)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) > len(device):
                _LOGGER.info("History device extraction: LLM said '%s', extracted '%s'", device, extracted)
                device = extracted
            break

    if not device:
        return {"error": "No device specified. Please specify a device name like 'front door', 'garage', etc."}

    entity_id, friendly_name = find_entity_by_name(hass, device, device_aliases)

    if not entity_id:
        return {"error": f"Could not find a device matching '{device}'. Try using the exact name as shown in Home Assistant."}

    try:
        current_state = hass.states.get(entity_id)
        if not current_state:
            return {"error": f"Entity '{entity_id}' not found"}

        friendly_name = get_friendly_name(entity_id, current_state)

        now = datetime.now(hass_timezone)

        if specific_date:
            try:
                target_date = datetime.strptime(specific_date, "%Y-%m-%d")
                start_time = target_date.replace(hour=0, minute=0, second=0, tzinfo=hass_timezone)
                end_time = target_date.replace(hour=23, minute=59, second=59, tzinfo=hass_timezone)
                period_desc = target_date.strftime("%B %d, %Y")
            except ValueError:
                return {"error": f"Invalid date format: {specific_date}. Use YYYY-MM-DD"}
        else:
            end_time = now
            start_time = now - timedelta(days=days_back)
            if days_back == 1:
                period_desc = "today"
            else:
                period_desc = f"last {days_back} days"

        from homeassistant.components.recorder import get_instance
        from homeassistant.components.recorder.history import get_significant_states

        _LOGGER.info("Fetching history for %s from %s to %s", entity_id, start_time, end_time)

        history_data = await get_instance(hass).async_add_executor_job(
            get_significant_states,
            hass,
            start_time.astimezone(),
            end_time.astimezone(),
            [entity_id],
        )

        state_changes = []
        last_on = None
        last_off = None
        on_count = 0
        off_count = 0

        domain = entity_id.split(".")[0]

        # Determine "on" and "off" states based on domain
        if domain == "lock":
            on_state, off_state = "unlocked", "locked"
            on_label, off_label = "unlocked", "locked"
        elif domain == "binary_sensor":
            on_state, off_state = "on", "off"
            if "door" in entity_id or "gate" in entity_id or "mailbox" in entity_id:
                on_label, off_label = "opened", "closed"
            else:
                on_label, off_label = "detected", "clear"
        elif domain in ("light", "switch", "fan"):
            on_state, off_state = "on", "off"
            on_label, off_label = "turned on", "turned off"
        else:
            on_state, off_state = "on", "off"
            on_label, off_label = "on", "off"

        if entity_id in history_data:
            for state in history_data[entity_id]:
                if state.state in ("unavailable", "unknown"):
                    continue

                try:
                    state_time = state.last_changed.astimezone(hass_timezone)
                    time_str = state_time.strftime("%B %d at %I:%M %p")

                    if state.state == on_state:
                        on_count += 1
                        last_on = time_str
                        state_changes.append({"action": on_label, "time": time_str})
                    elif state.state == off_state:
                        off_count += 1
                        last_off = time_str
                        state_changes.append({"action": off_label, "time": time_str})
                except Exception as parse_err:
                    _LOGGER.warning("Error parsing state time: %s", parse_err)

        result = {
            "device": friendly_name,
            "entity_id": entity_id,
            "period": period_desc,
            "total_changes": len(state_changes),
        }

        if last_on:
            result[f"last_{on_label.replace(' ', '_')}"] = last_on
        if last_off:
            result[f"last_{off_label.replace(' ', '_')}"] = last_off

        result[f"{on_label.replace(' ', '_')}_count"] = on_count
        result[f"{off_label.replace(' ', '_')}_count"] = off_count

        if state_changes:
            result["recent_activity"] = state_changes[-10:][::-1]
        else:
            result["message"] = f"No state changes found for {period_desc}"

        _LOGGER.info("Device history result: %s", result)
        return result

    except ImportError as ie:
        _LOGGER.error("Recorder import error: %s", ie)
        return {"error": "History component not available"}
    except Exception as err:
        _LOGGER.error("Error getting device history: %s", err, exc_info=True)
        return {"error": f"Failed to get device history: {str(err)}"}


async def control_device(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    device_aliases: dict[str, str],
) -> dict[str, Any]:
    """Control smart home devices.

    This is the main device control handler that supports:
    - Lights (with brightness, color)
    - Switches
    - Covers/blinds (with position, presets)
    - Locks
    - Fans
    - Media players
    - Climate
    - Vacuums
    - And more...

    Args:
        arguments: Tool arguments
        hass: Home Assistant instance
        device_aliases: Custom device name -> entity_id mapping

    Returns:
        Control result dict
    """
    from homeassistant.helpers import entity_registry as er
    from homeassistant.helpers import area_registry as ar
    from homeassistant.helpers import device_registry as dr

    action = arguments.get("action", "").strip().lower()
    brightness = arguments.get("brightness")
    position = arguments.get("position")
    color = arguments.get("color", "").strip().lower()
    color_temp = arguments.get("color_temp")
    volume = arguments.get("volume")
    temperature = arguments.get("temperature")
    hvac_mode = arguments.get("hvac_mode", "").strip().lower()
    fan_speed = arguments.get("fan_speed", "").strip().lower()

    direct_entity_id = arguments.get("entity_id", "").strip()
    entity_ids_list = arguments.get("entity_ids", [])
    area_name = arguments.get("area", "").strip()
    domain_filter = arguments.get("domain", "").strip().lower()
    device_name = arguments.get("device", "").strip()

    _LOGGER.debug("control_device: entity=%s, device=%s, area=%s, action=%s",
                  direct_entity_id or entity_ids_list, device_name, area_name, action)

    if not action:
        return {"error": "No action specified."}

    # Normalize actions
    action_aliases = {"favorite": "preset", "return_home": "dock", "activate": "turn_on"}
    action = action_aliases.get(action, action)

    # Service map
    service_map = {
        "light": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "switch": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "fan": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle", "set_speed": "set_percentage"},
        "lock": {"lock": "lock", "unlock": "unlock", "turn_on": "lock", "turn_off": "unlock"},
        "cover": {
            "open": "open_cover", "close": "close_cover", "toggle": "toggle",
            "turn_on": "open_cover", "turn_off": "close_cover",
            "stop": "stop_cover", "set_position": "set_cover_position", "preset": "set_cover_position"
        },
        "climate": {"turn_on": "turn_on", "turn_off": "turn_off", "set_temperature": "set_temperature", "set_hvac_mode": "set_hvac_mode"},
        "media_player": {
            "turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle",
            "play": "media_play", "pause": "media_pause", "stop": "media_stop",
            "next": "media_next_track", "previous": "media_previous_track",
            "volume_up": "volume_up", "volume_down": "volume_down",
            "set_volume": "volume_set", "mute": "volume_mute", "unmute": "volume_mute"
        },
        "vacuum": {"turn_on": "start", "start": "start", "turn_off": "return_to_base", "stop": "stop", "dock": "return_to_base", "locate": "locate"},
        "scene": {"turn_on": "turn_on"},
        "script": {"turn_on": "turn_on", "turn_off": "turn_off"},
        "input_boolean": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "automation": {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"},
        "button": {"turn_on": "press", "press": "press"},
        "siren": {"turn_on": "turn_on", "turn_off": "turn_off"},
        "humidifier": {"turn_on": "turn_on", "turn_off": "turn_off"},
    }

    action_words = {
        "turn_on": "turned on", "turn_off": "turned off", "toggle": "toggled",
        "lock": "locked", "unlock": "unlocked",
        "open_cover": "opened", "close_cover": "closed", "stop_cover": "stopped",
        "set_cover_position": "set position for",
        "start": "started", "return_to_base": "sent home", "stop": "stopped",
        "locate": "located", "press": "pressed",
        "media_play": "playing", "media_pause": "paused", "media_stop": "stopped",
        "media_next_track": "skipped to next", "media_previous_track": "went back to previous",
        "volume_up": "turned up", "volume_down": "turned down",
        "volume_set": "set volume for", "volume_mute": "muted/unmuted",
        "set_temperature": "set temperature for", "set_hvac_mode": "set mode for",
    }

    color_map = {
        "red": [255, 0, 0], "green": [0, 255, 0], "blue": [0, 0, 255],
        "yellow": [255, 255, 0], "orange": [255, 165, 0], "purple": [128, 0, 128],
        "pink": [255, 192, 203], "white": [255, 255, 255], "cyan": [0, 255, 255],
        "warm": None, "cool": None,
    }

    entities_to_control = []

    # Method 1: Direct entity_id
    if direct_entity_id:
        state = hass.states.get(direct_entity_id)
        if state:
            friendly_name = state.attributes.get("friendly_name", direct_entity_id)
            entities_to_control.append((direct_entity_id, friendly_name))
        else:
            return {"error": f"Entity '{direct_entity_id}' not found in Home Assistant."}

    # Method 2: Multiple entity_ids
    elif entity_ids_list:
        for eid in entity_ids_list:
            eid = eid.strip()
            state = hass.states.get(eid)
            if state:
                friendly_name = state.attributes.get("friendly_name", eid)
                entities_to_control.append((eid, friendly_name))
            else:
                _LOGGER.warning("Entity %s not found, skipping", eid)

    # Method 3: Area-based control
    elif area_name:
        ent_reg = er.async_get(hass)
        area_reg = ar.async_get(hass)
        dev_reg = dr.async_get(hass)

        target_area_id = None
        for area in area_reg.async_list_areas():
            if area.name.lower() == area_name.lower():
                target_area_id = area.id
                break

        if not target_area_id:
            for area in area_reg.async_list_areas():
                if area_name.lower() in area.name.lower() or area.name.lower() in area_name.lower():
                    target_area_id = area.id
                    break

        if not target_area_id:
            return {"error": f"Could not find area '{area_name}'."}

        device_areas = {device.id: True for device in dev_reg.devices.values() if device.area_id == target_area_id}

        controllable_domains = ["light", "switch", "fan", "lock", "cover", "media_player", "vacuum", "scene", "script", "input_boolean"]
        if domain_filter and domain_filter != "all":
            controllable_domains = [domain_filter]

        for state in hass.states.async_all():
            eid = state.entity_id
            domain = eid.split(".")[0]

            if domain not in controllable_domains:
                continue
            if state.state in ("unavailable", "unknown"):
                continue

            entity_entry = ent_reg.async_get(eid)
            if not entity_entry:
                continue

            in_area = entity_entry.area_id == target_area_id or (entity_entry.device_id and entity_entry.device_id in device_areas)

            if in_area:
                friendly_name = state.attributes.get("friendly_name", eid)
                entities_to_control.append((eid, friendly_name))

        if not entities_to_control:
            return {"error": f"No controllable devices found in area '{area_name}'."}

    # Method 4: Device name matching (uses fuzzy matching with aliases)
    elif device_name:
        found_entity_id, friendly_name = find_entity_by_name(hass, device_name, device_aliases)

        if found_entity_id:
            entities_to_control.append((found_entity_id, friendly_name))
        else:
            return {"error": f"Could not find a device matching '{device_name}'."}

    else:
        return {"error": "No device specified. Provide entity_id, entity_ids, area, or device name."}

    # Build service calls first, then execute in parallel
    service_calls: list[tuple[str, str, dict, str]] = []  # (domain, service, data, friendly_name)
    failed = []
    last_service = None

    for entity_id, friendly_name in entities_to_control:
        domain = entity_id.split(".")[0]
        domain_services = service_map.get(domain, {"turn_on": "turn_on", "turn_off": "turn_off", "toggle": "toggle"})
        service = domain_services.get(action)

        if not service:
            failed.append(f"{friendly_name} (unsupported action)")
            continue

        service_data = {"entity_id": entity_id}

        # Light controls
        if domain == "light" and action == "turn_on":
            if brightness is not None:
                service_data["brightness_pct"] = max(0, min(100, brightness))
            if color and color in color_map and color_map[color]:
                service_data["rgb_color"] = color_map[color]
            elif color == "warm":
                service_data["color_temp_kelvin"] = 2700
            elif color == "cool":
                service_data["color_temp_kelvin"] = 6500
            if color_temp is not None:
                service_data["color_temp_kelvin"] = max(2000, min(6500, color_temp))

        # Media player controls
        if domain == "media_player":
            if action == "set_volume" and volume is not None:
                service_data["volume_level"] = max(0, min(100, volume)) / 100.0
            if action == "mute":
                service_data["is_volume_muted"] = True
            if action == "unmute":
                service_data["is_volume_muted"] = False

        # Climate controls
        if domain == "climate":
            if action == "set_temperature" and temperature is not None:
                service_data["temperature"] = temperature
            if hvac_mode:
                if action == "set_hvac_mode" or (action == "turn_on" and hvac_mode):
                    service = "set_hvac_mode"
                    service_data["hvac_mode"] = hvac_mode

        # Fan controls
        if domain == "fan" and fan_speed:
            speed_map = {"low": 33, "medium": 66, "high": 100, "auto": 50}
            if fan_speed in speed_map:
                service_data["percentage"] = speed_map[fan_speed]

        # Cover position
        if domain == "cover" and action == "set_position" and position is not None:
            service_data["position"] = max(0, min(100, position))

        # Cover preset/favorite - try button.{name}_my_position first
        if domain == "cover" and action == "preset":
            cover_object_id = entity_id.split(".")[1]
            my_position_btn = f"button.{cover_object_id}_my_position"

            if hass.states.get(my_position_btn):
                service_calls.append(("button", "press", {"entity_id": my_position_btn}, friendly_name))
                last_service = service
                continue
            else:
                # Fall back to set_cover_position
                state = hass.states.get(entity_id)
                preset_pos = state.attributes.get("preset_position") if state else None
                if preset_pos is None and state:
                    preset_pos = state.attributes.get("favorite_position")
                service_data["position"] = preset_pos if preset_pos is not None else 50

        service_calls.append((domain, service, service_data, friendly_name))
        last_service = service

    # Execute all service calls in parallel
    async def execute_call(call_info: tuple[str, str, dict, str]) -> tuple[str, Exception | None]:
        domain, service, data, name = call_info
        try:
            await hass.services.async_call(domain, service, data, blocking=False)
            _LOGGER.info("Device control: %s.%s on %s", domain, service, name)
            return (name, None)
        except Exception as err:
            _LOGGER.error("Error controlling device %s: %s", name, err)
            return (name, err)

    if service_calls:
        results = await asyncio.gather(*[execute_call(call) for call in service_calls])
        controlled = [name for name, err in results if err is None]
        failed.extend([f"{name} ({str(err)[:30]})" for name, err in results if err is not None])
    else:
        controlled = []

    service = last_service  # For response generation

    # Build response
    if controlled:
        if len(controlled) == 1:
            if action == "preset":
                response = f"I've set the {controlled[0]} to its favorite position."
            elif action == "set_position" and position is not None:
                response = f"I've set the {controlled[0]} to {position}% position."
            elif brightness is not None and action == "turn_on":
                response = f"I've turned on the {controlled[0]} at {brightness}% brightness."
            else:
                action_word = action_words.get(service, action)
                response = f"I've {action_word} the {controlled[0]}."
        else:
            if action == "preset":
                response = f"I've set {len(controlled)} devices to their favorite positions: {', '.join(controlled[:5])}"
            elif action == "set_position" and position is not None:
                response = f"I've set {len(controlled)} devices to {position}%: {', '.join(controlled[:5])}"
            else:
                action_word = action_words.get(service, action)
                response = f"I've {action_word} {len(controlled)} devices: {', '.join(controlled[:5])}"
            if len(controlled) > 5:
                response += f" and {len(controlled) - 5} more"
            response += "."

        result = {
            "success": True,
            "controlled_count": len(controlled),
            "controlled_devices": controlled,
            "response_text": response
        }

        if failed:
            result["failed_count"] = len(failed)
            result["failed_devices"] = failed

        return result
    else:
        return {"error": f"Failed to control any devices. Failures: {', '.join(failed)}"}
