"""Timer tool handler for PolyVoice - COMPREHENSIVE EDITION.

Supports:
- Natural language: "half an hour", "2 and a half hours", "45 mins", "one minute"
- Named timers: "pizza timer", "laundry timer"
- Multiple concurrent timers
- Add time to existing timers: "add 5 minutes"
- Restart timers with same duration
- Fuzzy name matching
- TTS announcements when timers finish
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# Storage key for tracking PolyVoice-started timers
POLYVOICE_TIMERS_KEY = "polyvoice_active_timers"


def register_timer(
    hass: "HomeAssistant",
    entity_id: str,
    friendly_name: str,
    announce_player: str | None = None
) -> None:
    """Register a timer started by PolyVoice for finish notifications.

    Args:
        hass: Home Assistant instance
        entity_id: Timer entity ID
        friendly_name: Human-readable timer name
        announce_player: Media player to announce on when timer finishes
    """
    if POLYVOICE_TIMERS_KEY not in hass.data:
        hass.data[POLYVOICE_TIMERS_KEY] = {}
    hass.data[POLYVOICE_TIMERS_KEY][entity_id] = {
        "name": friendly_name,
        "announce_player": announce_player,
    }
    _LOGGER.debug("Registered PolyVoice timer: %s (%s) -> announce on %s",
                  entity_id, friendly_name, announce_player or "default")


def unregister_timer(hass: "HomeAssistant", entity_id: str) -> dict | None:
    """Unregister a timer and return its info if it was ours."""
    if POLYVOICE_TIMERS_KEY in hass.data:
        return hass.data[POLYVOICE_TIMERS_KEY].pop(entity_id, None)
    return None


def get_registered_timer(hass: "HomeAssistant", entity_id: str) -> dict | None:
    """Get the info of a registered timer.

    Returns:
        Dict with 'name' and 'announce_player' keys, or None
    """
    if POLYVOICE_TIMERS_KEY in hass.data:
        return hass.data[POLYVOICE_TIMERS_KEY].get(entity_id)
    return None


def get_player_for_device(
    hass: "HomeAssistant",
    device_id: str | None,
    room_player_mapping: dict[str, str] | None
) -> str | None:
    """Determine which media player to use for announcements.

    Logic:
    1. Try to find a media player linked to the device_id (voice satellite)
    2. Fall back to first player in room_player_mapping
    3. Return None if nothing configured
    """
    if not device_id and not room_player_mapping:
        return None

    # Try to find media player associated with this device
    if device_id:
        try:
            from homeassistant.helpers import device_registry as dr
            dev_reg = dr.async_get(hass)
            device = dev_reg.async_get(device_id)

            if device:
                # Check if this device has a linked media player
                # ESPHome satellites often have the same name as a media player
                device_name = (device.name or "").lower()

                # Search for matching media player in room mapping
                if room_player_mapping:
                    for room, player in room_player_mapping.items():
                        if device_name in room.lower() or room.lower() in device_name:
                            return player

                # Search all media players for a match
                all_states = hass.states.async_all()
                for state in all_states:
                    if state.entity_id.startswith("media_player."):
                        friendly = state.attributes.get("friendly_name", "").lower()
                        entity_name = state.entity_id.replace("media_player.", "").lower()
                        if device_name in friendly or device_name in entity_name:
                            return state.entity_id
        except Exception as err:
            _LOGGER.debug("Could not resolve device_id %s: %s", device_id, err)

    # Fall back to first configured player
    if room_player_mapping:
        return next(iter(room_player_mapping.values()), None)

    return None

# Word to number mapping for natural language
WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "a": 1, "an": 1,
}

# Common duration phrases
DURATION_PHRASES = {
    "half an hour": 30 * 60,
    "half hour": 30 * 60,
    "quarter hour": 15 * 60,
    "quarter of an hour": 15 * 60,
    "a minute": 60,
    "one minute": 60,
    "a second": 1,
    "a few minutes": 3 * 60,
    "a couple minutes": 2 * 60,
    "a couple of minutes": 2 * 60,
}


def _word_to_number(word: str) -> int | None:
    """Convert word to number, handling compound numbers like 'twenty five'."""
    word = word.lower().strip()

    # Direct lookup
    if word in WORD_TO_NUM:
        return WORD_TO_NUM[word]

    # Try compound numbers like "twenty five"
    parts = word.split()
    if len(parts) == 2:
        tens = WORD_TO_NUM.get(parts[0])
        ones = WORD_TO_NUM.get(parts[1])
        if tens and ones and tens >= 20:
            return tens + ones

    # Try numeric
    try:
        return int(word)
    except ValueError:
        pass

    # Try float and convert
    try:
        return int(float(word))
    except ValueError:
        pass

    return None


def _parse_duration(duration_str: str) -> timedelta | None:
    """Parse duration string with comprehensive natural language support.

    Handles:
    - "10 minutes", "1 hour 30 minutes", "90 seconds"
    - "half an hour", "quarter hour"
    - "an hour and a half", "2 and a half hours"
    - "one hour", "two minutes", "thirty seconds"
    - "1.5 hours", "2.5 minutes"
    - Just numbers like "10" (assumes minutes)
    - Military format "0:30" for 30 seconds, "1:30" for 1 min 30 sec
    """
    if not duration_str:
        return None

    original = duration_str
    duration_str = duration_str.lower().strip()

    # Remove common filler words
    duration_str = re.sub(r'\b(for|about|around|approximately|roughly|timer)\b', '', duration_str)
    duration_str = duration_str.strip()

    if not duration_str:
        return None

    total_seconds = 0

    # Check for known phrases first
    for phrase, seconds in DURATION_PHRASES.items():
        if phrase in duration_str:
            total_seconds += seconds
            duration_str = duration_str.replace(phrase, '')

    # Handle "X and a half hours/minutes"
    half_hour_match = re.search(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+and\s+a\s+half\s+hour', duration_str)
    if half_hour_match:
        num = _word_to_number(half_hour_match.group(1)) or 1
        total_seconds += num * 3600 + 1800  # hours + 30 min
        duration_str = duration_str[:half_hour_match.start()] + duration_str[half_hour_match.end():]

    half_min_match = re.search(r'(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+and\s+a\s+half\s+min', duration_str)
    if half_min_match:
        num = _word_to_number(half_min_match.group(1)) or 1
        total_seconds += num * 60 + 30  # minutes + 30 sec
        duration_str = duration_str[:half_min_match.start()] + duration_str[half_min_match.end():]

    # Handle decimal values like "1.5 hours"
    decimal_hour = re.search(r'(\d+\.?\d*)\s*(?:hour|hr|h)\b', duration_str)
    if decimal_hour:
        total_seconds += int(float(decimal_hour.group(1)) * 3600)
        duration_str = duration_str[:decimal_hour.start()] + duration_str[decimal_hour.end():]

    decimal_min = re.search(r'(\d+\.?\d*)\s*(?:minute|min|m)\b', duration_str)
    if decimal_min:
        total_seconds += int(float(decimal_min.group(1)) * 60)
        duration_str = duration_str[:decimal_min.start()] + duration_str[decimal_min.end():]

    decimal_sec = re.search(r'(\d+\.?\d*)\s*(?:second|sec|s)\b', duration_str)
    if decimal_sec:
        total_seconds += int(float(decimal_sec.group(1)))
        duration_str = duration_str[:decimal_sec.start()] + duration_str[decimal_sec.end():]

    # Handle word numbers like "one hour", "two minutes"
    word_hour = re.search(r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|a|an)\s*(?:hour|hr)', duration_str)
    if word_hour:
        num = _word_to_number(word_hour.group(1)) or 1
        total_seconds += num * 3600
        duration_str = duration_str[:word_hour.start()] + duration_str[word_hour.end():]

    word_min = re.search(r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|twenty[\s-]?five|thirty|forty|forty[\s-]?five|fifty|sixty|a|an)\s*(?:minute|min)', duration_str)
    if word_min:
        num = _word_to_number(word_min.group(1).replace('-', ' ')) or 1
        total_seconds += num * 60
        duration_str = duration_str[:word_min.start()] + duration_str[word_min.end():]

    word_sec = re.search(r'(one|two|three|four|five|six|seven|eight|nine|ten|fifteen|twenty|thirty|forty|fifty|sixty|a|an)\s*(?:second|sec)', duration_str)
    if word_sec:
        num = _word_to_number(word_sec.group(1)) or 1
        total_seconds += num

    # Handle MM:SS or HH:MM:SS format
    time_format = re.search(r'(\d+):(\d+)(?::(\d+))?', duration_str)
    if time_format:
        if time_format.group(3):  # HH:MM:SS
            total_seconds += int(time_format.group(1)) * 3600
            total_seconds += int(time_format.group(2)) * 60
            total_seconds += int(time_format.group(3))
        else:  # MM:SS
            total_seconds += int(time_format.group(1)) * 60
            total_seconds += int(time_format.group(2))

    # If just a number with no unit, assume minutes
    if total_seconds == 0:
        num_match = re.search(r'(\d+\.?\d*)', duration_str)
        if num_match:
            num = float(num_match.group(1))
            total_seconds = int(num * 60)  # Assume minutes

    if total_seconds <= 0:
        _LOGGER.warning("Could not parse duration from: %s", original)
        return None

    return timedelta(seconds=total_seconds)


def _format_duration(seconds: int) -> str:
    """Format seconds into human readable duration."""
    if seconds <= 0:
        return "0 seconds"

    parts = []

    hours = seconds // 3600
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        seconds %= 3600

    minutes = seconds // 60
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        seconds %= 60

    if seconds and (not parts or seconds > 0):
        # Only show seconds if it's the only unit or there are remaining seconds
        if not parts or (parts and seconds > 0):
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    if not parts:
        return "0 seconds"

    if len(parts) == 1:
        return parts[0]
    elif len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    else:
        return f"{parts[0]}, {parts[1]}, and {parts[2]}"


def _parse_remaining(remaining_str: str) -> int | None:
    """Parse HA remaining time format (HH:MM:SS or MM:SS) to seconds."""
    if not remaining_str:
        return None

    try:
        parts = remaining_str.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, AttributeError):
        pass
    return None


def _find_timer_by_name(timers: list, name: str) -> Any | None:
    """Find a timer by fuzzy name matching."""
    if not name:
        return None

    name_lower = name.lower().strip()

    # Remove common words
    name_clean = re.sub(r'\b(timer|the|my)\b', '', name_lower).strip()

    best_match = None
    best_score = 0

    for timer in timers:
        friendly = timer.attributes.get("friendly_name", "").lower()
        entity_name = timer.entity_id.replace("timer.", "").replace("_", " ").lower()

        # Exact match on friendly name
        if name_clean == friendly or name_clean == entity_name:
            return timer

        # Partial match
        if name_clean in friendly or name_clean in entity_name:
            score = len(name_clean) / max(len(friendly), len(entity_name))
            if score > best_score:
                best_score = score
                best_match = timer

        # Entity ID contains the name
        if name_clean in timer.entity_id.lower():
            score = 0.5
            if score > best_score:
                best_score = score
                best_match = timer

    return best_match if best_score > 0.3 else None


async def control_timer(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    device_id: str | None = None,
    room_player_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Control timers in Home Assistant.

    Args:
        arguments: Tool arguments (action, duration, name, add_time)
        hass: Home Assistant instance
        device_id: Device that initiated the command (for announcement targeting)
        room_player_mapping: Room to media player mapping

    Returns:
        Timer operation result
    """
    action = arguments.get("action", "").lower().strip()
    duration = arguments.get("duration", "")
    timer_name = arguments.get("name", "")
    add_time = arguments.get("add_time", "")  # For extending timers

    # Determine which player to announce on
    announce_player = get_player_for_device(hass, device_id, room_player_mapping)

    # Normalize action aliases
    action_aliases = {
        "create": "start", "set": "start", "begin": "start",
        "stop": "cancel", "delete": "cancel", "clear": "cancel", "end": "cancel", "kill": "cancel",
        "hold": "pause", "freeze": "pause",
        "continue": "resume", "unpause": "resume",
        "check": "status", "remaining": "status", "left": "status", "time": "status", "show": "status",
        "extend": "add", "increase": "add", "more": "add",
        "reset": "restart", "again": "restart", "repeat": "restart",
        "finish": "finish", "done": "finish", "complete": "finish",
    }
    action = action_aliases.get(action, action)

    try:
        # Get all timer entities
        all_states = hass.states.async_all()
        timers = [s for s in all_states if s.entity_id.startswith("timer.")]

        if not timers:
            return {
                "error": "No timer entities found in Home Assistant. Create a timer helper first (Settings → Devices & Services → Helpers → Add Helper → Timer).",
                "suggestion": "You can create a general-purpose timer called 'timer.kitchen' or 'timer.general'"
            }

        if action == "start":
            delta = _parse_duration(duration)
            if not delta:
                return {
                    "error": "Could not understand duration. Try something like '10 minutes', 'half an hour', '1 hour 30 minutes', or just '15' for 15 minutes.",
                    "examples": ["10 minutes", "half an hour", "1 hour 30 minutes", "90 seconds", "2 and a half hours"]
                }

            # Format duration for HA (HH:MM:SS)
            total_secs = int(delta.total_seconds())
            hours, remainder = divmod(total_secs, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Find the right timer to use
            timer_entity = None

            # If user specified a name, find matching timer
            if timer_name:
                matched_timer = _find_timer_by_name(timers, timer_name)
                if matched_timer:
                    timer_entity = matched_timer.entity_id

            # Otherwise, find an idle timer
            if not timer_entity:
                idle_timers = [t for t in timers if t.state == "idle"]
                if idle_timers:
                    # Prefer a general/kitchen timer
                    for t in idle_timers:
                        name = t.entity_id.lower()
                        if any(x in name for x in ["general", "kitchen", "main", "default"]):
                            timer_entity = t.entity_id
                            break
                    if not timer_entity:
                        timer_entity = idle_timers[0].entity_id

            # If all timers are busy, use the first one (will restart it)
            if not timer_entity:
                timer_entity = timers[0].entity_id

            await hass.services.async_call(
                "timer", "start",
                {"entity_id": timer_entity, "duration": duration_str},
                blocking=True
            )

            friendly = hass.states.get(timer_entity).attributes.get("friendly_name", "Timer")

            # Register for finish notification with target speaker
            register_timer(hass, timer_entity, friendly, announce_player)

            return {
                "success": True,
                "action": "started",
                "timer": timer_entity,
                "timer_name": friendly,
                "duration": _format_duration(total_secs),
                "duration_seconds": total_secs,
                "message": f"Started {friendly} for {_format_duration(total_secs)}"
            }

        elif action == "cancel":
            if timer_name:
                matched = _find_timer_by_name(timers, timer_name)
                if matched:
                    await hass.services.async_call(
                        "timer", "cancel",
                        {"entity_id": matched.entity_id},
                        blocking=True
                    )
                    unregister_timer(hass, matched.entity_id)
                    friendly = matched.attributes.get("friendly_name", "Timer")
                    return {
                        "success": True,
                        "action": "cancelled",
                        "timer": matched.entity_id,
                        "message": f"Cancelled {friendly}"
                    }
                return {"error": f"No timer found matching '{timer_name}'"}
            else:
                # Cancel all active timers
                active = [t for t in timers if t.state in ("active", "paused")]
                if not active:
                    return {"message": "No active timers to cancel"}

                for timer in active:
                    await hass.services.async_call(
                        "timer", "cancel",
                        {"entity_id": timer.entity_id},
                        blocking=True
                    )
                    unregister_timer(hass, timer.entity_id)

                return {
                    "success": True,
                    "action": "cancelled",
                    "count": len(active),
                    "message": f"Cancelled {len(active)} timer{'s' if len(active) != 1 else ''}"
                }

        elif action == "pause":
            target_timers = []
            if timer_name:
                matched = _find_timer_by_name(timers, timer_name)
                if matched and matched.state == "active":
                    target_timers = [matched]
                elif matched:
                    return {"message": f"Timer '{matched.attributes.get('friendly_name', timer_name)}' is not active (currently {matched.state})"}
            else:
                target_timers = [t for t in timers if t.state == "active"]

            if not target_timers:
                return {"message": "No active timers to pause"}

            for timer in target_timers:
                await hass.services.async_call(
                    "timer", "pause",
                    {"entity_id": timer.entity_id},
                    blocking=True
                )

            names = [t.attributes.get("friendly_name", "Timer") for t in target_timers]
            return {
                "success": True,
                "action": "paused",
                "count": len(target_timers),
                "timers": names,
                "message": f"Paused: {', '.join(names)}"
            }

        elif action == "resume":
            target_timers = []
            if timer_name:
                matched = _find_timer_by_name(timers, timer_name)
                if matched and matched.state == "paused":
                    target_timers = [matched]
                elif matched:
                    return {"message": f"Timer '{matched.attributes.get('friendly_name', timer_name)}' is not paused (currently {matched.state})"}
            else:
                target_timers = [t for t in timers if t.state == "paused"]

            if not target_timers:
                return {"message": "No paused timers to resume"}

            for timer in target_timers:
                await hass.services.async_call(
                    "timer", "start",
                    {"entity_id": timer.entity_id},
                    blocking=True
                )

            names = [t.attributes.get("friendly_name", "Timer") for t in target_timers]
            return {
                "success": True,
                "action": "resumed",
                "count": len(target_timers),
                "timers": names,
                "message": f"Resumed: {', '.join(names)}"
            }

        elif action == "status":
            active_info = []
            for timer in timers:
                if timer.state in ("active", "paused"):
                    remaining = timer.attributes.get("remaining", "")
                    remaining_secs = _parse_remaining(remaining)
                    friendly = timer.attributes.get("friendly_name", timer.entity_id)

                    active_info.append({
                        "name": friendly,
                        "entity_id": timer.entity_id,
                        "state": timer.state,
                        "remaining": remaining,
                        "remaining_friendly": _format_duration(remaining_secs) if remaining_secs else remaining,
                        "remaining_seconds": remaining_secs,
                    })

            if not active_info:
                return {
                    "message": "No active timers",
                    "timers": [],
                    "count": 0,
                    "total_timers": len(timers)
                }

            # Build message
            if len(active_info) == 1:
                t = active_info[0]
                state_word = "paused with" if t["state"] == "paused" else ""
                msg = f"{t['name']}: {state_word} {t['remaining_friendly']} remaining"
            else:
                parts = []
                for t in active_info:
                    state_prefix = "(paused) " if t["state"] == "paused" else ""
                    parts.append(f"{state_prefix}{t['name']}: {t['remaining_friendly']}")
                msg = "; ".join(parts)

            return {
                "count": len(active_info),
                "timers": active_info,
                "total_timers": len(timers),
                "message": msg
            }

        elif action == "add":
            # Add time to an existing timer
            add_duration = add_time or duration
            if not add_duration:
                return {"error": "Specify how much time to add (e.g., '5 minutes')"}

            delta = _parse_duration(add_duration)
            if not delta:
                return {"error": f"Could not parse duration: {add_duration}"}

            add_secs = int(delta.total_seconds())

            # Find target timer
            target = None
            if timer_name:
                target = _find_timer_by_name(timers, timer_name)
            else:
                # Find first active timer
                active = [t for t in timers if t.state in ("active", "paused")]
                if active:
                    target = active[0]

            if not target:
                return {"error": "No active timer to add time to"}

            # Get current remaining time
            current_remaining = _parse_remaining(target.attributes.get("remaining", "0:00")) or 0
            new_duration = current_remaining + add_secs

            # Format new duration
            hours, remainder = divmod(new_duration, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Restart timer with new duration
            await hass.services.async_call(
                "timer", "start",
                {"entity_id": target.entity_id, "duration": duration_str},
                blocking=True
            )

            friendly = target.attributes.get("friendly_name", "Timer")
            return {
                "success": True,
                "action": "extended",
                "timer": target.entity_id,
                "added": _format_duration(add_secs),
                "new_remaining": _format_duration(new_duration),
                "message": f"Added {_format_duration(add_secs)} to {friendly}. Now {_format_duration(new_duration)} remaining."
            }

        elif action == "restart":
            # Restart timer with same duration
            target = None
            if timer_name:
                target = _find_timer_by_name(timers, timer_name)
            else:
                # Find most recently used timer (check idle ones with duration attribute)
                for timer in timers:
                    if timer.attributes.get("duration"):
                        target = timer
                        break

            if not target:
                return {"error": "No timer found to restart"}

            original_duration = target.attributes.get("duration", "0:05:00")

            await hass.services.async_call(
                "timer", "start",
                {"entity_id": target.entity_id, "duration": original_duration},
                blocking=True
            )

            # Parse for friendly message
            duration_secs = _parse_remaining(original_duration) or 0
            friendly = target.attributes.get("friendly_name", "Timer")

            return {
                "success": True,
                "action": "restarted",
                "timer": target.entity_id,
                "duration": _format_duration(duration_secs),
                "message": f"Restarted {friendly} for {_format_duration(duration_secs)}"
            }

        elif action == "finish":
            # Manually finish/complete a timer
            target = None
            if timer_name:
                target = _find_timer_by_name(timers, timer_name)
            else:
                active = [t for t in timers if t.state in ("active", "paused")]
                if active:
                    target = active[0]

            if not target:
                return {"error": "No active timer to finish"}

            await hass.services.async_call(
                "timer", "finish",
                {"entity_id": target.entity_id},
                blocking=True
            )

            friendly = target.attributes.get("friendly_name", "Timer")
            return {
                "success": True,
                "action": "finished",
                "timer": target.entity_id,
                "message": f"Finished {friendly}"
            }

        else:
            return {
                "error": f"Unknown timer action: {action}",
                "valid_actions": ["start", "cancel", "pause", "resume", "status", "add", "restart", "finish"],
                "examples": [
                    "start a 10 minute timer",
                    "cancel the pizza timer",
                    "pause the timer",
                    "how much time is left",
                    "add 5 minutes to the timer",
                    "restart the timer"
                ]
            }

    except Exception as err:
        _LOGGER.error("Error controlling timer: %s", err, exc_info=True)
        return {"error": f"Timer error: {str(err)}"}
