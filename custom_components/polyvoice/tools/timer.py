"""Timer tool handler for PolyVoice."""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def _parse_duration(duration_str: str) -> timedelta | None:
    """Parse duration string like '10 minutes', '1 hour 30 minutes', '90 seconds'."""
    if not duration_str:
        return None

    duration_str = duration_str.lower().strip()
    total_seconds = 0

    # Match patterns like "1 hour", "30 minutes", "90 seconds"
    hour_match = re.search(r'(\d+)\s*(?:hour|hr|h)\b', duration_str)
    min_match = re.search(r'(\d+)\s*(?:minute|min|m)\b', duration_str)
    sec_match = re.search(r'(\d+)\s*(?:second|sec|s)\b', duration_str)

    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
    if sec_match:
        total_seconds += int(sec_match.group(1))

    # If just a number, assume minutes
    if total_seconds == 0:
        num_match = re.search(r'(\d+)', duration_str)
        if num_match:
            total_seconds = int(num_match.group(1)) * 60

    return timedelta(seconds=total_seconds) if total_seconds > 0 else None


def _format_duration(seconds: int) -> str:
    """Format seconds into human readable duration."""
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        if secs:
            return f"{minutes} minute{'s' if minutes != 1 else ''} {secs} second{'s' if secs != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        if minutes:
            return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
        return f"{hours} hour{'s' if hours != 1 else ''}"


async def control_timer(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
) -> dict[str, Any]:
    """Control timers in Home Assistant.

    Args:
        arguments: Tool arguments (action, duration, name)
        hass: Home Assistant instance

    Returns:
        Timer operation result
    """
    action = arguments.get("action", "").lower()
    duration = arguments.get("duration", "")
    timer_name = arguments.get("name", "").lower().strip()

    try:
        # Get all timer entities
        all_states = hass.states.async_all()
        timers = [s for s in all_states if s.entity_id.startswith("timer.")]

        if action == "start" or action == "create" or action == "set":
            # Parse duration
            delta = _parse_duration(duration)
            if not delta:
                return {"error": "Could not parse duration. Try '10 minutes', '1 hour', '90 seconds'"}

            # Format duration for HA (HH:MM:SS)
            total_secs = int(delta.total_seconds())
            hours, remainder = divmod(total_secs, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Find or use default timer
            timer_entity = None
            if timer_name:
                # Find timer by name
                for timer in timers:
                    friendly = timer.attributes.get("friendly_name", "").lower()
                    if timer_name in friendly or timer_name in timer.entity_id:
                        timer_entity = timer.entity_id
                        break

            # Use first available idle timer or create with timer.start
            if not timer_entity:
                for timer in timers:
                    if timer.state == "idle":
                        timer_entity = timer.entity_id
                        break

            if not timer_entity and timers:
                timer_entity = timers[0].entity_id

            if not timer_entity:
                return {"error": "No timer entities found. Create a timer helper in HA first."}

            await hass.services.async_call(
                "timer", "start",
                {"entity_id": timer_entity, "duration": duration_str},
                blocking=True
            )

            return {
                "success": True,
                "action": "started",
                "timer": timer_entity,
                "duration": _format_duration(total_secs),
                "message": f"Timer set for {_format_duration(total_secs)}"
            }

        elif action == "cancel" or action == "stop":
            # Find running timer(s)
            if timer_name:
                for timer in timers:
                    friendly = timer.attributes.get("friendly_name", "").lower()
                    if timer_name in friendly or timer_name in timer.entity_id:
                        await hass.services.async_call(
                            "timer", "cancel",
                            {"entity_id": timer.entity_id},
                            blocking=True
                        )
                        return {
                            "success": True,
                            "action": "cancelled",
                            "timer": timer.entity_id,
                            "message": f"Cancelled {timer.attributes.get('friendly_name', 'timer')}"
                        }
                return {"error": f"No timer found matching '{timer_name}'"}
            else:
                # Cancel all active timers
                active_timers = [t for t in timers if t.state in ("active", "paused")]
                if not active_timers:
                    return {"message": "No active timers to cancel"}

                for timer in active_timers:
                    await hass.services.async_call(
                        "timer", "cancel",
                        {"entity_id": timer.entity_id},
                        blocking=True
                    )

                return {
                    "success": True,
                    "action": "cancelled",
                    "count": len(active_timers),
                    "message": f"Cancelled {len(active_timers)} timer{'s' if len(active_timers) != 1 else ''}"
                }

        elif action == "pause":
            active_timers = [t for t in timers if t.state == "active"]
            if not active_timers:
                return {"message": "No active timers to pause"}

            for timer in active_timers:
                await hass.services.async_call(
                    "timer", "pause",
                    {"entity_id": timer.entity_id},
                    blocking=True
                )

            return {
                "success": True,
                "action": "paused",
                "count": len(active_timers),
                "message": f"Paused {len(active_timers)} timer{'s' if len(active_timers) != 1 else ''}"
            }

        elif action == "resume":
            paused_timers = [t for t in timers if t.state == "paused"]
            if not paused_timers:
                return {"message": "No paused timers to resume"}

            for timer in paused_timers:
                await hass.services.async_call(
                    "timer", "start",
                    {"entity_id": timer.entity_id},
                    blocking=True
                )

            return {
                "success": True,
                "action": "resumed",
                "count": len(paused_timers),
                "message": f"Resumed {len(paused_timers)} timer{'s' if len(paused_timers) != 1 else ''}"
            }

        elif action == "status" or action == "check":
            active_timers = []
            for timer in timers:
                if timer.state in ("active", "paused"):
                    remaining = timer.attributes.get("remaining", "")
                    friendly = timer.attributes.get("friendly_name", timer.entity_id)
                    active_timers.append({
                        "name": friendly,
                        "state": timer.state,
                        "remaining": remaining,
                    })

            if not active_timers:
                return {"message": "No active timers", "timers": []}

            return {
                "count": len(active_timers),
                "timers": active_timers,
                "message": f"{len(active_timers)} active timer{'s' if len(active_timers) != 1 else ''}"
            }

        else:
            return {"error": f"Unknown timer action: {action}"}

    except Exception as err:
        _LOGGER.error("Error controlling timer: %s", err, exc_info=True)
        return {"error": f"Timer error: {str(err)}"}
