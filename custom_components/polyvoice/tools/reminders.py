"""Reminders tool handler for PolyVoice."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


def _parse_reminder_time(time_str: str, now: datetime) -> datetime | None:
    """Parse reminder time from natural language.

    Examples:
    - "in 30 minutes" -> now + 30 minutes
    - "at 5pm" -> today at 5pm (or tomorrow if past)
    - "at 5:30pm" -> specific time
    - "tomorrow at noon" -> tomorrow at 12pm
    - "in 2 hours" -> now + 2 hours
    """
    if not time_str:
        return None

    time_str = time_str.lower().strip()

    # "in X minutes/hours" pattern
    in_match = re.search(r'in\s+(\d+)\s*(minute|min|hour|hr|h|m)\s*s?', time_str)
    if in_match:
        amount = int(in_match.group(1))
        unit = in_match.group(2)
        if unit in ('hour', 'hr', 'h'):
            return now + timedelta(hours=amount)
        else:
            return now + timedelta(minutes=amount)

    # "at X:XX pm/am" pattern
    at_match = re.search(r'(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a|p)?', time_str)
    if at_match:
        hour = int(at_match.group(1))
        minute = int(at_match.group(2) or 0)
        ampm = at_match.group(3) or ""

        if ampm.startswith('p') and hour != 12:
            hour += 12
        elif ampm.startswith('a') and hour == 12:
            hour = 0

        # Check for "tomorrow"
        is_tomorrow = "tomorrow" in time_str

        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

        if is_tomorrow:
            target += timedelta(days=1)
        elif target <= now:
            # If time has passed today, assume tomorrow
            target += timedelta(days=1)

        return target

    # "tomorrow" only
    if "tomorrow" in time_str:
        return (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)

    # "noon" pattern
    if "noon" in time_str:
        target = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if "tomorrow" in time_str or target <= now:
            target += timedelta(days=1)
        return target

    return None


async def create_reminder(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    hass_timezone,
) -> dict[str, Any]:
    """Create a reminder using HA todo list with due date.

    Args:
        arguments: Tool arguments (reminder, time)
        hass: Home Assistant instance
        hass_timezone: Home Assistant timezone

    Returns:
        Reminder creation result
    """
    reminder_text = arguments.get("reminder", "").strip()
    time_str = arguments.get("time", "").strip()

    if not reminder_text:
        return {"error": "Please specify what to remind you about"}

    try:
        now = datetime.now(hass_timezone)

        # Parse the reminder time
        reminder_time = _parse_reminder_time(time_str, now)
        if not reminder_time:
            # Default to 1 hour from now
            reminder_time = now + timedelta(hours=1)

        # Find a todo list for reminders (prefer one with "reminder" in name)
        all_states = hass.states.async_all()
        todo_lists = [s for s in all_states if s.entity_id.startswith("todo.")]

        target_list = None
        for todo in todo_lists:
            if "reminder" in todo.entity_id.lower():
                target_list = todo.entity_id
                break

        # Fall back to any todo list
        if not target_list and todo_lists:
            target_list = todo_lists[0].entity_id

        if not target_list:
            # Fall back to calendar event if no todo list
            calendar_entities = [s for s in all_states if s.entity_id.startswith("calendar.")]
            if calendar_entities:
                # Create a calendar event instead
                calendar_entity = calendar_entities[0].entity_id
                end_time = reminder_time + timedelta(minutes=15)

                await hass.services.async_call(
                    "calendar", "create_event",
                    {
                        "entity_id": calendar_entity,
                        "summary": f"Reminder: {reminder_text}",
                        "start_date_time": reminder_time.isoformat(),
                        "end_date_time": end_time.isoformat(),
                    },
                    blocking=True
                )

                return {
                    "success": True,
                    "type": "calendar_event",
                    "reminder": reminder_text,
                    "time": reminder_time.strftime("%I:%M %p on %A, %B %d"),
                    "message": f"Reminder set for {reminder_time.strftime('%I:%M %p')}: {reminder_text}"
                }

            return {"error": "No todo list or calendar found for reminders"}

        # Create todo item with due date
        await hass.services.async_call(
            "todo", "add_item",
            {
                "entity_id": target_list,
                "item": f"Reminder: {reminder_text}",
                "due_datetime": reminder_time.isoformat(),
            },
            blocking=True
        )

        return {
            "success": True,
            "type": "todo_item",
            "reminder": reminder_text,
            "time": reminder_time.strftime("%I:%M %p on %A, %B %d"),
            "list": hass.states.get(target_list).attributes.get("friendly_name", "reminders"),
            "message": f"Reminder set for {reminder_time.strftime('%I:%M %p')}: {reminder_text}"
        }

    except Exception as err:
        _LOGGER.error("Error creating reminder: %s", err, exc_info=True)
        return {"error": f"Reminder error: {str(err)}"}


async def get_reminders(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    hass_timezone,
) -> dict[str, Any]:
    """Get upcoming reminders.

    Args:
        arguments: Tool arguments
        hass: Home Assistant instance
        hass_timezone: Home Assistant timezone

    Returns:
        List of upcoming reminders
    """
    try:
        all_states = hass.states.async_all()
        todo_lists = [s for s in all_states if s.entity_id.startswith("todo.")]

        reminders = []

        for todo in todo_lists:
            result = await hass.services.async_call(
                "todo", "get_items",
                {"entity_id": todo.entity_id, "status": "needs_action"},
                blocking=True,
                return_response=True
            )

            if result and todo.entity_id in result:
                items = result[todo.entity_id].get("items", [])
                for item in items:
                    summary = item.get("summary", "")
                    due = item.get("due")
                    if summary.lower().startswith("reminder:") or due:
                        reminders.append({
                            "text": summary.replace("Reminder: ", "").replace("reminder: ", ""),
                            "due": due,
                            "list": todo.attributes.get("friendly_name", todo.entity_id)
                        })

        if not reminders:
            return {"message": "No upcoming reminders", "reminders": []}

        return {
            "count": len(reminders),
            "reminders": reminders,
            "message": f"You have {len(reminders)} reminder{'s' if len(reminders) != 1 else ''}"
        }

    except Exception as err:
        _LOGGER.error("Error getting reminders: %s", err, exc_info=True)
        return {"error": f"Error getting reminders: {str(err)}"}
