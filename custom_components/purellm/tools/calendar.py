"""Calendar tool handler."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


async def get_calendar_events(
    arguments: dict[str, Any],
    hass: "HomeAssistant",
    calendar_entities: list[str],
    hass_timezone,
) -> dict[str, Any]:
    """Get calendar events from Home Assistant.

    Args:
        arguments: Tool arguments (query_type)
        hass: Home Assistant instance
        calendar_entities: List of calendar entity IDs
        hass_timezone: Home Assistant timezone

    Returns:
        Calendar events dict
    """
    from homeassistant.util import dt as dt_util

    query_type = arguments.get("query_type", "upcoming").lower()

    try:
        now = datetime.now(hass_timezone)

        # Determine time range based on query type
        if query_type == "today":
            start_time = now.replace(hour=0, minute=0, second=0)
            end_time = now.replace(hour=23, minute=59, second=59)
            max_results = 100
            period_desc = "today"
        elif query_type == "tomorrow":
            tomorrow = now + timedelta(days=1)
            start_time = tomorrow.replace(hour=0, minute=0, second=0)
            end_time = tomorrow.replace(hour=23, minute=59, second=59)
            max_results = 100
            period_desc = "tomorrow"
        elif query_type == "week":
            start_time = now
            end_time = now + timedelta(days=7)
            max_results = 100
            period_desc = "this week"
        elif query_type == "month":
            start_time = now
            end_time = now + timedelta(days=30)
            max_results = 100
            period_desc = "this month"
        elif query_type == "upcoming":
            start_time = now
            end_time = now + timedelta(days=365)
            max_results = 5
            period_desc = "upcoming"
        elif query_type == "birthday":
            start_time = now
            end_time = now + timedelta(days=365)
            max_results = 1
            period_desc = "next birthday"
        else:
            start_time = now
            end_time = now + timedelta(days=365)
            max_results = 5
            period_desc = "upcoming"

        _LOGGER.info("Calendar search (%s): %s to %s, max_results=%d",
                     query_type, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"), max_results)

        all_calendar_entities = calendar_entities if calendar_entities else []

        if not all_calendar_entities:
            all_states = hass.states.async_all()
            all_calendar_entities = [s.entity_id for s in all_states if s.entity_id.startswith("calendar.")]
            _LOGGER.info("No calendars configured, auto-discovered: %s", all_calendar_entities)

        # Filter to birthday calendar only if query_type is "birthday"
        if query_type == "birthday":
            calendar_list = [c for c in all_calendar_entities if "birthday" in c.lower()]
            if not calendar_list:
                calendar_list = all_calendar_entities
            _LOGGER.info("Birthday query - using calendars: %s", calendar_list)
        else:
            calendar_list = all_calendar_entities

        # Verify calendars exist
        existing_calendars = []
        for cal in calendar_list:
            cal_state = hass.states.get(cal)
            if cal_state:
                existing_calendars.append(cal)
                _LOGGER.info("Found calendar: %s (state: %s)", cal, cal_state.state)
            else:
                _LOGGER.warning("Calendar not found: %s", cal)

        if not existing_calendars:
            all_states = hass.states.async_all()
            found_calendars = [s.entity_id for s in all_states if s.entity_id.startswith("calendar.")]
            _LOGGER.info("Available calendars in HA: %s", found_calendars)
            return {"error": f"No calendars found. Available: {found_calendars}"}

        all_events = []

        for cal_entity in existing_calendars:
            try:
                _LOGGER.info("Querying calendar: %s", cal_entity)
                result = await hass.services.async_call(
                    "calendar",
                    "get_events",
                    {
                        "entity_id": cal_entity,
                        "start_date_time": start_time.isoformat(),
                        "end_date_time": end_time.isoformat(),
                    },
                    blocking=True,
                    return_response=True,
                )

                _LOGGER.info("Calendar %s raw result: %s", cal_entity, result)

                if result and cal_entity in result:
                    events = result[cal_entity].get("events", [])
                    _LOGGER.info("Calendar %s returned %d events", cal_entity, len(events))
                    for event in events:
                        event_start = event.get("start")
                        event_summary = event.get("summary", "Untitled")

                        try:
                            if "T" in str(event_start):
                                event_dt = datetime.fromisoformat(str(event_start).replace("Z", "+00:00"))
                                event_dt = event_dt.astimezone(hass_timezone)
                                time_str = event_dt.strftime("%B %d at %I:%M %p")
                                sort_key = event_dt
                            else:
                                event_dt = datetime.strptime(str(event_start), "%Y-%m-%d")
                                event_dt = event_dt.replace(tzinfo=hass_timezone)
                                time_str = event_dt.strftime("%B %d") + " (all day)"
                                sort_key = event_dt
                        except Exception as parse_err:
                            _LOGGER.warning("Date parse error for %s: %s", event_start, parse_err)
                            time_str = str(event_start)
                            sort_key = now + timedelta(days=9999)

                        is_birthday = "birthday" in cal_entity.lower()

                        all_events.append({
                            "title": event_summary,
                            "time": time_str,
                            "is_birthday": is_birthday,
                            "calendar": "birthdays" if is_birthday else "main",
                            "_sort_key": sort_key
                        })
                else:
                    _LOGGER.warning("No events key in result for %s", cal_entity)
            except Exception as cal_err:
                _LOGGER.error("Error getting events from %s: %s", cal_entity, cal_err, exc_info=True)

        # Sort ALL events by actual date
        all_events.sort(key=lambda x: x["_sort_key"])

        # Remove sort key from output
        for e in all_events:
            del e["_sort_key"]

        if not all_events:
            return {
                "query_type": query_type,
                "period": period_desc,
                "message": f"No events or birthdays found for {period_desc}",
                "events": []
            }

        result_events = all_events[:max_results]

        result = {
            "query_type": query_type,
            "period": period_desc,
            "event_count": len(result_events),
            "events": result_events
        }

        if query_type == "next":
            result["next_event"] = result_events[0] if result_events else None
        elif query_type in ("week", "month"):
            result["total_events"] = len(result_events)
            if len(all_events) > max_results:
                result["note"] = f"Showing all {len(result_events)} events"

        _LOGGER.info("Calendar events (combined): %s", result)
        return result

    except Exception as err:
        _LOGGER.error("Error getting calendar events: %s", err, exc_info=True)
        return {"error": f"Failed to get calendar events: {str(err)}"}
