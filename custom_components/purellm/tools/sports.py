"""Sports tool handlers."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import aiohttp
    from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)

API_TIMEOUT = 15


async def get_sports_info(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get sports info from ESPN API with dynamic team search.

    Args:
        arguments: Tool arguments (team_name, query_type)
        session: aiohttp session
        hass_timezone: Home Assistant timezone
        track_api_call: Callback to track API usage

    Returns:
        Sports data dict
    """
    from homeassistant.util import dt as dt_util

    team_name = arguments.get("team_name", "")
    query_type = arguments.get("query_type", "both")

    if not team_name:
        return {"error": "No team name provided"}

    try:
        track_api_call("sports")
        headers = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}
        team_key = team_name.lower().strip()

        # Check for league-specific keywords to prioritize search
        champions_league_keywords = ["champions league", "ucl", "champions"]
        prioritize_ucl = any(kw in team_key for kw in champions_league_keywords)
        for kw in champions_league_keywords:
            team_key = team_key.replace(kw, "").strip()

        # Search for team in major leagues
        if prioritize_ucl:
            leagues_to_try = [
                ("soccer", "uefa.champions"),
                ("soccer", "eng.1"),
                ("basketball", "nba"),
                ("football", "nfl"),
                ("baseball", "mlb"),
                ("hockey", "nhl"),
                ("football", "college-football"),
                ("basketball", "mens-college-basketball"),
            ]
        else:
            leagues_to_try = [
                ("basketball", "nba"),
                ("football", "nfl"),
                ("baseball", "mlb"),
                ("hockey", "nhl"),
                ("soccer", "eng.1"),
                ("soccer", "uefa.champions"),
                ("football", "college-football"),
                ("basketball", "mens-college-basketball"),
            ]

        team_found = False
        url = None
        full_name = team_name
        team_leagues = []
        team_id = None

        search_words = team_key.split()

        for match_type in ["abbrev", "name"]:
            if team_found:
                break
            for sport, league in leagues_to_try:
                if team_found:
                    break
                teams_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams?limit=500"
                async with session.get(teams_url, headers=headers) as teams_resp:
                    if teams_resp.status == 200:
                        teams_data = await teams_resp.json()
                        for team in teams_data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
                            t = team.get("team", {})
                            match = False
                            if match_type == "abbrev":
                                match = team_key == t.get("abbreviation", "").lower()
                            else:
                                display_name = t.get("displayName", "").lower()
                                short_name = t.get("shortDisplayName", "").lower()
                                nickname = t.get("nickname", "").lower()
                                combined = f"{display_name} {short_name} {nickname}"
                                match = all(word in combined for word in search_words)

                            if match:
                                team_id = t.get("id", "")
                                full_name = t.get("displayName", team_name)
                                url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/teams/{team_id}/schedule"
                                team_leagues.append((sport, league))
                                if not team_found:
                                    team_found = True
                                break

        if not team_found:
            return {"error": f"Team '{team_name}' not found. Try the full team name (e.g., 'Miami Heat', 'New York Yankees')"}

        result = {"team": full_name}

        # Check scoreboard for live/upcoming games
        live_game_from_scoreboard = None
        next_game_from_scoreboard = None

        try:
            found_sport, found_league = team_leagues[0] if team_leagues else (None, None)
            scoreboards_to_check = [(found_sport, found_league)]
            if found_sport == "soccer":
                soccer_leagues = ["eng.1", "uefa.champions", "eng.fa", "eng.league_cup", "usa.1", "esp.1", "ger.1", "ita.1", "fra.1"]
                for sl in soccer_leagues:
                    if (found_sport, sl) not in scoreboards_to_check:
                        scoreboards_to_check.append((found_sport, sl))

            for sb_sport, sb_league in scoreboards_to_check:
                if live_game_from_scoreboard and next_game_from_scoreboard:
                    break

                scoreboard_url = f"https://site.api.espn.com/apis/site/v2/sports/{sb_sport}/{sb_league}/scoreboard"
                async with session.get(scoreboard_url, headers=headers) as sb_resp:
                    if sb_resp.status != 200:
                        continue
                    sb_data = await sb_resp.json()

                    for sb_event in sb_data.get("events", []):
                        sb_comp = sb_event.get("competitions", [{}])[0]
                        sb_status = sb_comp.get("status", {}).get("type", {})
                        sb_state = sb_status.get("state", "")

                        sb_competitors = sb_comp.get("competitors", [])
                        sb_team_ids = [c.get("team", {}).get("id", "") for c in sb_competitors]

                        if team_id not in sb_team_ids:
                            continue

                        home_team_sb = next((c for c in sb_competitors if c.get("homeAway") == "home"), {})
                        away_team_sb = next((c for c in sb_competitors if c.get("homeAway") == "away"), {})
                        home_name = home_team_sb.get("team", {}).get("displayName", "Home")
                        away_name = away_team_sb.get("team", {}).get("displayName", "Away")

                        if sb_state == "in":
                            home_score = home_team_sb.get("score", "0")
                            away_score = away_team_sb.get("score", "0")
                            if isinstance(home_score, dict):
                                home_score = home_score.get("displayValue", "0")
                            if isinstance(away_score, dict):
                                away_score = away_score.get("displayValue", "0")

                            status_detail = sb_status.get("detail", "In Progress")
                            result["live_game"] = {
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "status": status_detail,
                                "summary": f"LIVE: {away_name} {away_score} @ {home_name} {home_score} ({status_detail})"
                            }
                            live_game_from_scoreboard = True

                        elif sb_state == "pre" and not next_game_from_scoreboard:
                            game_date_str = sb_event.get("date", "")
                            if game_date_str:
                                try:
                                    game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                    game_dt_local = game_dt.astimezone(hass_timezone)
                                    now_local = datetime.now(hass_timezone)

                                    game_date_only = game_dt_local.date()
                                    today_date = now_local.date()
                                    tomorrow_date = today_date + timedelta(days=1)

                                    time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                                    if game_date_only == today_date:
                                        formatted_date = f"Today at {time_str}"
                                    elif game_date_only == tomorrow_date:
                                        formatted_date = f"Tomorrow at {time_str}"
                                    else:
                                        formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")
                                except (ValueError, KeyError, TypeError, AttributeError):
                                    formatted_date = sb_status.get("detail", "TBD")
                            else:
                                formatted_date = sb_status.get("detail", "TBD")

                            venue = sb_comp.get("venue", {}).get("fullName", "")
                            result["next_game"] = {
                                "date": formatted_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "venue": venue,
                                "summary": f"{away_name} @ {home_name} - {formatted_date}"
                            }
                            next_game_from_scoreboard = True

        except Exception as e:
            _LOGGER.warning("Failed to check scoreboard for live games: %s", e)

        # Get schedule for last game
        if url:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    events = data.get("events", [])

                    if query_type in ["last_game", "both"]:
                        # Find last completed game
                        last_game = None
                        for event in events:
                            status_info = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
                            is_completed = status_info.get("completed", False)
                            if is_completed:
                                game_date_str = event.get("date", "")
                                if game_date_str:
                                    try:
                                        game_date = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                        if last_game is None:
                                            last_game = event
                                        else:
                                            last_date = datetime.fromisoformat(last_game.get("date", "2000-01-01").replace("Z", "+00:00"))
                                            if game_date > last_date:
                                                last_game = event
                                    except (ValueError, KeyError, TypeError):
                                        pass

                        if last_game:
                            comp = last_game.get("competitions", [{}])[0]
                            competitors = comp.get("competitors", [])
                            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})

                            home_name = home_team.get("team", {}).get("displayName", "Home")
                            away_name = away_team.get("team", {}).get("displayName", "Away")
                            home_score_raw = home_team.get("score", "0")
                            away_score_raw = away_team.get("score", "0")
                            home_score = home_score_raw.get("displayValue", home_score_raw) if isinstance(home_score_raw, dict) else home_score_raw
                            away_score = away_score_raw.get("displayValue", away_score_raw) if isinstance(away_score_raw, dict) else away_score_raw

                            # Format last game date with relative dates
                            game_date_str = last_game.get("date", "")
                            formatted_last_date = game_date_str[:10]  # Default fallback
                            if game_date_str:
                                try:
                                    # Handle both full ISO timestamps and date-only strings
                                    if "T" in game_date_str:
                                        # Full timestamp: "2026-01-07T00:30:00Z"
                                        game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                    else:
                                        # Date only: "2026-01-07" - treat as UTC midnight
                                        game_dt = datetime.fromisoformat(game_date_str).replace(tzinfo=timezone.utc)

                                    game_dt_local = game_dt.astimezone(hass_timezone)
                                    now_local = datetime.now(hass_timezone)

                                    game_date_only = game_dt_local.date()
                                    today_date = now_local.date()
                                    yesterday_date = today_date - timedelta(days=1)

                                    if game_date_only == today_date:
                                        formatted_last_date = "today"
                                    elif game_date_only == yesterday_date:
                                        formatted_last_date = "yesterday"
                                    else:
                                        formatted_last_date = game_dt_local.strftime("%A, %B %d")
                                except (ValueError, KeyError, TypeError, AttributeError):
                                    pass

                            result["last_game"] = {
                                "date": formatted_last_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "home_score": home_score,
                                "away_score": away_score,
                                "summary": f"{away_name} {away_score} @ {home_name} {home_score}"
                            }

                    # Find next upcoming game from schedule if not found on scoreboard
                    if not next_game_from_scoreboard and query_type in ["next_game", "both"]:
                        now_utc = datetime.now(timezone.utc)
                        next_game = None
                        next_game_date = None

                        for event in events:
                            status_info = event.get("competitions", [{}])[0].get("status", {}).get("type", {})
                            is_completed = status_info.get("completed", False)
                            state = status_info.get("state", "")

                            if not is_completed and state == "pre":
                                game_date_str = event.get("date", "")
                                if game_date_str:
                                    try:
                                        game_dt = datetime.fromisoformat(game_date_str.replace("Z", "+00:00"))
                                        if game_dt > now_utc:
                                            if next_game is None or game_dt < next_game_date:
                                                next_game = event
                                                next_game_date = game_dt
                                    except (ValueError, KeyError, TypeError):
                                        pass

                        if next_game:
                            comp = next_game.get("competitions", [{}])[0]
                            competitors = comp.get("competitors", [])
                            home_team = next((c for c in competitors if c.get("homeAway") == "home"), {})
                            away_team = next((c for c in competitors if c.get("homeAway") == "away"), {})
                            home_name = home_team.get("team", {}).get("displayName", "Home")
                            away_name = away_team.get("team", {}).get("displayName", "Away")

                            # Format the date with relative dates
                            game_dt_local = next_game_date.astimezone(hass_timezone)
                            now_local = datetime.now(hass_timezone)
                            game_date_only = game_dt_local.date()
                            today_date = now_local.date()
                            tomorrow_date = today_date + timedelta(days=1)

                            time_str = game_dt_local.strftime("%I:%M %p").lstrip("0")
                            if game_date_only == today_date:
                                formatted_date = f"Today at {time_str}"
                            elif game_date_only == tomorrow_date:
                                formatted_date = f"Tomorrow at {time_str}"
                            else:
                                formatted_date = game_dt_local.strftime("%A, %B %d at %I:%M %p")

                            venue = comp.get("venue", {}).get("fullName", "")
                            result["next_game"] = {
                                "date": formatted_date,
                                "home_team": home_name,
                                "away_team": away_name,
                                "venue": venue,
                                "summary": f"{away_name} @ {home_name} - {formatted_date}"
                            }

        # Build response text
        response_parts = []
        if "live_game" in result:
            response_parts.append(result["live_game"]["summary"])
        if "last_game" in result:
            lg = result["last_game"]
            date_str = lg['date']
            # Build natural sentence for last game
            home = lg['home_team']
            away = lg['away_team']
            h_score = lg['home_score']
            a_score = lg['away_score']
            # Determine winner and build natural response
            try:
                if int(h_score) > int(a_score):
                    winner, loser = home, away
                    w_score, l_score = h_score, a_score
                else:
                    winner, loser = away, home
                    w_score, l_score = a_score, h_score
                # Natural phrasing the LLM won't want to change
                if date_str in ["today", "yesterday"]:
                    response_parts.append(f"{winner} beat {loser} {w_score}-{l_score} {date_str}")
                else:
                    response_parts.append(f"{winner} beat {loser} {w_score}-{l_score} on {date_str}")
            except ValueError:
                if date_str in ["today", "yesterday"]:
                    response_parts.append(f"{lg['summary']} {date_str}")
                else:
                    response_parts.append(f"{lg['summary']} on {date_str}")
        if "next_game" in result:
            ng = result["next_game"]
            response_parts.append(f"Next game: {ng['summary']}")

        result["response_text"] = ". ".join(response_parts) if response_parts else f"No game info found for {full_name}"

        _LOGGER.info("Sports info for %s: %s", full_name, result.get("response_text", ""))
        # Return only response_text to prevent LLM from reformatting dates
        return {"response_text": result["response_text"], "team": result.get("team", "")}

    except Exception as err:
        _LOGGER.error("Sports API error: %s", err, exc_info=True)
        return {"error": f"Failed to get sports info: {str(err)}"}


async def get_ufc_info(
    arguments: dict[str, Any],
    session: "aiohttp.ClientSession",
    hass_timezone,
    track_api_call: callable,
) -> dict[str, Any]:
    """Get UFC/MMA event information from ESPN API.

    Args:
        arguments: Tool arguments (query_type)
        session: aiohttp session
        hass_timezone: Home Assistant timezone
        track_api_call: Callback to track API usage

    Returns:
        UFC event data dict
    """
    query_type = arguments.get("query_type", "next_event")

    try:
        track_api_call("sports")
        headers = {"User-Agent": "HomeAssistant-PolyVoice/1.0"}

        async with asyncio.timeout(API_TIMEOUT):
            events_url = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
            async with session.get(events_url, headers=headers) as resp:
                if resp.status != 200:
                    return {"error": f"ESPN UFC API error: {resp.status}"}
                data = await resp.json()

        leagues = data.get("leagues", [{}])
        calendar = leagues[0].get("calendar", []) if leagues else []

        if not calendar:
            return {"error": "No upcoming UFC events found"}

        result = {"events": []}
        now_local = datetime.now(hass_timezone)
        today_date = now_local.date()
        tomorrow_date = today_date + timedelta(days=1)
        yesterday_date = today_date - timedelta(days=1)

        for event in calendar[:5]:
            event_info = {
                "name": event.get("label", "Unknown Event"),
                "date": event.get("startDate", "")[:10] if event.get("startDate") else "TBD"
            }
            if event_info["date"] and event_info["date"] != "TBD":
                try:
                    event_dt = datetime.fromisoformat(event.get("startDate", "").replace("Z", "+00:00"))
                    event_dt_local = event_dt.astimezone(hass_timezone)
                    event_date_only = event_dt_local.date()

                    if event_date_only == today_date:
                        event_info["formatted_date"] = "Today"
                    elif event_date_only == tomorrow_date:
                        event_info["formatted_date"] = "Tomorrow"
                    elif event_date_only == yesterday_date:
                        event_info["formatted_date"] = "Yesterday"
                    else:
                        event_info["formatted_date"] = event_dt_local.strftime("%A, %B %d")
                except:
                    event_info["formatted_date"] = event_info["date"]
            result["events"].append(event_info)

        if query_type == "next_event" and result["events"]:
            next_evt = result["events"][0]
            result["response_text"] = f"The next UFC event is {next_evt['name']} on {next_evt.get('formatted_date', next_evt['date'])}."
        elif query_type == "upcoming" and result["events"]:
            event_list = [f"{e['name']} ({e.get('formatted_date', e['date'])})" for e in result["events"]]
            result["response_text"] = "Upcoming UFC events: " + ", ".join(event_list)
        else:
            result["response_text"] = "No upcoming UFC events found."

        _LOGGER.info("UFC info: %s", result.get("response_text", ""))
        return result

    except Exception as err:
        _LOGGER.error("UFC API error: %s", err, exc_info=True)
        return {"error": f"Failed to get UFC info: {str(err)}"}
