"""Music control tool handler."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)


class MusicController:
    """Controller for music playback operations.

    This class manages music state (last paused player, debouncing)
    and handles all music control operations via Music Assistant.
    """

    def __init__(self, hass: "HomeAssistant", room_player_mapping: dict[str, str]):
        """Initialize the music controller.

        Args:
            hass: Home Assistant instance
            room_player_mapping: Dict of room name -> media_player entity_id
        """
        self._hass = hass
        self._players = room_player_mapping
        self._last_paused_player: str | None = None
        self._last_music_command: str | None = None
        self._last_music_command_time: datetime | None = None
        self._music_debounce_seconds = 3.0

    async def control_music(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Control music playback.

        Args:
            arguments: Tool arguments (action, query, room, media_type, shuffle)

        Returns:
            Result dict
        """
        action = arguments.get("action", "").lower()
        query = arguments.get("query", "")
        media_type = arguments.get("media_type", "artist")
        room = arguments.get("room", "").lower() if arguments.get("room") else ""
        shuffle = arguments.get("shuffle", False)

        _LOGGER.debug("Music control: action=%s, room=%s, query=%s", action, room, query)

        all_players = list(self._players.values())

        if not all_players:
            _LOGGER.error("No players configured! room_player_mapping is empty")
            return {"error": "No music players configured. Go to PolyVoice → Entity Configuration → Room to Player Mapping."}

        # Debounce check
        now = datetime.now()
        debounce_actions = {"skip_next", "skip_previous", "restart_track", "pause", "resume", "stop"}
        if action in debounce_actions:
            if (self._last_music_command == action and
                self._last_music_command_time and
                (now - self._last_music_command_time).total_seconds() < self._music_debounce_seconds):
                _LOGGER.info("DEBOUNCE: Ignoring duplicate '%s' command", action)
                return {"status": "debounced", "message": f"Command '{action}' ignored (duplicate)"}

        self._last_music_command = action
        self._last_music_command_time = now

        try:
            _LOGGER.info("=== MUSIC: %s ===", action.upper())

            # Determine target player(s)
            target_players = self._find_target_players(room)

            if action == "play":
                return await self._play(query, media_type, room, shuffle, target_players)
            elif action == "pause":
                return await self._pause(all_players)
            elif action == "resume":
                return await self._resume(all_players)
            elif action == "stop":
                return await self._stop(all_players)
            elif action == "skip_next":
                return await self._skip_next(all_players)
            elif action == "skip_previous":
                return await self._skip_previous(all_players)
            elif action == "restart_track":
                return await self._restart_track(all_players)
            elif action == "what_playing":
                return await self._what_playing(all_players)
            elif action == "transfer":
                return await self._transfer(all_players, target_players, room)
            elif action == "shuffle":
                return await self._shuffle(query, room, target_players)
            else:
                return {"error": f"Unknown action: {action}"}

        except Exception as err:
            _LOGGER.error("Music control error: %s", err, exc_info=True)
            return {"error": f"Music control failed: {str(err)}"}

    def _find_target_players(self, room: str) -> list[str]:
        """Find target players for a room."""
        if room in self._players:
            return [self._players[room]]
        elif room:
            for rname, pid in self._players.items():
                if room in rname or rname in room:
                    return [pid]
        return []

    def _find_player_by_state(self, target_state: str, all_players: list[str]) -> str | None:
        """Find a player in a specific state."""
        for pid in all_players:
            state = self._hass.states.get(pid)
            if state:
                _LOGGER.info("  %s → %s", pid, state.state)
                if state.state == target_state:
                    return pid
        return None

    def _get_room_name(self, entity_id: str) -> str:
        """Get room name from entity_id."""
        for rname, pid in self._players.items():
            if pid == entity_id:
                return rname
        return "unknown"

    async def _search_media(self, query: str, media_type: str) -> tuple[str, str, str] | None:
        """Search for media and return (media_id, name, resolved_type) or None."""
        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                _LOGGER.error("Music Assistant integration not found")
                return None
            ma_config_entry_id = ma_entries[0].entry_id

            # Search based on media type
            search_types = {
                "artist": ["artist"],
                "genre": ["playlist", "artist"],  # Search playlist first for genre
                "playlist": ["playlist"],
                "album": ["album"],
                "track": ["track"],
            }
            types_to_search = search_types.get(media_type, ["artist", "playlist"])

            for search_type in types_to_search:
                search_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": query, "media_type": [search_type], "limit": 1},
                    blocking=True, return_response=True
                )

                if search_result:
                    items = []
                    if isinstance(search_result, dict):
                        # Try various result keys
                        for key in [f"{search_type}s", "items", search_type]:
                            if key in search_result and search_result[key]:
                                items = search_result[key]
                                break
                    elif isinstance(search_result, list):
                        items = search_result

                    if items:
                        item = items[0]
                        name = item.get("name") or item.get("title", query)
                        media_id = item.get("uri") or item.get("media_id")
                        if media_id:
                            _LOGGER.info("Found %s: %s (%s)", search_type, name, media_id)
                            return (media_id, name, search_type)

            _LOGGER.warning("No results found for '%s' (type: %s)", query, media_type)
            return None

        except Exception as err:
            _LOGGER.error("Search error: %s", err, exc_info=True)
            return None

    async def _play(self, query: str, media_type: str, room: str, shuffle: bool, target_players: list[str]) -> dict:
        """Play music - searches and plays top result with shuffle."""
        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        # Search for the best match
        search_result = await self._search_media(query, media_type)
        if not search_result:
            return {"error": f"Could not find {media_type} matching '{query}'"}

        media_id, media_name, resolved_type = search_result

        # Play on all target players in parallel
        for player in target_players:
            # Fire play + shuffle in parallel (always shuffle for artist/genre)
            tasks = [
                self._hass.services.async_call(
                    "music_assistant", "play_media",
                    {"media_id": media_id, "media_type": resolved_type, "enqueue": "replace", "radio_mode": False},
                    target={"entity_id": player},
                    blocking=False
                )
            ]
            if shuffle or media_type in ("genre", "artist"):
                tasks.append(
                    self._hass.services.async_call(
                        "media_player", "shuffle_set",
                        {"entity_id": player, "shuffle": True},
                        blocking=False
                    )
                )
            await asyncio.gather(*tasks)

        return {"status": "playing", "message": f"Playing {media_name} in the {room}"}

    async def _pause(self, all_players: list[str]) -> dict:
        """Pause music."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_pause", {"entity_id": playing})
            self._last_paused_player = playing
            _LOGGER.info("Stored %s as last paused player", playing)
            return {"status": "paused", "message": f"Paused in {self._get_room_name(playing)}"}
        return {"error": "No music is currently playing"}

    async def _resume(self, all_players: list[str]) -> dict:
        """Resume music."""
        _LOGGER.info("Looking for player to resume...")

        if self._last_paused_player and self._last_paused_player in all_players:
            _LOGGER.info("Resuming last paused player: %s", self._last_paused_player)
            await self._hass.services.async_call("media_player", "media_play", {"entity_id": self._last_paused_player})
            room_name = self._get_room_name(self._last_paused_player)
            self._last_paused_player = None
            return {"status": "resumed", "message": f"Resumed in {room_name}"}

        paused = self._find_player_by_state("paused", all_players)
        if paused:
            await self._hass.services.async_call("media_player", "media_play", {"entity_id": paused})
            return {"status": "resumed", "message": f"Resumed in {self._get_room_name(paused)}"}

        return {"error": "No paused music to resume"}

    async def _stop(self, all_players: list[str]) -> dict:
        """Stop music."""
        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_stop", {"entity_id": playing})
            return {"status": "stopped", "message": f"Stopped in {self._get_room_name(playing)}"}
        paused = self._find_player_by_state("paused", all_players)
        if paused:
            await self._hass.services.async_call("media_player", "media_stop", {"entity_id": paused})
            return {"status": "stopped", "message": f"Stopped in {self._get_room_name(paused)}"}
        return {"message": "No music is playing"}

    async def _skip_next(self, all_players: list[str]) -> dict:
        """Skip to next track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_next_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Skipped to next track"}
        return {"error": "No music is playing to skip"}

    async def _skip_previous(self, all_players: list[str]) -> dict:
        """Skip to previous track."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Previous track"}
        return {"error": "No music is playing"}

    async def _restart_track(self, all_players: list[str]) -> dict:
        """Restart current track from beginning."""
        _LOGGER.info("Looking for player in 'playing' state to restart track...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            await self._hass.services.async_call("media_player", "media_seek", {"entity_id": playing, "seek_position": 0})
            return {"status": "restarted", "message": "Bringing it back from the top"}
        return {"error": "No music is playing"}

    async def _what_playing(self, all_players: list[str]) -> dict:
        """Get currently playing track info."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if playing:
            state = self._hass.states.get(playing)
            attrs = state.attributes
            return {
                "title": attrs.get("media_title", "Unknown"),
                "artist": attrs.get("media_artist", "Unknown"),
                "album": attrs.get("media_album_name", ""),
                "room": self._get_room_name(playing)
            }
        return {"message": "No music currently playing"}

    async def _transfer(self, all_players: list[str], target_players: list[str], room: str) -> dict:
        """Transfer music to another room."""
        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state("playing", all_players)
        if not playing:
            return {"error": "No music playing to transfer"}
        if not target_players:
            return {"error": f"No target room specified. Available: {', '.join(self._players.keys())}"}

        target = target_players[0]
        _LOGGER.info("Transferring from %s to %s", playing, target)

        await self._hass.services.async_call(
            "music_assistant", "transfer_queue",
            {"source_player": playing, "auto_play": True},
            target={"entity_id": target},
            blocking=True
        )
        return {"status": "transferred", "message": f"Music transferred to {self._get_room_name(target)}"}

    async def _shuffle(self, query: str, room: str, target_players: list[str]) -> dict:
        """Search and play shuffled - uses same logic as _play."""
        # Delegate to _play with shuffle=True (same unified logic)
        return await self._play(query, "playlist", room, shuffle=True, target_players=target_players)
