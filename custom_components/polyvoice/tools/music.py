"""Music control tool handler."""
from __future__ import annotations

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

    Uses blueprint pattern for O(1) action dispatch and single-pass state caching.
    """

    # Valid media types for validation
    VALID_MEDIA_TYPES = frozenset({"track", "album", "artist", "playlist", "genre"})

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

        # Blueprint: O(1) action dispatch dictionary
        self._action_handlers: dict[str, callable] = {
            "play": self._handle_play,
            "pause": self._handle_pause,
            "resume": self._handle_resume,
            "stop": self._handle_stop,
            "skip_next": self._handle_skip_next,
            "skip_previous": self._handle_skip_previous,
            "restart_track": self._handle_restart_track,
            "what_playing": self._handle_what_playing,
            "transfer": self._handle_transfer,
            "shuffle": self._handle_shuffle,
        }

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

        # Validate media_type
        if media_type and media_type not in self.VALID_MEDIA_TYPES:
            _LOGGER.warning("Invalid media_type '%s', defaulting to 'artist'", media_type)
            media_type = "artist"

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

            # Blueprint: Single-pass state caching - gather all states once
            player_states = self._cache_player_states(all_players)

            # Blueprint: O(1) action dispatch
            handler = self._action_handlers.get(action)
            if handler is None:
                return {"error": f"Unknown action: {action}"}

            # Build context for handlers
            ctx = {
                "query": query,
                "media_type": media_type,
                "room": room,
                "shuffle": shuffle,
                "all_players": all_players,
                "target_players": target_players,
                "player_states": player_states,
            }

            return await handler(ctx)

        except Exception as err:
            _LOGGER.error("Music control error: %s", err, exc_info=True)
            return {"error": f"Music control failed: {str(err)}"}

    def _cache_player_states(self, all_players: list[str]) -> dict[str, dict]:
        """Cache all player states in a single pass.

        Returns dict: {entity_id: {"state": str, "attributes": dict}}
        """
        states = {}
        for pid in all_players:
            state_obj = self._hass.states.get(pid)
            if state_obj:
                states[pid] = {
                    "state": state_obj.state,
                    "attributes": dict(state_obj.attributes),
                }
                _LOGGER.debug("  %s → %s", pid, state_obj.state)
        return states

    def _find_player_by_state_cached(self, target_state: str, player_states: dict[str, dict]) -> str | None:
        """Find a player in a specific state using cached states."""
        for pid, data in player_states.items():
            if data["state"] == target_state:
                return pid
        return None

    def _find_target_players(self, room: str) -> list[str]:
        """Find target players for a room."""
        if room in self._players:
            return [self._players[room]]
        elif room:
            for rname, pid in self._players.items():
                if room in rname or rname in room:
                    return [pid]
        return []

    def _get_room_name(self, entity_id: str) -> str:
        """Get room name from entity_id."""
        for rname, pid in self._players.items():
            if pid == entity_id:
                return rname
        return "unknown"

    # ========== Blueprint Action Handlers ==========
    # All handlers receive ctx dict with: query, media_type, room, shuffle,
    # all_players, target_players, player_states

    async def _handle_play(self, ctx: dict) -> dict:
        """Play music."""
        query = ctx["query"]
        media_type = ctx["media_type"]
        room = ctx["room"]
        shuffle = ctx["shuffle"]
        target_players = ctx["target_players"]

        if not query:
            return {"error": "No music query specified"}
        if not target_players:
            return {"error": f"Unknown room: {room}. Available: {', '.join(self._players.keys())}"}

        for player in target_players:
            _LOGGER.info("Playing '%s' (%s) in %s - radio_mode=False", query, media_type, room)
            await self._hass.services.async_call(
                "music_assistant", "play_media",
                {"media_id": query, "media_type": media_type, "enqueue": "replace", "radio_mode": False},
                target={"entity_id": player},
                blocking=True
            )
            if shuffle or media_type == "genre":
                await self._hass.services.async_call(
                    "media_player", "shuffle_set",
                    {"entity_id": player, "shuffle": True},
                    blocking=True
                )

        # Natural response - include name and room
        shuffled = shuffle or media_type == "genre"
        if shuffled:
            speech = f"Now shuffling {query} in the {room}"
        else:
            speech = f"Now playing {query} in the {room}"

        return {
            "status": "ok",
            "name": query,
            "type": media_type,
            "room": room,
            "speech": speech
        }

    async def _handle_pause(self, ctx: dict) -> dict:
        """Pause music."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            await self._hass.services.async_call("media_player", "media_pause", {"entity_id": playing})
            self._last_paused_player = playing
            _LOGGER.info("Stored %s as last paused player", playing)
            return {"status": "paused", "message": f"Paused in {self._get_room_name(playing)}"}
        return {"error": "No music is currently playing"}

    async def _handle_resume(self, ctx: dict) -> dict:
        """Resume music."""
        all_players = ctx["all_players"]
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player to resume...")

        if self._last_paused_player and self._last_paused_player in all_players:
            _LOGGER.info("Resuming last paused player: %s", self._last_paused_player)
            await self._hass.services.async_call("media_player", "media_play", {"entity_id": self._last_paused_player})
            room_name = self._get_room_name(self._last_paused_player)
            self._last_paused_player = None
            return {"status": "resumed", "message": f"Resumed in {room_name}"}

        paused = self._find_player_by_state_cached("paused", player_states)
        if paused:
            await self._hass.services.async_call("media_player", "media_play", {"entity_id": paused})
            return {"status": "resumed", "message": f"Resumed in {self._get_room_name(paused)}"}

        return {"error": "No paused music to resume"}

    async def _handle_stop(self, ctx: dict) -> dict:
        """Stop music."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' or 'paused' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            await self._hass.services.async_call("media_player", "media_stop", {"entity_id": playing})
            return {"status": "stopped", "message": f"Stopped in {self._get_room_name(playing)}"}
        paused = self._find_player_by_state_cached("paused", player_states)
        if paused:
            await self._hass.services.async_call("media_player", "media_stop", {"entity_id": paused})
            return {"status": "stopped", "message": f"Stopped in {self._get_room_name(paused)}"}
        return {"message": "No music is playing"}

    async def _handle_skip_next(self, ctx: dict) -> dict:
        """Skip to next track."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            await self._hass.services.async_call("media_player", "media_next_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Skipped to next track"}
        return {"error": "No music is playing to skip"}

    async def _handle_skip_previous(self, ctx: dict) -> dict:
        """Skip to previous track."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            await self._hass.services.async_call("media_player", "media_previous_track", {"entity_id": playing})
            return {"status": "skipped", "message": "Previous track"}
        return {"error": "No music is playing"}

    async def _handle_restart_track(self, ctx: dict) -> dict:
        """Restart current track from beginning."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state to restart track...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            await self._hass.services.async_call("media_player", "media_seek", {"entity_id": playing, "seek_position": 0})
            return {"status": "restarted", "message": "Bringing it back from the top"}
        return {"error": "No music is playing"}

    async def _handle_what_playing(self, ctx: dict) -> dict:
        """Get currently playing track info using cached states."""
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
        if playing:
            # Use cached attributes instead of re-fetching
            attrs = player_states[playing]["attributes"]
            return {
                "title": attrs.get("media_title", "Unknown"),
                "artist": attrs.get("media_artist", "Unknown"),
                "album": attrs.get("media_album_name", ""),
                "room": self._get_room_name(playing)
            }
        return {"message": "No music currently playing"}

    async def _handle_transfer(self, ctx: dict) -> dict:
        """Transfer music to another room."""
        room = ctx["room"]
        target_players = ctx["target_players"]
        player_states = ctx["player_states"]

        _LOGGER.info("Looking for player in 'playing' state...")
        playing = self._find_player_by_state_cached("playing", player_states)
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

    async def _handle_shuffle(self, ctx: dict) -> dict:
        """Shuffle by genre or artist - deterministic matching.

        Search priority:
        1. Playlist with exact name match
        2. Playlist containing query in name
        3. Artist exact match (play artist discography shuffled)
        """
        query = ctx["query"]
        room = ctx["room"]
        target_players = ctx["target_players"]
        query_lower = query.lower().strip()

        if not query:
            return {"error": "No search query specified for shuffle"}
        if not target_players:
            return {"error": f"No room specified. Available: {', '.join(self._players.keys())}"}

        _LOGGER.info("Shuffle search for: '%s'", query)

        try:
            ma_entries = self._hass.config_entries.async_entries("music_assistant")
            if not ma_entries:
                return {"error": "Music Assistant integration not found"}
            ma_config_entry_id = ma_entries[0].entry_id

            # Step 1: Search playlists
            search_result = await self._hass.services.async_call(
                "music_assistant", "search",
                {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["playlist"], "limit": 10},
                blocking=True, return_response=True
            )

            matched_name = None
            matched_uri = None
            media_type_to_use = "playlist"

            # Parse playlist results
            playlists = []
            if search_result:
                if isinstance(search_result, dict):
                    playlists = search_result.get("playlists", []) or search_result.get("items", [])
                elif isinstance(search_result, list):
                    playlists = search_result

            # Deterministic matching: exact > contains
            if playlists:
                for pl in playlists:
                    pl_name = (pl.get("name") or pl.get("title", "")).lower()
                    # Exact match - use immediately
                    if pl_name == query_lower:
                        matched_name = pl.get("name") or pl.get("title")
                        matched_uri = pl.get("uri") or pl.get("media_id")
                        _LOGGER.info("Exact playlist match: %s", matched_name)
                        break
                    # Contains match - take first one found
                    if query_lower in pl_name and not matched_uri:
                        matched_name = pl.get("name") or pl.get("title")
                        matched_uri = pl.get("uri") or pl.get("media_id")
                        _LOGGER.info("Playlist contains match: %s", matched_name)

            # Step 2: Fall back to artist if no playlist
            if not matched_uri:
                _LOGGER.info("No playlist match, searching artist: %s", query)
                artist_result = await self._hass.services.async_call(
                    "music_assistant", "search",
                    {"config_entry_id": ma_config_entry_id, "name": query, "media_type": ["artist"], "limit": 5},
                    blocking=True, return_response=True
                )

                artists = []
                if artist_result:
                    if isinstance(artist_result, dict):
                        artists = artist_result.get("artists", [])
                    elif isinstance(artist_result, list):
                        artists = artist_result

                # Deterministic: exact > contains
                if artists:
                    for artist in artists:
                        artist_name = (artist.get("name", "")).lower()
                        if artist_name == query_lower:
                            matched_name = artist.get("name")
                            matched_uri = artist.get("uri") or artist.get("media_id")
                            media_type_to_use = "artist"
                            _LOGGER.info("Exact artist match: %s", matched_name)
                            break
                        if query_lower in artist_name and not matched_uri:
                            matched_name = artist.get("name")
                            matched_uri = artist.get("uri") or artist.get("media_id")
                            media_type_to_use = "artist"
                            _LOGGER.info("Artist contains match: %s", matched_name)

            if not matched_uri:
                return {"error": f"No playlist or artist found matching '{query}'"}

            # Play and shuffle
            player = target_players[0]
            _LOGGER.info("Playing %s (%s) shuffled in %s", matched_name, media_type_to_use, room)

            await self._hass.services.async_call(
                "music_assistant", "play_media",
                {"media_id": matched_uri, "media_type": media_type_to_use, "enqueue": "replace", "radio_mode": False},
                target={"entity_id": player},
                blocking=True
            )

            await self._hass.services.async_call(
                "media_player", "shuffle_set",
                {"entity_id": player, "shuffle": True},
                blocking=True
            )

            # Natural response - include playlist name and room
            return {
                "status": "ok",
                "name": matched_name,
                "type": media_type_to_use,
                "room": room,
                "speech": f"Now shuffling {matched_name} in the {room}"
            }

        except Exception as search_err:
            _LOGGER.error("Shuffle error: %s", search_err, exc_info=True)
            return {"error": f"Shuffle failed: {str(search_err)}"}
