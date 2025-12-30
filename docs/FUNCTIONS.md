# 🛠️ Built-in Functions Reference

PolyVoice includes 15+ built-in functions that can be toggled on/off individually.

## Overview

| Function | Description | Requires |
|----------|-------------|----------|
| `get_weather_forecast` | Weather conditions + forecast | OpenWeatherMap API |
| `get_calendar_events` | Upcoming calendar events | Calendar entities |
| `control_music` | Play, pause, skip, transfer | Music players |
| `check_*_camera` | AI camera analysis | ha_video_vision |
| `get_sports_info` | Live scores & schedules | None |
| `get_news` | Headlines by category | NewsAPI key |
| `find_nearby_places` | Location search | Google Places API |
| `get_restaurant_recommendations` | Food recommendations | Yelp API |
| `control_thermostat` | Temperature control | Climate entity |
| `check_device_status` | Doors, locks, sensors | Device aliases |
| `get_wikipedia_summary` | Knowledge lookup | None |
| `calculate_age` | Celebrity ages | None |
| `get_current_datetime` | Current date/time | None |

---

## Weather

### `get_weather_forecast`

**Requires:** OpenWeatherMap API key

**Example prompts:**
- "What's the weather?"
- "Will it rain tomorrow?"
- "What's the forecast for this week?"

**Returns:**
- Current temperature, feels like, humidity
- Current conditions (sunny, cloudy, etc.)
- 5-day forecast with highs/lows

**Configuration:**
1. Get free API key at [openweathermap.org/api](https://openweathermap.org/api)
2. Add key in **Settings → PolyVoice → API Keys**

---

## Calendar

### `get_calendar_events`

**Requires:** Calendar entity IDs configured

**Example prompts:**
- "What's on my calendar today?"
- "What do I have tomorrow?"
- "Any meetings this week?"

**Configuration:**
In **Entity Configuration**, add calendar entities (one per line):
```
calendar.personal
calendar.work
calendar.family_birthdays
```

---

## Music Control

### `control_music`

**Requires:** Music Assistant or media_player entities

**Actions:**
| Action | Description |
|--------|-------------|
| `play` | Start playback (with query) |
| `pause` | Pause playback |
| `stop` | Stop playback |
| `resume` | Resume paused playback |
| `skip_next` | Next track |
| `skip_previous` | Previous track |
| `what_playing` | Current track info |
| `transfer` | Move to different room |

**Example prompts:**
- "Play jazz in the living room"
- "Shuffle Taylor Swift"
- "What's playing?"
- "Pause the music"
- "Move the music to the kitchen"
- "Play everywhere"

**Configuration:**
In **Entity Configuration**:

**Default Music Player:**
```
media_player.living_room_speaker
```

**Music Players by Room:**
```
living room:media_player.living_room_speaker
kitchen:media_player.kitchen_speaker
bedroom:media_player.bedroom_speaker
everywhere:media_player.whole_home_group
```

---

## Camera Vision

### `check_porch_camera`, `check_driveway_camera`, `check_backyard_camera`

**Requires:** [ha_video_vision](https://github.com/LosCV29/ha-video-vision) integration

**Example prompts:**
- "Check the front door"
- "Who's at the driveway?"
- "What's happening in the backyard?"

**Returns:**
- AI description of the scene
- Identified people (with facial recognition)
- Person detected flag

---

## Sports

### `get_sports_info`

**Requires:** Nothing (built-in)

**Supported leagues:**
- NFL, NBA, MLB, NHL, MLS
- Premier League, La Liga, Champions League
- College Football, College Basketball

**Example prompts:**
- "Did the Panthers win?"
- "What's the Lakers score?"
- "When do the Chiefs play next?"
- "NFL scores today"

---

## News

### `get_news`

**Requires:** TheNewsAPI key

**Categories:**
- general, business, tech, sports, entertainment, science, health

**Example prompts:**
- "What's in the news?"
- "Tech news"
- "Any sports headlines?"

**Configuration:**
1. Get API key at [thenewsapi.com](https://www.thenewsapi.com)
2. Add key in **Settings → PolyVoice → API Keys**

---

## Places

### `find_nearby_places`

**Requires:** Google Places API key

**Example prompts:**
- "Find the nearest gas station"
- "Coffee shops near me"
- "Where's the closest pharmacy?"

**Returns:**
- Place name and address
- Distance from you
- Rating
- Open/closed status

**Configuration:**
1. Enable Places API in [Google Cloud Console](https://console.cloud.google.com)
2. Add key in **Settings → PolyVoice → API Keys**

---

## Restaurants

### `get_restaurant_recommendations`

**Requires:** Yelp API key

**Example prompts:**
- "Best tacos near me"
- "Italian restaurants"
- "Where should we get sushi?"

**Returns:**
- Restaurant name
- Rating and review count
- Price range
- Cuisine type
- Distance
- Open status

**Configuration:**
1. Get API key at [yelp.com/developers](https://www.yelp.com/developers)
2. Add key in **Settings → PolyVoice → API Keys**

---

## Thermostat

### `control_thermostat`

**Requires:** Climate entity configured

**Actions:**
| Action | Description |
|--------|-------------|
| `set` | Set specific temperature |
| `raise` | Increase temperature |
| `lower` | Decrease temperature |

**Example prompts:**
- "Set the AC to 72"
- "Make it warmer"
- "Lower the temperature"
- "Turn down the AC"

**Configuration:**
In **Entity Configuration**:
```
climate.thermostat
```

---

## Device Status

### `check_device_status`

**Requires:** Device aliases configured

**Example prompts:**
- "Is the front door locked?"
- "Is the garage open?"
- "Check the back door"
- "Did the mail come?"

**Configuration:**
In **Entity Configuration**, add device aliases:
```
front door:lock.front_door_lock
back door:binary_sensor.back_door_contact
garage:cover.garage_door
mailbox:binary_sensor.mailbox_contact
living room light:light.living_room
```

---

## Wikipedia

### `get_wikipedia_summary`

**Requires:** Nothing (built-in)

**Example prompts:**
- "Tell me about the Eiffel Tower"
- "What is quantum computing?"
- "Who was Napoleon?"

---

## Age Calculator

### `calculate_age`

**Requires:** Nothing (built-in)

Uses Wikipedia + current date to calculate ages.

**Example prompts:**
- "How old is Tom Hanks?"
- "What's Taylor Swift's age?"
- "When was Elon Musk born?"

---

## Date/Time

### `get_current_datetime`

**Requires:** Nothing (built-in)

**Example prompts:**
- "What time is it?"
- "What's today's date?"
- "What day is it?"

---

## Enabling/Disabling Functions

All functions can be toggled in:
**Settings → PolyVoice → Configure → Enable/Disable Features**

Only enabled functions are sent to the LLM, reducing token usage and preventing hallucinated tool calls.
