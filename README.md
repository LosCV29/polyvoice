# 🎙️ PolyVoice

[![HACS Custom](https://img.shields.io/badge/HACS-Custom-41BDF5.svg)](https://github.com/hacs/integration)
[![Home Assistant](https://img.shields.io/badge/Home%20Assistant-2024.1+-blue.svg)](https://www.home-assistant.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**The multi-provider voice assistant for Home Assistant** — 15+ built-in functions, 6 LLM providers, local-first, and completely free.

> 🎯 **Like Alexa, but you choose the brain. Local or cloud. Your call.**

---

## ✨ Why PolyVoice?

*"Poly"* = Many. Many voices. Many providers. Many functions. One seamless assistant.

| Feature | Alexa/Google | PolyVoice |
|---------|--------------|-----------|
| Voice Control | ✅ | ✅ |
| Smart Home | ✅ | ✅ + Native HA |
| Weather | ✅ | ✅ OpenWeatherMap |
| Music Control | ✅ | ✅ Music Assistant |
| Calendar | ✅ | ✅ Any HA Calendar |
| Sports Scores | ✅ | ✅ Live Scores |
| **Choose Your AI** | ❌ Locked in | ✅ 6 Providers |
| **AI Camera Vision** | ❌ | ✅ "Who's at the door?" |
| **100% Local Option** | ❌ | ✅ Your Hardware |
| **Privacy** | ❌ Cloud | ✅ Local First |
| **Monthly Cost** | $0-10+ | **$0** |

---

## 🔌 Choose Your Brain

| Provider | Type | Cost | Best For |
|----------|------|------|----------|
| **LM Studio** | Local | FREE | Privacy, Offline |
| **OpenRouter** | Cloud | FREE tier | Best Models |
| **Groq** | Cloud | FREE | ⚡ Fastest |
| **OpenAI** | Cloud | Paid | GPT-4 Quality |
| **Anthropic** | Cloud | Paid | Claude Quality |
| **Google** | Cloud | FREE tier | Gemini |

**Switch providers anytime.** Your config stays the same.

---

## 🚀 Quick Start

### Installation (HACS)

1. Open HACS → Integrations → ⋮ → Custom Repositories
2. Add: `https://github.com/YOUR_USERNAME/polyvoice`
3. Install "PolyVoice"
4. Restart Home Assistant
5. Settings → Devices & Services → Add Integration → "PolyVoice"

### Installation (Manual)

```bash
cp -r polyvoice /config/custom_components/
```

---

## 🛠️ Built-in Functions (15+)

Toggle each on/off in the UI. Only enable what you need!

| Function | Description | Requires |
|----------|-------------|----------|
| 🌤️ **Weather** | Current + 5-day forecast | OpenWeatherMap API |
| 📅 **Calendar** | View upcoming events | HA Calendar entities |
| 🎵 **Music** | Play, pause, skip, transfer | Music Assistant |
| 📹 **Cameras** | AI video analysis | ha_video_vision |
| 🏈 **Sports** | Live scores & schedules | — |
| 📰 **News** | Headlines by category | NewsAPI |
| 📍 **Places** | Find nearby locations | Google Places API |
| 🍕 **Restaurants** | Ratings & recommendations | Yelp API |
| 🌡️ **Thermostat** | Temperature control | Climate entity |
| 🚪 **Devices** | Doors, locks, sensors | Device aliases |
| 📚 **Wikipedia** | Knowledge lookup | — |
| 🎂 **Age** | Celebrity ages | — |
| ⏰ **Time** | Current date/time | — |

---

## 💬 Example Commands

```
"What's the weather?"
"Play jazz in the living room"
"Is the front door locked?"
"Set the AC to 72"
"Did the Lakers win?"
"Who's at the driveway?"
"Find the nearest gas station"
"What's on my calendar tomorrow?"
"How old is Tom Hanks?"
```

---

## ⚙️ Configuration

After setup, configure via:
**Settings → Devices & Services → PolyVoice → Configure**

| Section | Configure |
|---------|-----------|
| **Connection** | Provider, API key, URL |
| **Model** | Temperature, tokens, model |
| **Features** | Toggle functions on/off |
| **Entities** | Thermostat, calendars, players |
| **API Keys** | Weather, Places, Yelp, News |
| **Location** | Override HA location |
| **Intents** | Native HA handling |
| **Advanced** | System prompt |

---

## 📋 Entity Configuration

**Calendars** (one per line):
```
calendar.personal
calendar.work
```

**Music Players** (room:entity_id):
```
living room:media_player.living_room
kitchen:media_player.kitchen
everywhere:media_player.whole_home
```

**Device Aliases** (alias:entity_id):
```
front door:lock.front_door
garage:cover.garage_door
```

---

## 🔑 API Keys

| Feature | Provider | Free Key |
|---------|----------|----------|
| Weather | OpenWeatherMap | [openweathermap.org](https://openweathermap.org/api) |
| Places | Google | [console.cloud.google.com](https://console.cloud.google.com) |
| Restaurants | Yelp | [yelp.com/developers](https://www.yelp.com/developers) |
| News | TheNewsAPI | [thenewsapi.com](https://www.thenewsapi.com) |

---

## 📹 Camera Integration

For AI camera vision, install the companion:

### [HA Video Vision](https://github.com/YOUR_USERNAME/ha-video-vision)

- Real **video analysis** (not snapshots!)
- **Facial recognition** — "Who's at the door?"
- Works with any RTSP camera

---

## 💡 Recommended Setup

```
┌─────────────────────────────────────────┐
│            YOUR SETUP                   │
├─────────────────────────────────────────┤
│  Primary:   LM Studio (local)           │
│             └── Qwen 7B or Llama 3.2    │
│                                         │
│  Backup:    OpenRouter (free)           │
│             └── Llama 3.3 70B           │
│                                         │
│  Cameras:   HA Video Vision             │
│             └── Nemotron (free)         │
└─────────────────────────────────────────┘
         Total Monthly Cost: $0
```

---

## 🔧 Troubleshooting

**No tools available?**
- Enable features in options
- Add required API keys
- Configure entities

**Can't connect?**
- LM Studio: Check server URL
- Cloud: Verify API key

**Slow responses?**
- Try smaller model (7B)
- Use Groq (fastest cloud)

---

## 🤝 Works Great With

- **[HA Video Vision](https://github.com/YOUR_USERNAME/ha-video-vision)** — AI cameras
- **[Music Assistant](https://music-assistant.io/)** — Multi-room audio
- **ESPHome Voice** — Local wake word
- **Wyoming** — Voice pipelines

---

## 📋 Version History

| Version | Changes |
|---------|---------|
| **1.0.0** | Initial release — 6 providers, 15+ functions |

---

## 📄 License

MIT License — See [LICENSE](LICENSE)

---

## 🙏 Credits

Built with ❤️ for the Home Assistant community.

**⭐ Star this repo if PolyVoice helps you!**
