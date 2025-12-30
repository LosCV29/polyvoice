# 🔌 LLM Providers Guide

PolyVoice supports multiple LLM providers. Choose based on your priorities.

## Provider Comparison

| Provider | Type | Cost | Speed | Quality | Privacy |
|----------|------|------|-------|---------|---------|
| **LM Studio** | Local | FREE | Medium | Varies | ✅ Best |
| **OpenRouter** | Cloud | FREE tier | Fast | Excellent | Good |
| **Groq** | Cloud | FREE | ⚡ Fastest | Good | Good |
| **OpenAI** | Cloud | $$ | Fast | Excellent | Okay |
| **Anthropic** | Cloud | $$ | Fast | Excellent | Okay |
| **Google** | Cloud | FREE tier | Fast | Good | Okay |

## Recommended Setups

### 🏠 Privacy-Focused (100% Local)
```
Provider: LM Studio
Model: Qwen2.5-7B-Instruct or Llama-3.2-3B
Hardware: Any decent CPU, GPU optional
```

### 💰 Budget-Conscious (Free Cloud)
```
Provider: OpenRouter
Model: meta-llama/llama-3.3-70b-instruct:free
```

### ⚡ Speed-Focused
```
Provider: Groq
Model: llama-3.3-70b-versatile
```

### 🎯 Quality-Focused
```
Provider: OpenAI
Model: gpt-4o
```

---

## Provider Details

### LM Studio (Local)

**Pros:**
- 100% private - nothing leaves your network
- No API costs ever
- Works offline
- No rate limits

**Cons:**
- Requires capable hardware
- Model quality varies
- Setup required

**Setup:**
1. Download [LM Studio](https://lmstudio.ai)
2. Download a model (Qwen2.5-7B recommended)
3. Start local server (default: `http://localhost:1234/v1`)
4. Configure integration with the URL

**Recommended Models:**
| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| Qwen2.5-7B-Instruct | 6GB | Fast | Good |
| Llama-3.2-3B | 4GB | Very Fast | Okay |
| Qwen2.5-14B-Instruct | 12GB | Medium | Very Good |
| Qwen3-30B | 24GB | Slow | Excellent |

---

### OpenRouter

**Pros:**
- Access to many models through one API
- Free tier available
- Good model selection
- Fallback options

**Cons:**
- Cloud-based (privacy trade-off)
- Rate limits on free tier

**Setup:**
1. Create account at [openrouter.ai](https://openrouter.ai)
2. Get API key from dashboard
3. Configure integration with key

**Recommended Models:**
| Model | Cost | Quality |
|-------|------|---------|
| `meta-llama/llama-3.3-70b-instruct:free` | FREE | Excellent |
| `google/gemma-2-9b-it:free` | FREE | Good |
| `anthropic/claude-3.5-sonnet` | ~$0.003/req | Excellent |
| `openai/gpt-4o-mini` | ~$0.001/req | Very Good |

---

### Groq

**Pros:**
- Incredibly fast inference
- Free tier generous
- Great for voice (low latency)

**Cons:**
- Limited model selection
- Rate limits

**Setup:**
1. Create account at [groq.com](https://console.groq.com)
2. Get API key
3. Configure integration

**Recommended Models:**
| Model | Speed | Quality |
|-------|-------|---------|
| `llama-3.3-70b-versatile` | ⚡⚡⚡ | Excellent |
| `llama-3.1-8b-instant` | ⚡⚡⚡⚡ | Good |
| `mixtral-8x7b-32768` | ⚡⚡⚡ | Good |

---

### OpenAI

**Pros:**
- Highest quality models
- Reliable and stable
- Great tool calling

**Cons:**
- Costs money
- Privacy concerns

**Setup:**
1. Create account at [platform.openai.com](https://platform.openai.com)
2. Add payment method
3. Generate API key
4. Configure integration

**Recommended Models:**
| Model | Cost/1K tokens | Quality |
|-------|----------------|---------|
| `gpt-4o` | $0.005 | Best |
| `gpt-4o-mini` | $0.00015 | Very Good |
| `gpt-3.5-turbo` | $0.0005 | Good |

---

### Anthropic (Claude)

**Pros:**
- Excellent reasoning
- Great at following instructions
- Good for complex tasks

**Cons:**
- Costs money
- Tool calling slightly different

**Setup:**
1. Create account at [console.anthropic.com](https://console.anthropic.com)
2. Add payment method
3. Generate API key
4. Configure integration

**Recommended Models:**
| Model | Cost/1K tokens | Quality |
|-------|----------------|---------|
| `claude-sonnet-4-20250514` | $0.003 | Excellent |
| `claude-3-5-sonnet-20241022` | $0.003 | Excellent |
| `claude-3-haiku-20240307` | $0.00025 | Good |

---

### Google Gemini

**Pros:**
- Free tier available
- Good multimodal capabilities
- Fast

**Cons:**
- API can be quirky
- Quality varies

**Setup:**
1. Go to [Google AI Studio](https://aistudio.google.com)
2. Generate API key
3. Configure integration

**Recommended Models:**
| Model | Cost | Quality |
|-------|------|---------|
| `gemini-1.5-flash` | FREE | Good |
| `gemini-1.5-pro` | FREE tier | Very Good |
| `gemini-2.0-flash-exp` | FREE | Good |

---

## Switching Providers

You can change providers anytime:

1. Go to **Settings → Devices & Services**
2. Find **PolyVoice**
3. Click **Configure**
4. Select **Connection & Provider**
5. Choose new provider and enter credentials

Your feature configurations are preserved when switching providers.

---

## Hybrid Setup

Many users run a hybrid setup:

```
Primary: LM Studio (local)
  ↓ (if unavailable)
Fallback: OpenRouter (cloud)
```

To implement this, you'd need to set up two instances of the integration or handle failover in automations.

---

## Model Selection Tips

### For Voice Assistants
- Prioritize **speed** over quality
- 7B-8B parameter models are usually sufficient
- Groq is excellent for low-latency voice

### For Complex Tasks
- Use larger models (70B+)
- OpenAI GPT-4o or Claude for best results
- Accept slightly higher latency

### For Privacy
- Always use LM Studio local
- No cloud = no data leaving your home
- Qwen models work great locally
