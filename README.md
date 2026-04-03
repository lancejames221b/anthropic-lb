# anthropic-lb

Usage-aware load balancer for the Anthropic API with built-in PII redaction. Routes requests to the account with the most remaining capacity. Automatically tokenizes PII before it reaches the LLM and restores it in responses. Single file, one required dependency.

## Why

If you have multiple Anthropic accounts (personal, team, org) you can spread API usage across all of them transparently. No client changes beyond pointing the base URL at the proxy.

The proxy also sits at the ideal chokepoint for PII protection -- every prompt passes through it, so you can strip sensitive data before it ever hits the Anthropic API.

- **Least-loaded routing** -- sends requests to the account with the most remaining capacity
- **PII redaction** -- regex-based detection of emails, phones, SSNs, credit cards, API keys, IPs
- **Reversible tokenization** -- PII replaced with deterministic tokens, restored in responses
- **ML-based NER** -- optional Presidio + spaCy for names, orgs, locations (opt-in)
- **Response scanning** -- catch PII the LLM generates, not just what was in the prompt
- **Real usage tracking** -- per-account input/output/cache token counts
- **Rate limit aware** -- reads Anthropic's rate limit headers
- **Automatic 429 failover** -- rate-limited account cools down, next request goes elsewhere
- Full SSE streaming passthrough with PII-aware buffering
- Custom PII patterns via JSON file with hot-reload
- Single file, ~1300 lines, one required dependency (`aiohttp`)

## Quick Start

```bash
git clone https://github.com/lancejames221b/anthropic-lb.git
cd anthropic-lb
pip install aiohttp

# Add your API keys
cp keys.json.example keys.json
# Edit keys.json with your actual keys

# Start with PII redaction enabled
ANTHROPIC_LB_PII=regex python3 proxy.py
```

Then point your client at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8891
```

## PII Redaction

### How It Works

```
Client: "Email john.doe@acme.com about Q3"
    |
    v
anthropic-lb detects PII, tokenizes:
    john.doe@acme.com -> __PII_EMAIL_a3f7b2c1__
    Stores mapping in per-request vault
    |
    v
Anthropic sees: "Email __PII_EMAIL_a3f7b2c1__ about Q3"
    |
    v
Response: "Drafted email to __PII_EMAIL_a3f7b2c1__..."
    |
    v
Proxy detokenizes: __PII_EMAIL_a3f7b2c1__ -> john.doe@acme.com
    |
    v
Client sees: "Drafted email to john.doe@acme.com..."
```

The LLM never sees real PII. The client gets a fully restored response. The vault is in-memory only -- never persisted, never logged.

### Two Detection Tiers

**Tier 1: Regex (default, zero extra dependencies)**

Sub-1ms overhead. Detects:
- Email addresses
- Phone numbers (US and international formats)
- Social Security Numbers
- Credit card numbers (Luhn-validated to avoid false positives)
- API keys (Anthropic `sk-ant-*`, GitHub `ghp_*`, AWS `AKIA*`, Bearer tokens)
- IPv4 addresses
- Custom patterns from `patterns.json`

**Tier 2: Presidio + spaCy NER (opt-in)**

~20-50ms overhead. Adds ML-based detection for:
- Person names
- Organizations
- Locations and addresses
- All 18 HIPAA Safe Harbor identifiers

Install the optional dependencies:
```bash
pip install -r requirements-presidio.txt
python -m spacy download en_core_web_lg
```

Regex always runs first (faster). Presidio catches what regex can't -- names like "John Smith" or orgs like "Acme Corp". Overlapping detections are automatically deduplicated.

### Response Scanning

Three modes for handling responses:

| Mode | What it does | Overhead |
|---|---|---|
| `detokenize` (default) | Restores vault tokens to original values | ~0ms |
| `scan` | Detokenizes + scans for NEW PII the LLM generated | 1-2s for streaming |
| `off` | No response processing | 0ms |

In `scan` mode, LLM-generated PII is replaced with `[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, etc. (non-reversible, since there's no original to restore).

For SSE streaming, a sliding window buffer accumulates text until a sentence boundary before scanning, ensuring PII that spans chunk boundaries is caught.

### Custom Patterns

Create a `patterns.json` for domain-specific PII:

```json
{
    "employee_id": {"pattern": "EMP-\\d{6}", "type": "EMPLOYEE_ID"},
    "project_code": {"pattern": "PRJ-[A-Z]{3}-\\d{4}", "type": "PROJECT_CODE"},
    "ticket_id": {"pattern": "TICKET-\\d{5,8}", "type": "TICKET_ID"}
}
```

Hot-reloaded when the file changes -- no restart needed.

## Configuration

```bash
# Server
ANTHROPIC_LB_PORT=8891              # default: 8891
ANTHROPIC_LB_UPSTREAM=https://api.anthropic.com
ANTHROPIC_LB_KEYS=./keys.json
ANTHROPIC_LB_STRATEGY=least-loaded  # or: round-robin

# PII redaction
ANTHROPIC_LB_PII=regex              # regex | presidio | off (default: off)
ANTHROPIC_LB_PII_RESPONSE=detokenize  # detokenize | scan | off (default: detokenize)
ANTHROPIC_LB_PII_PATTERNS=./patterns.json  # custom patterns (default: ./patterns.json)
ANTHROPIC_LB_SPACY_MODEL=en_core_web_lg  # en_core_web_sm (~50MB) | en_core_web_md (~40MB) | en_core_web_lg (~1.2GB)
```

### keys.json

```json
{
    "personal": "sk-ant-api03-...",
    "team": "sk-ant-api03-...",
    "org": "sk-ant-api03-..."
}
```

Or use environment variables:
```bash
ANTHROPIC_KEY_1=sk-ant-api03-...
ANTHROPIC_KEY_1_NAME=personal
ANTHROPIC_KEY_2=sk-ant-api03-...
```

## Load Balancing

### Strategies

| Strategy | Description |
|---|---|
| `least-loaded` (default) | Route to account with most remaining capacity. Uses rate limit headers when available, falls back to least-tokens-consumed. |
| `round-robin` | Simple per-request rotation. |

### 429 Failover

When an account returns 429:
1. Account enters cooldown (duration from `retry-after` header, default 60s)
2. Request is immediately retried on the next best account
3. Cooldown account is skipped until timer expires
4. If ALL accounts are rate-limited, requests go to the one that resets soonest

## Monitoring

```bash
# Health check
curl http://localhost:8891/health

# Full status
curl -s http://localhost:8891/status | python3 -m json.tool
```

Status includes:
- Per-account request counts, token usage, rate limits, 429 history
- PII stats: total redacted, breakdown by type, response-side redactions
- Uptime, strategy, PII mode

Response headers on every proxied request:
- `X-LB-Account` -- which account handled the request
- `X-LB-PII-Redacted` -- number of PII items tokenized in the request
- `X-LB-PII-Response-Redacted` -- number of PII items caught in the response (scan mode)

## Works With

- **Claude Code** -- `export ANTHROPIC_BASE_URL=http://localhost:8891`
- **Python/TypeScript SDK** -- `Anthropic(base_url="http://localhost:8891")`
- **Cursor / Windsurf / any AI editor** -- set the Anthropic base URL in settings
- **Agent frameworks** (OpenClaw, LangChain, CrewAI) -- point the provider URL at the proxy
- **curl / httpie** -- direct HTTP calls

The `x-api-key` header you send is replaced by the proxy. You can send any value or omit it entirely.

## systemd Service

```ini
[Unit]
Description=Anthropic API Load Balancer
After=network-online.target

[Service]
Type=simple
WorkingDirectory=/path/to/anthropic-lb
Environment=ANTHROPIC_LB_KEYS=/path/to/keys.json
Environment=ANTHROPIC_LB_PII=regex
Environment=ANTHROPIC_LB_PII_RESPONSE=detokenize
ExecStart=/path/to/venv/bin/python3 proxy.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

## License

MIT
