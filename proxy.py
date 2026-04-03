#!/usr/bin/env python3
"""
anthropic-lb -- Usage-aware load balancer for the Anthropic API.

Routes requests to the account with the most remaining capacity based on
real-time rate limit headers from Anthropic. Falls back to round-robin
when no usage data is available yet.

Strategies:
    least-loaded    Route to account with most remaining tokens (default)
    round-robin     Simple per-request rotation

Configuration:
    Environment variables:
        ANTHROPIC_LB_PORT       Port to listen on (default: 8891)
        ANTHROPIC_LB_UPSTREAM   Upstream API URL (default: https://api.anthropic.com)
        ANTHROPIC_LB_KEYS       Path to keys file (default: ./keys.json)
        ANTHROPIC_LB_STRATEGY   Routing strategy (default: least-loaded)

        # PII redaction (Phase 1: regex-based)
        ANTHROPIC_LB_PII        PII mode: regex | off (default: off)
        ANTHROPIC_LB_PII_RESPONSE  Response handling: detokenize | off (default: detokenize)
        ANTHROPIC_LB_PII_PATTERNS  Path to custom patterns JSON (default: ./patterns.json)

    keys.json format:
        {
            "account-1": "sk-ant-...",
            "account-2": "sk-ant-...",
            "account-3": "sk-ant-..."
        }

    Or use environment variables directly:
        ANTHROPIC_KEY_1=sk-ant-...
        ANTHROPIC_KEY_2=sk-ant-...
        ANTHROPIC_KEY_3=sk-ant-...

Usage:
    python3 anthropic-lb.py

    Then set your client's base URL:
        ANTHROPIC_BASE_URL=http://localhost:8891

Author: Lance James, Unit 221B, Inc.
License: MIT
"""

import asyncio
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict

from aiohttp import web, ClientSession, ClientTimeout

PORT = int(os.environ.get("ANTHROPIC_LB_PORT", "8891"))
UPSTREAM = os.environ.get("ANTHROPIC_LB_UPSTREAM", "https://api.anthropic.com")
KEYS_FILE = os.environ.get("ANTHROPIC_LB_KEYS", "./keys.json")
STRATEGY = os.environ.get("ANTHROPIC_LB_STRATEGY", "least-loaded")

# PII configuration
PII_MODE = os.environ.get("ANTHROPIC_LB_PII", "off").lower()          # regex | off
PII_RESPONSE = os.environ.get("ANTHROPIC_LB_PII_RESPONSE", "detokenize").lower()  # detokenize | off
PII_PATTERNS_FILE = os.environ.get("ANTHROPIC_LB_PII_PATTERNS", "./patterns.json")


# ---------------------------------------------------------------------------
# PII Redaction — PIIVault + regex patterns
# ---------------------------------------------------------------------------

# Global PII statistics (cumulative across all requests)
PII_GLOBAL_STATS = {
    "total_redacted": 0,
    "by_type": defaultdict(int),
}


def _luhn_valid(number: str) -> bool:
    """Luhn algorithm check to validate credit card numbers."""
    digits = [int(c) for c in number if c.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _build_builtin_patterns():
    """
    Build the built-in Tier-1 regex PII patterns.
    Returns list of (name, compiled_regex, type_label, validator_fn_or_None).
    """
    patterns = []

    # Email addresses
    patterns.append((
        "email",
        re.compile(
            r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b',
        ),
        "EMAIL",
        None,
    ))

    # US Social Security Numbers  (XXX-XX-XXXX)
    patterns.append((
        "ssn",
        re.compile(
            r'\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b',
        ),
        "SSN",
        None,
    ))

    # Credit card numbers — 13-19 digits with common separators, Luhn-validated
    # Matches Visa, MC, Amex, Discover, etc.
    patterns.append((
        "credit_card",
        re.compile(
            r'\b(?:4[0-9]{12}(?:[0-9]{3,6})?'        # Visa
            r'|(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}'  # MC
            r'|3[47][0-9]{13}'                          # Amex
            r'|3(?:0[0-5]|[68][0-9])[0-9]{11}'         # Diners
            r'|6(?:011|5[0-9]{2})[0-9]{12,15}'         # Discover
            r'|(?:2131|1800|35\d{3})\d{11}'            # JCB
            r')\b'
        ),
        "CREDIT_CARD",
        _luhn_valid,
    ))

    # Phone numbers — US/international, common formats only
    # +1-XXX-XXX-XXXX, (XXX) XXX-XXXX, XXX-XXX-XXXX, +country-code...
    patterns.append((
        "phone",
        re.compile(
            r'(?<!\d)'
            r'(?:'
                r'\+1[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'  # +1 with area
            r'|'
                r'\(\d{3}\)\s?\d{3}[-.\s]\d{4}'                     # (XXX) XXX-XXXX
            r'|'
                r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'                  # XXX-XXX-XXXX
            r'|'
                r'\+\d{1,3}[-.\s]\d{1,4}[-.\s]\d{1,4}[-.\s]\d{1,9}'  # International
            r')'
            r'(?!\d)'
        ),
        "PHONE",
        None,
    ))

    # AWS Access Key IDs
    patterns.append((
        "aws_key",
        re.compile(
            r'\bAKIA[A-Z0-9]{16}\b',
        ),
        "API_KEY",
        None,
    ))

    # GitHub personal access tokens (ghp_*, ghs_*, ghr_*, github_pat_*)
    patterns.append((
        "github_token",
        re.compile(
            r'\b(?:ghp|ghs|ghr|gho|ghu)_[A-Za-z0-9]{36,}\b'
            r'|github_pat_[A-Za-z0-9_]{82}',
        ),
        "API_KEY",
        None,
    ))

    # Anthropic API keys
    patterns.append((
        "anthropic_key",
        re.compile(
            r'\bsk-ant-(?:api03|oat)[A-Za-z0-9\-_]{20,}\b',
        ),
        "API_KEY",
        None,
    ))

    # Generic Bearer tokens (Authorization: Bearer <token>)
    patterns.append((
        "bearer_token",
        re.compile(
            r'\bBearer\s+([A-Za-z0-9._\-]{20,})\b',
        ),
        "API_KEY",
        None,
    ))

    # IPv4 addresses (not localhost / private ranges are still redacted — caller decides)
    patterns.append((
        "ipv4",
        re.compile(
            r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b',
        ),
        "IP_ADDRESS",
        None,
    ))

    return patterns


def _load_custom_patterns(path: str):
    """
    Load custom regex patterns from a JSON file (hot-reloadable).
    Format: {"name": {"pattern": "...", "type": "LABEL"}}
    Returns list of (name, compiled_regex, type_label, None).
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            raw = json.load(f)
        result = []
        for name, cfg in raw.items():
            # Skip comment/metadata keys and non-dict values
            if not isinstance(cfg, dict):
                continue
            pat = cfg.get("pattern")
            label = cfg.get("type", name.upper())
            if pat:
                result.append((name, re.compile(pat), label, None))
        return result
    except Exception as e:
        print(f"[pii] Failed to load custom patterns from {path}: {e}", file=sys.stderr)
        return []


# Build pattern list once at startup; custom patterns are hot-reloaded per-request
_BUILTIN_PATTERNS = _build_builtin_patterns()
# mtime cache so we reload patterns.json only when it changes
_CUSTOM_PATTERN_MTIME = 0.0
_CUSTOM_PATTERNS = []


def _get_all_patterns():
    """Return builtin patterns + hot-reloaded custom patterns."""
    global _CUSTOM_PATTERN_MTIME, _CUSTOM_PATTERNS
    if PII_MODE == "regex" and os.path.exists(PII_PATTERNS_FILE):
        try:
            mtime = os.path.getmtime(PII_PATTERNS_FILE)
            if mtime != _CUSTOM_PATTERN_MTIME:
                _CUSTOM_PATTERNS = _load_custom_patterns(PII_PATTERNS_FILE)
                _CUSTOM_PATTERN_MTIME = mtime
        except OSError:
            pass
    return _BUILTIN_PATTERNS + _CUSTOM_PATTERNS


class PIIVault:
    """
    Per-request PII tokenization vault.

    Deterministic: same input value → same token within the request lifetime.
    Token format: __PII_{TYPE}_{8-char-md5-hash}__
    Discarded after the request-response cycle.
    """

    def __init__(self):
        # original_value → token
        self._fwd: dict[str, str] = {}
        # token → original_value  (for detokenization)
        self._rev: dict[str, str] = {}
        self.count = 0                      # items redacted this request
        self.by_type: dict[str, int] = defaultdict(int)

    def tokenize(self, value: str, pii_type: str) -> str:
        """Return (or create) a stable token for this value."""
        if value in self._fwd:
            return self._fwd[value]
        h8 = hashlib.md5(value.encode()).hexdigest()[:8]
        token = f"__PII_{pii_type}_{h8}__"
        self._fwd[value] = token
        self._rev[token] = value
        self.count += 1
        self.by_type[pii_type] += 1
        return token

    def detokenize(self, text: str) -> str:
        """Replace all tokens in text with their original values."""
        for token, original in self._rev.items():
            text = text.replace(token, original)
        return text

    def detokenize_bytes(self, data: bytes) -> bytes:
        """Byte-safe detokenization (for response bodies)."""
        for token, original in self._rev.items():
            data = data.replace(token.encode(), original.encode())
        return data

    def flush_to_global_stats(self):
        """Merge per-request counts into global PII statistics."""
        PII_GLOBAL_STATS["total_redacted"] += self.count
        for pii_type, cnt in self.by_type.items():
            PII_GLOBAL_STATS["by_type"][pii_type] += cnt


def _redact_text(text: str, vault: PIIVault, patterns) -> str:
    """
    Run all PII patterns against `text` and replace matches with vault tokens.
    Patterns are applied in order; earlier patterns take priority.
    """
    for name, regex, pii_type, validator in patterns:
        def replace_match(m):
            full = m.group(0)
            # For patterns with a capturing group (e.g. Bearer token body)
            # we tokenize the full match so context is preserved
            if validator is not None:
                # Strip non-digits for validation (credit cards)
                digits_only = re.sub(r'\D', '', full)
                if not validator(digits_only):
                    return full
            return vault.tokenize(full, pii_type)
        text = regex.sub(replace_match, text)
    return text


def _redact_content_block(block, vault: PIIVault, patterns):
    """
    Redact PII from a single Anthropic content block in-place.
    Handles both plain string content and the list-of-blocks format.
    """
    if isinstance(block, str):
        return _redact_text(block, vault, patterns)

    if isinstance(block, dict):
        block_type = block.get("type", "")
        if block_type == "text" and "text" in block:
            block["text"] = _redact_text(block["text"], vault, patterns)
        # tool_use input values may also contain PII
        elif block_type == "tool_use" and isinstance(block.get("input"), dict):
            for k, v in block["input"].items():
                if isinstance(v, str):
                    block["input"][k] = _redact_text(v, vault, patterns)
        # tool_result content
        elif block_type == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                block["content"] = _redact_text(inner, vault, patterns)
            elif isinstance(inner, list):
                block["content"] = [_redact_content_block(b, vault, patterns) for b in inner]

    return block


def redact_request_body(body_bytes: bytes, vault: PIIVault) -> bytes:
    """
    Parse the request JSON body, walk message content and system field,
    tokenize PII in-place, and return the re-serialized body.
    Returns original bytes unchanged if body is not valid JSON.
    """
    if PII_MODE != "regex":
        return body_bytes

    try:
        data = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        return body_bytes

    patterns = _get_all_patterns()

    # Redact system prompt (may be string or list of content blocks)
    system = data.get("system")
    if isinstance(system, str):
        data["system"] = _redact_text(system, vault, patterns)
    elif isinstance(system, list):
        data["system"] = [_redact_content_block(b, vault, patterns) for b in system]

    # Redact messages[].content
    for msg in data.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _redact_text(content, vault, patterns)
        elif isinstance(content, list):
            msg["content"] = [_redact_content_block(b, vault, patterns) for b in content]

    if vault.count == 0:
        # Nothing was redacted — return original bytes to avoid re-serialization overhead
        return body_bytes

    return json.dumps(data).encode()


# ---------------------------------------------------------------------------
# Key loading & account management (unchanged)
# ---------------------------------------------------------------------------

def load_keys():
    """Load API keys from keys.json or ANTHROPIC_KEY_N environment variables."""
    keys = {}

    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            keys = json.load(f)
        if keys:
            return keys

    i = 1
    while True:
        key = os.environ.get(f"ANTHROPIC_KEY_{i}")
        if not key:
            break
        name = os.environ.get(f"ANTHROPIC_KEY_{i}_NAME", f"account-{i}")
        keys[name] = key
        i += 1

    if not keys:
        print("No API keys found.", file=sys.stderr)
        print(f"  Create {KEYS_FILE} or set ANTHROPIC_KEY_1, ANTHROPIC_KEY_2, ...", file=sys.stderr)
        sys.exit(1)

    return keys


KEYS = load_keys()
KEY_NAMES = list(KEYS.keys())
KEY_CYCLE = itertools.cycle(range(len(KEY_NAMES)))
START_TIME = time.time()

# Per-account tracking
STATS = {}
for name in KEYS:
    STATS[name] = {
        "requests": 0,
        "errors": 0,
        "last_used": 0,
        # Token usage (from response bodies)
        "tokens_in": 0,
        "tokens_out": 0,
        "tokens_cache_read": 0,
        "tokens_cache_write": 0,
        # Rate limits (from response headers)
        "rate_requests_limit": None,
        "rate_requests_remaining": None,
        "rate_requests_reset": None,
        "rate_tokens_limit": None,
        "rate_tokens_remaining": None,
        "rate_tokens_reset": None,
        # 429 tracking
        "rate_limited_until": 0,
        "rate_limit_hits": 0,
    }


def update_rate_limits(name, headers):
    """Extract Anthropic rate limit headers and update account stats."""
    s = STATS[name]
    h = lambda k: headers.get(k)

    rl = h("anthropic-ratelimit-requests-limit")
    if rl:
        s["rate_requests_limit"] = int(rl)
    rr = h("anthropic-ratelimit-requests-remaining")
    if rr:
        s["rate_requests_remaining"] = int(rr)
    s["rate_requests_reset"] = h("anthropic-ratelimit-requests-reset")

    tl = h("anthropic-ratelimit-tokens-limit")
    if tl:
        s["rate_tokens_limit"] = int(tl)
    tr = h("anthropic-ratelimit-tokens-remaining")
    if tr:
        s["rate_tokens_remaining"] = int(tr)
    s["rate_tokens_reset"] = h("anthropic-ratelimit-tokens-reset")


def update_token_usage(name, body_bytes):
    """Extract token usage from response body (non-streaming only)."""
    try:
        data = json.loads(body_bytes)
        usage = data.get("usage", {})
        s = STATS[name]
        s["tokens_in"] += usage.get("input_tokens", 0)
        s["tokens_out"] += usage.get("output_tokens", 0)
        s["tokens_cache_read"] += usage.get("cache_read_input_tokens", 0)
        s["tokens_cache_write"] += usage.get("cache_creation_input_tokens", 0)
    except (json.JSONDecodeError, AttributeError):
        pass


def update_stream_usage(name, chunks):
    """Extract token usage from SSE stream (message_delta event has usage)."""
    try:
        for chunk in chunks:
            if b'"type":"message_delta"' in chunk or b'"type": "message_delta"' in chunk:
                for line in chunk.split(b"\n"):
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        usage = data.get("usage", {})
                        s = STATS[name]
                        s["tokens_out"] += usage.get("output_tokens", 0)
            if b'"type":"message_start"' in chunk or b'"type": "message_start"' in chunk:
                for line in chunk.split(b"\n"):
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        usage = data.get("message", {}).get("usage", {})
                        s = STATS[name]
                        s["tokens_in"] += usage.get("input_tokens", 0)
                        s["tokens_cache_read"] += usage.get("cache_read_input_tokens", 0)
                        s["tokens_cache_write"] += usage.get("cache_creation_input_tokens", 0)
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass


# ---------------------------------------------------------------------------
# Account selection (unchanged)
# ---------------------------------------------------------------------------

def pick_account_least_loaded():
    """Pick the account with the most remaining capacity.

    Priority:
    1. If rate limit headers are available, use tokens_remaining (real data)
    2. Otherwise, use least total tokens consumed (balanced distribution)
    3. Skip any account that is currently rate-limited (429 cooldown)
    """
    now = time.time()
    available = []

    for name in KEY_NAMES:
        s = STATS[name]
        if s["rate_limited_until"] > now:
            continue
        available.append(name)

    if not available:
        # All rate-limited: pick the one that resets soonest
        soonest = min(KEY_NAMES, key=lambda n: STATS[n]["rate_limited_until"])
        return soonest, KEYS[soonest]

    # Check if any account has real rate limit data
    has_rate_data = any(
        STATS[n]["rate_tokens_remaining"] is not None for n in available
    )

    if has_rate_data:
        # Use real rate limit data: pick highest remaining tokens
        best = max(available, key=lambda n: STATS[n]["rate_tokens_remaining"] or 0)
    else:
        # No rate limit headers: pick least-used by total tokens + request count
        # Request count breaks ties when token data hasn't arrived yet
        best = min(
            available,
            key=lambda n: (
                STATS[n]["tokens_in"] + STATS[n]["tokens_out"],
                STATS[n]["requests"],
            ),
        )

    return best, KEYS[best]


def pick_account_round_robin():
    """Simple round-robin selection."""
    idx = next(KEY_CYCLE)
    name = KEY_NAMES[idx]
    return name, KEYS[name]


def pick_account():
    if STRATEGY == "round-robin":
        return pick_account_round_robin()
    return pick_account_least_loaded()


# ---------------------------------------------------------------------------
# Main proxy handler — now with PII redaction/detokenization
# ---------------------------------------------------------------------------

async def proxy_handler(request: web.Request) -> web.StreamResponse:
    name, api_key = pick_account()
    STATS[name]["requests"] += 1
    STATS[name]["last_used"] = time.time()

    path = request.path
    if request.query_string:
        path += f"?{request.query_string}"

    headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding"):
            continue
        headers[k] = v
    # Set auth -- OAuth tokens (sk-ant-oat*) need Bearer + beta flag, API keys use x-api-key
    if api_key.startswith("sk-ant-oat"):
        headers["Authorization"] = f"Bearer {api_key}"
        headers.pop("x-api-key", None)
        # OAuth requires the beta flag to be accepted by the API
        existing_beta = headers.get("anthropic-beta", "")
        if "oauth-2025-04-20" not in existing_beta:
            beta_parts = [b for b in existing_beta.split(",") if b.strip()] + ["oauth-2025-04-20"]
            headers["anthropic-beta"] = ",".join(beta_parts)
    else:
        headers["x-api-key"] = api_key
        headers.pop("authorization", None)
        headers.pop("Authorization", None)

    body = await request.read()

    # ------------------------------------------------------------------
    # PII Redaction — tokenize PII in request body before forwarding
    # ------------------------------------------------------------------
    vault = PIIVault()
    if PII_MODE == "regex":
        body = redact_request_body(body, vault)
    # ------------------------------------------------------------------

    url = f"{UPSTREAM}{path}"
    timeout = ClientTimeout(total=600, sock_read=300)

    session = ClientSession(timeout=timeout)
    try:
        resp = await session.request(request.method, url, headers=headers, data=body)

        # On 429, mark account and try next
        if resp.status == 429:
            STATS[name]["errors"] += 1
            STATS[name]["rate_limit_hits"] += 1
            # Parse retry-after or default to 60s
            retry_after = resp.headers.get("retry-after")
            cooldown = int(retry_after) if retry_after and retry_after.isdigit() else 60
            STATS[name]["rate_limited_until"] = time.time() + cooldown
            STATS[name]["rate_tokens_remaining"] = 0
            await resp.release()

            name2, api_key2 = pick_account()
            STATS[name2]["requests"] += 1
            STATS[name2]["last_used"] = time.time()
            if api_key2.startswith("sk-ant-oat"):
                headers["Authorization"] = f"Bearer {api_key2}"
                headers.pop("x-api-key", None)
                existing_beta = headers.get("anthropic-beta", "")
                if "oauth-2025-04-20" not in existing_beta:
                    beta_parts = [b for b in existing_beta.split(",") if b.strip()] + ["oauth-2025-04-20"]
                    headers["anthropic-beta"] = ",".join(beta_parts)
            else:
                headers["x-api-key"] = api_key2
            resp = await session.request(request.method, url, headers=headers, data=body)
            name = name2

        # Capture rate limit headers
        update_rate_limits(name, resp.headers)

        # Build response headers
        resp_headers = {}
        for k, v in resp.headers.items():
            kl = k.lower()
            if kl in ("transfer-encoding", "content-encoding"):
                continue
            resp_headers[k] = v
        resp_headers["X-LB-Account"] = name

        # Add PII redaction count header
        if vault.count > 0:
            resp_headers["X-LB-PII-Redacted"] = str(vault.count)

        content_type = resp.headers.get("content-type", "")
        is_stream = "text/event-stream" in content_type

        if is_stream:
            stream = web.StreamResponse(status=resp.status, headers=resp_headers)
            stream.enable_chunked_encoding()
            await stream.prepare(request)
            collected_chunks = []
            try:
                async for chunk in resp.content.iter_any():
                    # ----------------------------------------------------------
                    # PII Detokenization — replace tokens in each SSE chunk
                    # Tokens are fixed strings so per-chunk replacement is safe
                    # ----------------------------------------------------------
                    if PII_RESPONSE == "detokenize" and vault.count > 0:
                        chunk = vault.detokenize_bytes(chunk)
                    await stream.write(chunk)
                    collected_chunks.append(chunk)
                await stream.write_eof()
            except (ConnectionResetError, asyncio.CancelledError):
                pass
            finally:
                update_stream_usage(name, collected_chunks)
                vault.flush_to_global_stats()
                await resp.release()
                await session.close()
            return stream
        else:
            response_body = await resp.read()
            if resp.status == 429:
                STATS[name]["errors"] += 1
                STATS[name]["rate_limit_hits"] += 1
            else:
                update_token_usage(name, response_body)

            # ------------------------------------------------------------------
            # PII Detokenization — replace tokens in full response body
            # ------------------------------------------------------------------
            if PII_RESPONSE == "detokenize" and vault.count > 0:
                response_body = vault.detokenize_bytes(response_body)

            vault.flush_to_global_stats()
            await resp.release()
            await session.close()
            return web.Response(
                body=response_body, status=resp.status, headers=resp_headers
            )
    except Exception as e:
        STATS[name]["errors"] += 1
        vault.flush_to_global_stats()
        await session.close()
        return web.Response(text=f"upstream error: {e}", status=502)


# ---------------------------------------------------------------------------
# Status / health endpoints
# ---------------------------------------------------------------------------

async def status_handler(request: web.Request) -> web.Response:
    now = time.time()
    uptime = now - START_TIME
    total_in = sum(s["tokens_in"] for s in STATS.values())
    total_out = sum(s["tokens_out"] for s in STATS.values())

    accounts = {}
    for name, s in STATS.items():
        rate_limited = s["rate_limited_until"] > now
        accounts[name] = {
            "requests": s["requests"],
            "errors": s["errors"],
            "last_used_ago": f"{now - s['last_used']:.0f}s" if s["last_used"] else "never",
            "tokens": {
                "input": s["tokens_in"],
                "output": s["tokens_out"],
                "cache_read": s["tokens_cache_read"],
                "cache_write": s["tokens_cache_write"],
                "total": s["tokens_in"] + s["tokens_out"],
            },
            "rate_limits": {
                "requests_remaining": s["rate_requests_remaining"],
                "requests_limit": s["rate_requests_limit"],
                "tokens_remaining": s["rate_tokens_remaining"],
                "tokens_limit": s["rate_tokens_limit"],
                "tokens_reset": s["rate_tokens_reset"],
                "rate_limited": rate_limited,
                "rate_limited_for": f"{s['rate_limited_until'] - now:.0f}s" if rate_limited else None,
                "total_429s": s["rate_limit_hits"],
            },
        }

    return web.json_response({
        "status": "ok",
        "strategy": STRATEGY,
        "uptime_seconds": int(uptime),
        "totals": {
            "requests": sum(s["requests"] for s in STATS.values()),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "tokens_total": total_in + total_out,
        },
        "accounts": accounts,
        # PII statistics (always included; zeroes when PII_MODE=off)
        "pii": {
            "mode": PII_MODE,
            "response_mode": PII_RESPONSE,
            "total_redacted": PII_GLOBAL_STATS["total_redacted"],
            "by_type": dict(PII_GLOBAL_STATS["by_type"]),
        },
    })


async def health_handler(request: web.Request) -> web.Response:
    now = time.time()
    available = sum(
        1 for s in STATS.values() if s["rate_limited_until"] <= now
    )
    return web.json_response({
        "ok": available > 0,
        "accounts": len(KEYS),
        "available": available,
        "strategy": STRATEGY,
    })


app = web.Application()
app.router.add_get("/status", status_handler)
app.router.add_get("/health", health_handler)
app.router.add_route("*", "/{path:.*}", proxy_handler)

if __name__ == "__main__":
    print(f"anthropic-lb starting on :{PORT}")
    print(f"  accounts: {', '.join(KEY_NAMES)} ({len(KEYS)} keys)")
    print(f"  upstream: {UPSTREAM}")
    print(f"  strategy: {STRATEGY}")
    print(f"  pii mode: {PII_MODE}" + (f" (patterns: {PII_PATTERNS_FILE})" if PII_MODE != "off" else ""))
    web.run_app(app, port=PORT, print=None)
