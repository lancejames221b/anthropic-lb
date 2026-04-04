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
        ANTHROPIC_LB_PII        PII mode: regex | presidio | off (default: off)
        ANTHROPIC_LB_PII_RESPONSE  Response handling: detokenize | scan | off (default: detokenize)
        ANTHROPIC_LB_PII_PATTERNS  Path to custom patterns JSON (default: ./patterns.json)
        ANTHROPIC_LB_SPACY_MODEL   spaCy model for Presidio NER (default: en_core_web_lg)

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

# ---------------------------------------------------------------------------
# Optional Presidio + spaCy NER (Phase 2) — graceful import
# ---------------------------------------------------------------------------
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

PORT = int(os.environ.get("ANTHROPIC_LB_PORT", "8891"))
UPSTREAM = os.environ.get("ANTHROPIC_LB_UPSTREAM", "https://api.anthropic.com")
KEYS_FILE = os.environ.get("ANTHROPIC_LB_KEYS", "./keys.json")
STRATEGY = os.environ.get("ANTHROPIC_LB_STRATEGY", "least-loaded")

# PII configuration
PII_MODE = os.environ.get("ANTHROPIC_LB_PII", "off").lower()          # regex | presidio | off
PII_RESPONSE = os.environ.get("ANTHROPIC_LB_PII_RESPONSE", "detokenize").lower()  # detokenize | scan | off
PII_PATTERNS_FILE = os.environ.get("ANTHROPIC_LB_PII_PATTERNS", "./patterns.json")
SPACY_MODEL = os.environ.get("ANTHROPIC_LB_SPACY_MODEL", "en_core_web_lg")  # en_core_web_sm | en_core_web_md | en_core_web_lg


# ---------------------------------------------------------------------------
# PII Redaction — PIIVault + regex patterns
# ---------------------------------------------------------------------------

# Global PII statistics (cumulative across all requests)
PII_GLOBAL_STATS = {
    "total_redacted": 0,
    "by_type": defaultdict(int),
    "response_redacted": 0,
    "response_by_type": defaultdict(int),
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

    # US Social Security Numbers — context-required to avoid false positives
    # Only matches XXX-XX-XXXX when preceded by SSN-related keywords
    patterns.append((
        "ssn",
        re.compile(
            r'(?i)(?:ssn|social\s+security(?:\s+number)?|ss#|soc\s*sec)\s*[:#]?\s*'
            r'(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b',
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


# ---------------------------------------------------------------------------
# PresidioDetector — lazy-initialized NER engine (only used in presidio mode)
# ---------------------------------------------------------------------------

class PresidioDetector:
    """
    Wraps presidio-analyzer with a spaCy NLP engine.
    Model controlled by ANTHROPIC_LB_SPACY_MODEL (default: en_core_web_lg).
    Lazy-initialised on first call so import-time cost is zero.
    """

    # Entity types to detect via NER (structured PII is handled by regex)
    ENTITY_TYPES = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "DATE_TIME",
        "NRP",             # Nationality / Religious / Political group
        "MEDICAL_LICENSE",
    ]

    def __init__(self):
        self._engine = None

    def _init_engine(self):
        """Build the AnalyzerEngine with spaCy NLP backend (called once)."""
        global PRESIDIO_AVAILABLE
        try:
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()
            self._engine = AnalyzerEngine(
                nlp_engine=nlp_engine,
                supported_languages=["en"],
            )
        except OSError as e:
            # Typically: spaCy model not installed
            print(
                f"[pii] WARNING: presidio/spaCy init failed — {e}\n"
                f"  Install the model with: python -m spacy download {SPACY_MODEL}\n"
                "  Falling back to regex-only redaction.",
                file=sys.stderr,
            )
            PRESIDIO_AVAILABLE = False
            self._engine = None
        except Exception as e:
            print(
                f"[pii] WARNING: presidio init failed — {e}. Falling back to regex-only.",
                file=sys.stderr,
            )
            PRESIDIO_AVAILABLE = False
            self._engine = None

    def detect(self, text: str) -> list:
        """
        Run NER on *text* and return a list of
        (entity_type: str, start: int, end: int, matched_text: str) tuples.
        Returns [] if the engine is unavailable or text is empty.
        """
        global PRESIDIO_AVAILABLE
        if not PRESIDIO_AVAILABLE:
            return []
        if not text or not text.strip():
            return []
        if self._engine is None:
            self._init_engine()
        if self._engine is None:
            return []
        try:
            results = self._engine.analyze(
                text=text,
                entities=self.ENTITY_TYPES,
                language="en",
            )
            return [
                (r.entity_type, r.start, r.end, text[r.start:r.end])
                for r in results
            ]
        except Exception as e:
            print(f"[pii] presidio analyze error: {e}", file=sys.stderr)
            return []


# Module-level singleton — shared across all requests
_PRESIDIO_DETECTOR = PresidioDetector()


def _get_all_patterns():
    """Return builtin patterns + hot-reloaded custom patterns."""
    global _CUSTOM_PATTERN_MTIME, _CUSTOM_PATTERNS
    if PII_MODE in ("regex", "presidio") and os.path.exists(PII_PATTERNS_FILE):
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
    Run all PII patterns (and optionally Presidio NER) against `text` and
    replace matches with vault tokens.

    Strategy (Option A — operate on original text throughout):
    1. Collect all regex matches (start, end, type, value) from original text.
    2. If PII_MODE=="presidio", collect NER matches from original text.
    3. Filter NER matches that overlap with any regex match (no double-tokenisation).
    4. Merge all matches, sort descending by start position.
    5. Apply right-to-left so earlier positions stay valid.
    """
    if not text:
        return text

    # ------------------------------------------------------------------
    # Step 1: Collect regex matches on the original text
    # ------------------------------------------------------------------
    regex_matches = []  # list of (start, end, pii_type, original_value)
    for name, regex, pii_type, validator in patterns:
        for m in regex.finditer(text):
            full = m.group(0)
            if validator is not None:
                digits_only = re.sub(r'\D', '', full)
                if not validator(digits_only):
                    continue
            regex_matches.append((m.start(), m.end(), pii_type, full))

    # Build a list of (start, end) spans already covered by regex so we can
    # quickly test overlap below.
    regex_spans = [(s, e) for s, e, _, _ in regex_matches]

    def _overlaps(start: int, end: int) -> bool:
        """Return True if (start, end) overlaps with any regex span."""
        for rs, re_ in regex_spans:
            if start < re_ and end > rs:
                return True
        return False

    # ------------------------------------------------------------------
    # Step 2+3: Collect Presidio NER matches (presidio mode only)
    # ------------------------------------------------------------------
    ner_matches = []
    if PII_MODE == "presidio" and PRESIDIO_AVAILABLE:
        for entity_type, start, end, matched_text in _PRESIDIO_DETECTOR.detect(text):
            if not _overlaps(start, end):
                ner_matches.append((start, end, entity_type, matched_text))

    # ------------------------------------------------------------------
    # Step 4: Merge and sort descending (right-to-left application)
    # ------------------------------------------------------------------
    all_matches = regex_matches + ner_matches
    # Sort by start position DESCENDING so we can safely replace right→left
    all_matches.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Step 5: Apply replacements right-to-left
    # ------------------------------------------------------------------
    for start, end, pii_type, original_value in all_matches:
        token = vault.tokenize(original_value, pii_type)
        text = text[:start] + token + text[end:]

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


# ---------------------------------------------------------------------------
# Response-side PII scanning (Phase 3)
# ---------------------------------------------------------------------------

def _redact_for_response(text: str) -> tuple:
    """
    Scan text for PII and replace with [REDACTED_TYPE] placeholders.
    Uses non-reversible placeholders since the LLM may have hallucinated this data.
    Returns (redacted_text, count, by_type_dict).
    """
    if not text:
        return text, 0, {}

    patterns = _get_all_patterns()
    count = 0
    by_type: dict = defaultdict(int)

    # Collect all regex matches
    regex_matches = []  # (start, end, pii_type, original_value)
    for name, regex, pii_type, validator in patterns:
        for m in regex.finditer(text):
            full = m.group(0)
            if validator is not None:
                digits_only = re.sub(r'\D', '', full)
                if not validator(digits_only):
                    continue
            regex_matches.append((m.start(), m.end(), pii_type, full))

    # Build span coverage for overlap detection
    regex_spans = [(s, e) for s, e, _, _ in regex_matches]

    def _overlaps_resp(start: int, end: int) -> bool:
        for rs, re_ in regex_spans:
            if start < re_ and end > rs:
                return True
        return False

    # Presidio NER matches (presidio mode only)
    ner_matches = []
    if PII_MODE == "presidio" and PRESIDIO_AVAILABLE:
        for entity_type, start, end, matched_text in _PRESIDIO_DETECTOR.detect(text):
            if not _overlaps_resp(start, end):
                ner_matches.append((start, end, entity_type, matched_text))

    # Merge, deduplicate overlapping spans, sort descending
    all_matches = regex_matches + ner_matches
    # Remove overlapping spans (keep leftmost of each overlapping group)
    all_matches_sorted_asc = sorted(all_matches, key=lambda x: x[0])
    deduped = []
    last_end = -1
    for match in all_matches_sorted_asc:
        start, end, pii_type, original_value = match
        if start >= last_end:
            deduped.append(match)
            last_end = end

    # Apply right-to-left replacements
    deduped.sort(key=lambda x: x[0], reverse=True)
    for start, end, pii_type, original_value in deduped:
        placeholder = f"[REDACTED_{pii_type}]"
        text = text[:start] + placeholder + text[end:]
        count += 1
        by_type[pii_type] += 1

    return text, count, dict(by_type)


def _scan_response_body(response_body: bytes, vault: PIIVault) -> tuple:
    """
    For non-streaming responses with PII_RESPONSE=scan:
    1. Detokenize known vault tokens from the request
    2. Scan text content blocks for new PII the LLM may have generated
    Returns (response_body_bytes, response_redaction_count).
    """
    # First: detokenize vault tokens if any
    if vault.count > 0:
        response_body = vault.detokenize_bytes(response_body)

    # Parse response JSON and walk content blocks
    try:
        data = json.loads(response_body)
    except (json.JSONDecodeError, ValueError):
        return response_body

    total_count = 0
    total_by_type: dict = defaultdict(int)
    modified = False

    content_blocks = data.get("content", [])
    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                original_text = block["text"]
                scanned_text, count, by_type = _redact_for_response(original_text)
                if count > 0:
                    block["text"] = scanned_text
                    total_count += count
                    for pii_type, cnt in by_type.items():
                        total_by_type[pii_type] += cnt
                    modified = True

    if total_count > 0:
        PII_GLOBAL_STATS["response_redacted"] += total_count
        for pii_type, cnt in total_by_type.items():
            PII_GLOBAL_STATS["response_by_type"][pii_type] += cnt

    if modified:
        return json.dumps(data).encode(), total_count
    return response_body, total_count


# ---------------------------------------------------------------------------
# ResponsePIIScanner -- SSE streaming response scanner (Phase 3)
# ---------------------------------------------------------------------------

class ResponsePIIScanner:
    """
    Buffers SSE text deltas and scans for PII before flushing to client.

    Sliding-window approach:
    - Accumulates text from content_block_delta (text_delta) events
    - Flushes at sentence boundaries (.!? followed by space) or at 500 chars
    - Passes all non-text-delta events through unchanged
    - On content_block_stop: flushes remaining buffer, then passes stop event
    """

    FORCE_FLUSH_SIZE = 500  # force-flush threshold in characters
    # Safe flush: sentence-ending punctuation followed by a space
    _SAFE_FLUSH_RE = re.compile(r'[.!?]\s')

    def __init__(self, vault: PIIVault):
        self.vault = vault          # for detokenizing request-side tokens first
        self.buffer = ""            # accumulated unscanned text
        self.response_redactions = 0
        self._by_type: dict = defaultdict(int)
        # Track which content_block_index we're buffering for
        self._block_index: int = 0
        # Partial SSE line accumulator (handles chunks that split mid-line)
        self._partial_line = b""

    def _scan_and_flush(self, text: str) -> str:
        """
        Detokenize vault tokens, then scan for new PII.
        Returns the processed (safe-to-send) text.
        """
        # Step 1: restore any vault tokens the LLM echoed back
        if self.vault.count > 0:
            for token, original in self.vault._rev.items():
                text = text.replace(token, original)

        # Step 2: scan for new PII in the text
        scanned, count, by_type = _redact_for_response(text)
        if count > 0:
            self.response_redactions += count
            for pii_type, cnt in by_type.items():
                self._by_type[pii_type] += cnt

        return scanned

    def _find_safe_flush_pos(self) -> int:
        """
        Find the last sentence-end position in the buffer that's safe to flush.
        Returns the character position (exclusive end) or -1 if none found.
        """
        best = -1
        for m in self._SAFE_FLUSH_RE.finditer(self.buffer):
            # m.end() is after the space, which is a clean break point
            best = m.end()
        return best

    def _make_text_delta_event(self, text: str) -> bytes:
        """Construct a single SSE content_block_delta text_delta event."""
        event_data = json.dumps({
            "type": "content_block_delta",
            "index": self._block_index,
            "delta": {"type": "text_delta", "text": text},
        })
        return b"event: content_block_delta\ndata: " + event_data.encode() + b"\n\n"

    def _flush_buffer(self, force: bool = False) -> bytes:
        """
        Attempt to flush the buffer.
        If force=True, flush everything.
        Otherwise, flush up to the last safe position.
        Returns SSE bytes to write (may be empty).
        """
        if not self.buffer:
            return b""

        if force:
            flush_text = self.buffer
            self.buffer = ""
        else:
            pos = self._find_safe_flush_pos()
            if pos <= 0:
                # No safe point found -- only force-flush if buffer is large
                if len(self.buffer) >= self.FORCE_FLUSH_SIZE:
                    flush_text = self.buffer
                    self.buffer = ""
                else:
                    return b""
            else:
                flush_text = self.buffer[:pos]
                self.buffer = self.buffer[pos:]

        if not flush_text:
            return b""

        scanned = self._scan_and_flush(flush_text)
        if not scanned:
            return b""
        return self._make_text_delta_event(scanned)

    def process_chunk(self, chunk: bytes) -> bytes:
        """
        Process an SSE chunk. Buffers text_delta content and returns
        modified/passthrough bytes. May return empty bytes when buffering.
        """
        output = []
        # Prepend any partial line from previous chunk
        data = self._partial_line + chunk
        self._partial_line = b""

        # SSE streams are line-delimited; split on \\n
        # We collect lines into events, then process each complete event
        lines = data.split(b"\n")

        # If data doesn't end with \\n, the last piece is incomplete -- save it
        if not data.endswith(b"\n"):
            self._partial_line = lines[-1]
            lines = lines[:-1]

        # Group lines into events (SSE events are separated by blank lines)
        current_event_lines = []
        for line in lines:
            stripped = line.rstrip(b"\r")
            if stripped == b"":
                # Blank line = event boundary
                if current_event_lines:
                    event_out = self._process_event_lines(current_event_lines)
                    if event_out is not None:
                        output.append(event_out)
                    current_event_lines = []
            else:
                current_event_lines.append(stripped)

        # Any remaining lines are part of an incomplete event -- save for next chunk
        if current_event_lines:
            saved = b"\n".join(current_event_lines)
            if self._partial_line:
                self._partial_line = saved + b"\n" + self._partial_line
            else:
                self._partial_line = saved

        return b"".join(output)

    def _process_event_lines(self, lines: list) -> bytes:
        """
        Process a single complete SSE event (a list of stripped lines).
        Returns bytes to emit, or b"" for buffered events (never None).
        """
        # Reconstruct raw event bytes for pass-through
        raw = b"\n".join(lines) + b"\n\n"

        # Find event type and data payload
        event_type = None
        data_payload = None
        for line in lines:
            if line.startswith(b"event: "):
                event_type = line[7:].decode("utf-8", errors="replace").strip()
            elif line.startswith(b"data: "):
                data_payload = line[6:]

        if data_payload is None:
            # No data line (e.g. comment line starting with ':')
            return raw

        # Try to parse JSON payload
        try:
            parsed = json.loads(data_payload)
        except (json.JSONDecodeError, ValueError):
            return raw

        event_msg_type = parsed.get("type", "")

        # Handle content_block_delta with text_delta
        if event_msg_type == "content_block_delta":
            delta = parsed.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                # Track the block index for reconstructed events
                self._block_index = parsed.get("index", self._block_index)
                # Accumulate into buffer
                self.buffer += text
                # Attempt a non-forced flush at a safe boundary
                flushed = self._flush_buffer(force=False)
                return flushed  # may be b"" if still buffering

            else:
                # Non-text delta (e.g. input_json_delta for tool use) -- pass through
                return raw

        elif event_msg_type == "content_block_stop":
            # Flush remaining buffer first, then emit the stop event
            final = self._flush_buffer(force=True)
            return final + raw

        else:
            # All other events (message_start, content_block_start, ping,
            # message_delta, message_stop, error, etc.) -- pass through unchanged
            return raw

    def flush(self) -> bytes:
        """
        Force-flush any remaining buffer.
        Also merges scanner stats into global PII stats.
        Call at stream end to ensure nothing is left unscanned.
        """
        out = self._flush_buffer(force=True)
        # Merge per-scanner counts into global stats
        PII_GLOBAL_STATS["response_redacted"] += self.response_redactions
        for pii_type, cnt in self._by_type.items():
            PII_GLOBAL_STATS["response_by_type"][pii_type] += cnt
        # Reset to avoid double-counting if flush() is called again
        self.response_redactions = 0
        self._by_type = defaultdict(int)
        return out


def redact_request_body(body_bytes: bytes, vault: PIIVault) -> bytes:
    """
    Parse the request JSON body, walk message content and system field,
    tokenize PII in-place, and return the re-serialized body.
    Returns original bytes unchanged if body is not valid JSON.
    """
    if PII_MODE not in ("regex", "presidio"):
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
    if PII_MODE in ("regex", "presidio"):
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
            scanner = None
            try:
                if PII_RESPONSE == "scan" and PII_MODE != "off":
                    # Phase 3: SSE streaming with PII scanning via buffered window
                    scanner = ResponsePIIScanner(vault)
                    async for chunk in resp.content.iter_any():
                        collected_chunks.append(chunk)  # originals for usage tracking
                        processed = scanner.process_chunk(chunk)
                        if processed:
                            await stream.write(processed)
                    # Flush remaining buffered text
                    final = scanner.flush()
                    if final:
                        await stream.write(final)
                elif PII_RESPONSE == "detokenize" and vault.count > 0:
                    # Phase 1: simple token replacement per chunk
                    async for chunk in resp.content.iter_any():
                        chunk = vault.detokenize_bytes(chunk)
                        await stream.write(chunk)
                        collected_chunks.append(chunk)
                else:
                    # No PII processing — pass through
                    async for chunk in resp.content.iter_any():
                        await stream.write(chunk)
                        collected_chunks.append(chunk)
                await stream.write_eof()
            except (ConnectionResetError, asyncio.CancelledError):
                pass
            finally:
                update_stream_usage(name, collected_chunks)
                vault.flush_to_global_stats()
                # Add response-side scan header after streaming
                if scanner and scanner.response_redactions > 0:
                    # Can't add headers after prepare(), but stats are tracked
                    pass
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
            # PII Response Processing
            # ------------------------------------------------------------------
            resp_scan_count = 0
            if PII_RESPONSE == "scan" and PII_MODE != "off":
                # Phase 3: detokenize vault tokens + scan for LLM-generated PII
                response_body, resp_scan_count = _scan_response_body(response_body, vault)
            elif PII_RESPONSE == "detokenize" and vault.count > 0:
                # Phase 1: simple vault token replacement
                response_body = vault.detokenize_bytes(response_body)

            if resp_scan_count > 0:
                resp_headers["X-LB-PII-Response-Redacted"] = str(resp_scan_count)

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
            "presidio_available": PRESIDIO_AVAILABLE if PII_MODE == "presidio" else None,
            "response_mode": PII_RESPONSE,
            "total_redacted": PII_GLOBAL_STATS["total_redacted"],
            "by_type": dict(PII_GLOBAL_STATS["by_type"]),
            "response_redacted": PII_GLOBAL_STATS["response_redacted"],
            "response_by_type": dict(PII_GLOBAL_STATS["response_by_type"]),
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
    if PII_MODE == "presidio":
        if PRESIDIO_AVAILABLE:
            print(f"  pii mode: presidio (presidio: available, spacy model: {SPACY_MODEL})")
        else:
            print(
                f"  pii mode: presidio (WARNING: presidio not installed, falling back to regex)",
                file=sys.stderr,
            )
            print(f"  pii mode: presidio (WARNING: presidio not installed, falling back to regex)")
    elif PII_MODE != "off":
        print(f"  pii mode: {PII_MODE} (patterns: {PII_PATTERNS_FILE})")
    else:
        print(f"  pii mode: {PII_MODE}")
    web.run_app(app, port=PORT, print=None)
