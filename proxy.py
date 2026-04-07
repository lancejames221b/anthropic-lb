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
import logging
from collections import defaultdict, deque
from datetime import datetime
from zoneinfo import ZoneInfo

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

# ---------------------------------------------------------------------------
# Optional Aho-Corasick for O(text_len + matches) known-secrets scanning
# ---------------------------------------------------------------------------
try:
    import ahocorasick
    AHOCORASICK_AVAILABLE = True
except ImportError:
    AHOCORASICK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional Redis for persistent PII vault backend
# ---------------------------------------------------------------------------
try:
    import redis as _redis_mod
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

PORT = int(os.environ.get("ANTHROPIC_LB_PORT", "8891"))
UPSTREAM = os.environ.get("ANTHROPIC_LB_UPSTREAM", "https://api.anthropic.com")
KEYS_FILE = os.environ.get("ANTHROPIC_LB_KEYS", "./keys.json")
STRATEGY = os.environ.get("ANTHROPIC_LB_STRATEGY", "least-loaded")

# Routing weights — configurable via env, no redeploy needed (Issue #12)
# W_7D=2.0: 7-day window takes ~34x longer to recover, so it carries double the penalty
ROUTING_W_5H = float(os.environ.get("ANTHROPIC_LB_W5H", "1.0"))
ROUTING_W_7D = float(os.environ.get("ANTHROPIC_LB_W7D", "2.0"))
# TTL thresholds: window penalty decays to zero as it approaches reset
ROUTING_TTL_5H_FULL = float(os.environ.get("ANTHROPIC_LB_TTL_5H", "1800"))    # 30 min
ROUTING_TTL_7D_FULL = float(os.environ.get("ANTHROPIC_LB_TTL_7D", "86400"))   # 24 h

# Backpressure — queue requests when all accounts are hot (Issue #10)
# Disabled by default (threshold > 1.0 means never triggers). Set to 0.9 to enable.
BACKPRESSURE_THRESHOLD = float(os.environ.get("ANTHROPIC_LB_BACKPRESSURE", "1.1"))
BACKPRESSURE_QUEUE_TIMEOUT = float(os.environ.get("ANTHROPIC_LB_QUEUE_TIMEOUT", "30"))
_backpressure_queue_depth = 0
_backpressure_total_queued = 0
_backpressure_total_timeouts = 0

# Budget throttle — soft cap on 5h utilization before routing degrades (Issue #15)
# At this threshold, score gets a steep penalty. At 1.0 Anthropic rejects outright.
BUDGET_5H_THRESHOLD = float(os.environ.get("ANTHROPIC_LB_BUDGET_5H", "0.6"))

# Peak-hour penalty — Anthropic shrinks quotas during 05:00-11:00 PT (March 2026)
# Multiplier applied to capacity scores during peak hours (0.6 = 40% penalty)
PEAK_HOUR_PENALTY = float(os.environ.get("ANTHROPIC_LB_PEAK_PENALTY", "0.6"))
PEAK_HOUR_START = int(os.environ.get("ANTHROPIC_LB_PEAK_START", "5"))   # hour in PT
PEAK_HOUR_END   = int(os.environ.get("ANTHROPIC_LB_PEAK_END", "11"))   # hour in PT
_TZ_PT = ZoneInfo("America/Los_Angeles")

# PII configuration
PII_MODE = os.environ.get("ANTHROPIC_LB_PII", "off").lower()          # regex | presidio | off
PII_RESPONSE = os.environ.get("ANTHROPIC_LB_PII_RESPONSE", "detokenize").lower()  # detokenize | scan | off
PII_PATTERNS_FILE = os.environ.get("ANTHROPIC_LB_PII_PATTERNS", "./patterns.json")
SPACY_MODEL = os.environ.get("ANTHROPIC_LB_SPACY_MODEL", "en_core_web_lg")  # en_core_web_sm | en_core_web_md | en_core_web_lg
PII_AUDIT_LOG = os.environ.get("ANTHROPIC_LB_PII_AUDIT_LOG", "")  # path to audit log file (empty = disabled)

# Known secrets source — exact-match layer that catches credentials no regex can find
# Supports: 1password (op CLI), bitwarden (bw CLI), file (JSON), or vault server
SECRETS_SOURCE = os.environ.get("ANTHROPIC_LB_SECRETS", "").lower()  # 1password | bitwarden | file | vault | off
SECRETS_FILE = os.environ.get("ANTHROPIC_LB_SECRETS_FILE", "./secrets.json")  # for file mode
SECRETS_VAULT = os.environ.get("ANTHROPIC_LB_SECRETS_VAULT", "")  # 1Password vault name filter
SECRETS_REFRESH = int(os.environ.get("ANTHROPIC_LB_SECRETS_REFRESH", "3600"))  # refresh interval in seconds

# Protected files — extract all values from .env, .json, .yaml, .ini, plain text
# Comma-separated paths/globs. Supports: .env, .json, .yaml/.yml, .ini/.cfg, .conf, plain text
PROTECT_FILES = os.environ.get("ANTHROPIC_LB_PROTECT_FILES", "")  # e.g. ".env,.env.*,keys.json,~/.ssh/config"

# Redis backend for PII vault persistence (optional, falls back to file-based)
REDIS_URL = os.environ.get("ANTHROPIC_LB_REDIS", "")  # redis://localhost:6379/0 or empty for file-based

# PII audit logger — writes redaction events to a dedicated log file
_pii_audit_logger = None
if PII_AUDIT_LOG:
    _pii_audit_logger = logging.getLogger("pii_audit")
    _pii_audit_logger.setLevel(logging.INFO)
    _pii_audit_logger.propagate = False
    _audit_handler = logging.FileHandler(PII_AUDIT_LOG)
    _audit_handler.setFormatter(logging.Formatter("%(message)s"))
    _pii_audit_logger.addHandler(_audit_handler)

# Module-level logger (configured in __main__ with basicConfig)
logger = logging.getLogger("anthropic-lb")


# ---------------------------------------------------------------------------
# Known Secrets — exact-match from password managers (1Password, Bitwarden, file)
# ---------------------------------------------------------------------------

import subprocess

class KnownSecretsStore:
    """
    Loads credentials from password managers and provides exact-match detection.
    Catches secrets that no regex can find (random passwords, custom tokens, etc.).

    Supports:
        1password  — uses `op` CLI (must be signed in)
        bitwarden  — uses `bw` CLI (must be unlocked)
        file       — loads from secrets.json [{label, value, type}]
        vault      — HTTP vault server

    Secrets are stored as (value, label, type) tuples. The proxy scans text
    for exact substring matches of each value and replaces with a synthetic.
    Values shorter than 6 chars are skipped (too many false positives).
    """

    MIN_SECRET_LEN = 6  # ignore short values like "yes" or "true"

    def __init__(self):
        self._secrets: list[tuple[str, str, str]] = []  # (value, label, type)
        self._last_refresh = 0.0
        self._loaded = False
        self._automaton = None  # pyahocorasick automaton (built on load)

    @property
    def count(self):
        return len(self._secrets)

    PREMAP_FILE = os.environ.get("ANTHROPIC_LB_SECRETS_PREMAP", "./secrets_premap.enc")
    PREMAP_KEY_ENV = "ANTHROPIC_LB_SECRETS_KEY"

    def _encrypt_premap(self, data: dict) -> bytes:
        """Encrypt the pre-map with AES-256-CBC via openssl."""
        key = os.environ.get(self.PREMAP_KEY_ENV, "")
        if not key:
            # No encryption key -- store as plain JSON (less secure but functional)
            return json.dumps(data).encode()
        result = subprocess.run(
            ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-salt", "-pass", f"pass:{key}"],
            input=json.dumps(data).encode(), capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"[secrets] Encryption failed: {result.stderr.decode()}", file=sys.stderr)
            return json.dumps(data).encode()
        return result.stdout

    def _decrypt_premap(self) -> dict:
        """Decrypt the pre-map from disk."""
        if not os.path.exists(self.PREMAP_FILE):
            return {}
        raw = open(self.PREMAP_FILE, "rb").read()
        key = os.environ.get(self.PREMAP_KEY_ENV, "")
        if not key:
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}
        result = subprocess.run(
            ["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2", "-pass", f"pass:{key}"],
            input=raw, capture_output=True, timeout=10,
        )
        if result.returncode != 0:
            print(f"[secrets] Decryption failed (wrong key?): {result.stderr.decode()}", file=sys.stderr)
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {}

    def _save_premap(self):
        """Save current secrets + pre-computed synthetics to encrypted disk cache."""
        generator = SyntheticPIIGenerator()
        premap = {}
        for value, label, stype in self._secrets:
            synthetic = generator.generate(stype, value)
            premap[value] = {"synthetic": synthetic, "label": label, "type": stype}
        encrypted = self._encrypt_premap(premap)
        with open(self.PREMAP_FILE, "wb") as f:
            f.write(encrypted)
        print(f"[secrets] Saved encrypted pre-map ({len(premap)} entries) to {self.PREMAP_FILE}")

    def load(self):
        """
        Load secrets from the configured source.
        Strategy:
        1. Try encrypted pre-map first (instant boot, no password manager call)
        2. If stale or missing, pull from source and rebuild pre-map
        """
        if not SECRETS_SOURCE or SECRETS_SOURCE == "off":
            return

        # Try pre-map first for instant boot
        premap = self._decrypt_premap()
        premap_fresh = False
        if premap:
            premap_age = time.time() - os.path.getmtime(self.PREMAP_FILE)
            if premap_age < SECRETS_REFRESH:
                self._secrets = [
                    (value, entry["label"], entry["type"])
                    for value, entry in premap.items()
                ]
                self._last_refresh = time.time()
                self._loaded = True
                premap_fresh = True
                print(f"[secrets] Loaded {len(self._secrets)} secrets from encrypted pre-map "
                      f"(age: {int(premap_age)}s)")

        # Pull from source if pre-map is stale or missing
        if not premap_fresh:
            try:
                if SECRETS_SOURCE == "1password":
                    self._load_1password()
                elif SECRETS_SOURCE == "bitwarden":
                    self._load_bitwarden()
                elif SECRETS_SOURCE == "file":
                    self._load_file()
                else:
                    print(f"[secrets] Unknown source: {SECRETS_SOURCE}", file=sys.stderr)
                    return
            except Exception as e:
                print(f"[secrets] Failed to load from {SECRETS_SOURCE}: {e}", file=sys.stderr)
                # Fall back to stale pre-map if available
                if premap:
                    self._secrets = [
                        (value, entry["label"], entry["type"])
                        for value, entry in premap.items()
                    ]
                    print(f"[secrets] Using stale pre-map ({len(self._secrets)} entries)")
                return

            self._last_refresh = time.time()
            self._loaded = True

        # Deduplicate by value
        seen = set()
        unique = []
        for val, label, stype in self._secrets:
            if val not in seen and len(val) >= self.MIN_SECRET_LEN:
                seen.add(val)
                unique.append((val, label, stype))
        self._secrets = unique

        # Save/refresh encrypted pre-map
        if not premap_fresh:
            try:
                self._save_premap()
            except Exception as e:
                print(f"[secrets] Failed to save pre-map: {e}", file=sys.stderr)

        # Build Aho-Corasick automaton for single-pass text scanning
        self._build_automaton()

        print(f"[secrets] {len(self._secrets)} known secrets active from {SECRETS_SOURCE}"
              f" (aho-corasick: {'yes' if self._automaton else 'no'})")

    def _build_automaton(self):
        """Build Aho-Corasick automaton from current secrets for O(N+M) scanning."""
        if not AHOCORASICK_AVAILABLE or not self._secrets:
            self._automaton = None
            return
        try:
            A = ahocorasick.Automaton()
            for value, label, stype in self._secrets:
                # Store (value, label, type) as the payload for each pattern
                A.add_word(value, (value, label, stype))
            A.make_automaton()
            self._automaton = A
        except Exception as e:
            print(f"[secrets] Failed to build Aho-Corasick automaton: {e}", file=sys.stderr)
            self._automaton = None

    def needs_refresh(self) -> bool:
        if not SECRETS_SOURCE or SECRETS_SOURCE == "off":
            return False
        return (time.time() - self._last_refresh) > SECRETS_REFRESH

    # 1Password field label → PII type mapping
    _OP_FIELD_TYPES = {
        # Login fields
        "password": "PASSWORD", "passwd": "PASSWORD", "master password": "PASSWORD",
        "username": "USERNAME", "login": "USERNAME", "user": "USERNAME",
        "email": "EMAIL", "e-mail": "EMAIL",
        "one-time password": "SECRET", "otp": "SECRET", "totp": "SECRET",
        # API / tokens
        "api key": "API_KEY", "apikey": "API_KEY", "api_key": "API_KEY",
        "key": "API_KEY", "token": "API_KEY", "secret": "SECRET",
        "access key": "API_KEY", "secret key": "API_KEY",
        "client id": "API_KEY", "client secret": "SECRET",
        # Identity fields
        "first name": "PERSON", "last name": "PERSON", "full name": "PERSON",
        "phone": "PHONE", "cell": "PHONE", "mobile": "PHONE", "telephone": "PHONE",
        "address": "LOCATION", "street": "LOCATION", "city": "LOCATION",
        "state": "LOCATION", "zip": "LOCATION", "postal code": "LOCATION",
        "country": "LOCATION",
        "social security number": "SSN", "ssn": "SSN",
        "date of birth": "DATE_OF_BIRTH", "dob": "DATE_OF_BIRTH", "birthday": "DATE_OF_BIRTH",
        "passport number": "PASSPORT", "passport": "PASSPORT",
        "driver's license": "DRIVERS_LICENSE", "license number": "DRIVERS_LICENSE",
        # Credit card fields
        "card number": "CREDIT_CARD", "number": "CREDIT_CARD",
        "cvv": "SECRET", "security code": "SECRET",
        "cardholder name": "PERSON", "cardholder": "PERSON",
        "expiry date": "SECRET", "expiration": "SECRET",
        # Bank
        "routing number": "SECRET", "account number": "SECRET",
        "iban": "IBAN", "swift": "SECRET", "bic": "SECRET",
    }

    def _classify_op_field(self, label: str, section: str = "", category: str = "") -> str:
        """Classify a 1Password field into a PII type."""
        label_lower = label.lower().strip()
        # Direct label match
        if label_lower in self._OP_FIELD_TYPES:
            return self._OP_FIELD_TYPES[label_lower]
        # Partial match
        for key, pii_type in self._OP_FIELD_TYPES.items():
            if key in label_lower:
                return pii_type
        # Category-based fallback
        if category in ("login", "password"):
            return "PASSWORD"
        if category == "credit_card":
            return "CREDIT_CARD"
        if category == "identity":
            return "SECRET"
        return "SECRET"

    def _load_1password(self):
        """
        Load from 1Password via `op` CLI.
        Pulls ALL item types: logins, identities, credit cards, API credentials,
        secure notes, and custom fields. Every value becomes an exact-match target.
        """
        vault_flag = ["--vault", SECRETS_VAULT] if SECRETS_VAULT else []

        # List all items (all categories)
        result = subprocess.run(
            ["op", "item", "list", "--format=json"] + vault_flag,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"op item list failed: {result.stderr.strip()}")
        items = json.loads(result.stdout)

        secrets = []
        for item in items:
            item_id = item.get("id", "")
            title = item.get("title", "unknown")
            category = item.get("category", "").lower().replace(" ", "_")

            # Get full item details
            detail = subprocess.run(
                ["op", "item", "get", item_id, "--format=json"] + vault_flag,
                capture_output=True, text=True, timeout=15,
            )
            if detail.returncode != 0:
                continue
            try:
                data = json.loads(detail.stdout)
            except json.JSONDecodeError:
                continue

            # Extract ALL fields
            for field in data.get("fields", []):
                value = field.get("value", "")
                label = field.get("label", "")
                section = field.get("section", {}).get("label", "") if isinstance(field.get("section"), dict) else ""
                if not value or len(value) < self.MIN_SECRET_LEN:
                    continue
                field_type = self._classify_op_field(label, section, category)
                secrets.append((value, f"{title}/{label}", field_type))

            # Extract URLs from login items
            for url_entry in data.get("urls", []):
                url = url_entry.get("href", "")
                if url and len(url) >= self.MIN_SECRET_LEN:
                    secrets.append((url, f"{title}/url", "URL"))

            # Secure notes body
            if category == "secure_note":
                note_field = next((f for f in data.get("fields", []) if f.get("id") == "notesPlain"), None)
                if note_field and note_field.get("value"):
                    # For notes, extract line-by-line values (each line might be a secret)
                    for line in note_field["value"].splitlines():
                        line = line.strip()
                        if len(line) >= self.MIN_SECRET_LEN and not line.startswith("#"):
                            secrets.append((line, f"{title}/note-line", "SECRET"))

        self._secrets = secrets

    def _load_bitwarden(self):
        """Load from Bitwarden via `bw` CLI. Pulls logins, cards, identities, notes."""
        result = subprocess.run(
            ["bw", "list", "items"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"bw list failed: {result.stderr.strip()}")

        items = json.loads(result.stdout)
        secrets = []
        for item in items:
            name = item.get("name", "unknown")

            # Login items
            login = item.get("login") or {}
            if login.get("username"):
                secrets.append((login["username"], f"{name}/username", "USERNAME"))
            if login.get("password"):
                secrets.append((login["password"], f"{name}/password", "PASSWORD"))
            if login.get("totp"):
                secrets.append((login["totp"], f"{name}/totp", "SECRET"))
            for uri in login.get("uris", []):
                u = uri.get("uri", "")
                if u and len(u) >= self.MIN_SECRET_LEN:
                    secrets.append((u, f"{name}/uri", "URL"))

            # Card items
            card = item.get("card") or {}
            if card.get("number"):
                secrets.append((card["number"], f"{name}/card_number", "CREDIT_CARD"))
            if card.get("code"):
                secrets.append((card["code"], f"{name}/cvv", "SECRET"))
            if card.get("cardholderName"):
                secrets.append((card["cardholderName"], f"{name}/cardholder", "PERSON"))

            # Identity items
            identity = item.get("identity") or {}
            for id_field, pii_type in [
                ("firstName", "PERSON"), ("lastName", "PERSON"),
                ("email", "EMAIL"), ("phone", "PHONE"),
                ("address1", "LOCATION"), ("address2", "LOCATION"),
                ("city", "LOCATION"), ("state", "LOCATION"),
                ("postalCode", "LOCATION"), ("country", "LOCATION"),
                ("ssn", "SSN"), ("passportNumber", "PASSPORT"),
                ("licenseNumber", "DRIVERS_LICENSE"),
                ("company", "ORGANIZATION"), ("username", "USERNAME"),
            ]:
                val = identity.get(id_field, "")
                if val and len(val) >= self.MIN_SECRET_LEN:
                    secrets.append((val, f"{name}/{id_field}", pii_type))

            # Custom fields
            for field in item.get("fields", []):
                val = field.get("value", "")
                if val and len(val) >= self.MIN_SECRET_LEN:
                    secrets.append((val, f"{name}/{field.get('name','field')}", "SECRET"))

            # Notes
            if item.get("notes") and len(item["notes"]) >= self.MIN_SECRET_LEN:
                for line in item["notes"].splitlines():
                    line = line.strip()
                    if len(line) >= self.MIN_SECRET_LEN:
                        secrets.append((line, f"{name}/note", "SECRET"))

        self._secrets = secrets

    def _load_file(self):
        """
        Load from a JSON file (plain or encrypted).

        Plain JSON format:
            [{"label": "AWS prod key", "value": "AKIA...", "type": "API_KEY"}, ...]
            Or simple: {"label": "value", ...}

        Encrypted (.enc) format:
            openssl enc -aes-256-cbc -pbkdf2 -salt -in secrets.json -out secrets.json.enc
            Set ANTHROPIC_LB_SECRETS_KEY env var with the passphrase.
        """
        path = SECRETS_FILE
        if not os.path.exists(path):
            raise FileNotFoundError(f"Secrets file not found: {path}")

        # Handle encrypted files
        if path.endswith(".enc") or path.endswith(".gpg"):
            passphrase = os.environ.get("ANTHROPIC_LB_SECRETS_KEY", "")
            if not passphrase:
                raise ValueError("ANTHROPIC_LB_SECRETS_KEY required for encrypted secrets file")

            if path.endswith(".gpg"):
                result = subprocess.run(
                    ["gpg", "--batch", "--yes", "--passphrase", passphrase, "--decrypt", path],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                result = subprocess.run(
                    ["openssl", "enc", "-d", "-aes-256-cbc", "-pbkdf2",
                     "-pass", f"pass:{passphrase}", "-in", path],
                    capture_output=True, text=True, timeout=10,
                )
            if result.returncode != 0:
                raise RuntimeError(f"Decryption failed: {result.stderr.strip()}")
            data = json.loads(result.stdout)
        else:
            with open(path) as f:
                data = json.load(f)

        if isinstance(data, list):
            self._secrets = [
                (s["value"], s.get("label", "unknown"), s.get("type", "SECRET"))
                for s in data if s.get("value")
            ]
        elif isinstance(data, dict):
            self._secrets = [
                (v, k, "SECRET") for k, v in data.items()
                if isinstance(v, str) and len(v) >= self.MIN_SECRET_LEN
            ]

    def find_in_text(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """
        Scan text for known secrets. Returns list of (start, end, type, value, label).
        Sorted by start position. Skips overlapping matches (leftmost/longest wins).

        Uses Aho-Corasick automaton for O(text_len + matches) when available,
        falls back to O(N * text_len) str.find() loop otherwise.
        """
        if not self._secrets or not text:
            return []

        # Refresh if stale
        if self.needs_refresh():
            try:
                self.load()
            except Exception:
                pass  # keep using old secrets

        if self._automaton:
            matches = self._find_ahocorasick(text)
        else:
            matches = self._find_linear(text)

        # Sort by start position, then longest match first (for overlap dedup)
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        deduped = []
        last_end = -1
        for m in matches:
            if m[0] >= last_end:
                deduped.append(m)
                last_end = m[1]

        return deduped

    def _find_ahocorasick(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """Single-pass Aho-Corasick scan. O(text_len + matches)."""
        matches = []
        for end_idx, (value, label, stype) in self._automaton.iter(text):
            start = end_idx - len(value) + 1
            matches.append((start, end_idx + 1, stype, value, label))
        return matches

    def _find_linear(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """Fallback O(N * text_len) str.find() loop."""
        matches = []
        for value, label, stype in self._secrets:
            start = 0
            while True:
                idx = text.find(value, start)
                if idx == -1:
                    break
                matches.append((idx, idx + len(value), stype, value, label))
                start = idx + len(value)
        return matches


# ---------------------------------------------------------------------------
# Protected Files — extract secrets from .env, .json, .yaml, .ini, config files
# ---------------------------------------------------------------------------

import glob as _glob_mod
import configparser

def _extract_secrets_from_file(path: str) -> list[tuple[str, str, str]]:
    """
    Parse a file and extract all values that could be secrets.
    Returns list of (value, label, type).
    Supports: .env, .json, .yaml/.yml, .ini/.cfg/.conf, plain text.
    """
    secrets = []
    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)

    try:
        if ext in (".env", "") and basename.startswith(".env"):
            secrets.extend(_extract_env(path))
        elif ext == ".json":
            secrets.extend(_extract_json(path))
        elif ext in (".yaml", ".yml"):
            secrets.extend(_extract_yaml(path))
        elif ext in (".ini", ".cfg", ".conf"):
            secrets.extend(_extract_ini(path))
        else:
            secrets.extend(_extract_lines(path))
    except Exception as e:
        print(f"[protect] Failed to parse {path}: {e}", file=sys.stderr)

    return secrets


def _classify_key(key: str) -> str:
    """Classify a config key name into a PII type."""
    k = key.lower()
    if any(w in k for w in ("password", "passwd", "pass")):
        return "PASSWORD"
    if any(w in k for w in ("secret", "token", "api_key", "apikey", "auth")):
        return "API_KEY"
    if any(w in k for w in ("email", "mail")):
        return "EMAIL"
    if any(w in k for w in ("user", "login", "username")):
        return "USERNAME"
    if any(w in k for w in ("host", "hostname", "server", "endpoint", "url", "uri", "webhook", "hook")):
        return "URL"
    if any(w in k for w in ("key", "cert", "pem", "private")):
        return "API_KEY"
    if any(w in k for w in ("phone", "mobile", "cell")):
        return "PHONE"
    if any(w in k for w in ("ssn", "social")):
        return "SSN"
    return "SECRET"


def _extract_env(path: str) -> list[tuple[str, str, str]]:
    """Parse .env file (KEY=VALUE, ignoring comments and empty lines)."""
    secrets = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")  # strip quotes
            if len(value) >= KnownSecretsStore.MIN_SECRET_LEN:
                secrets.append((value, f"{path}:{lineno}/{key}", _classify_key(key)))
    return secrets


def _extract_json(path: str) -> list[tuple[str, str, str]]:
    """Recursively extract all string values from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    secrets = []

    def _walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}/{k}" if prefix else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _walk(v, f"{prefix}[{i}]")
        elif isinstance(obj, str) and len(obj) >= KnownSecretsStore.MIN_SECRET_LEN:
            secrets.append((obj, f"{path}/{prefix}", _classify_key(prefix.split("/")[-1] if "/" in prefix else prefix)))

    _walk(data)
    return secrets


def _extract_yaml(path: str) -> list[tuple[str, str, str]]:
    """Extract values from YAML files (requires PyYAML, falls back to line parsing)."""
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, (dict, list)):
            return _extract_lines(path)

        secrets = []

        def _walk(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _walk(v, f"{prefix}/{k}" if prefix else str(k))
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    _walk(v, f"{prefix}[{i}]")
            elif isinstance(obj, str) and len(obj) >= KnownSecretsStore.MIN_SECRET_LEN:
                key = prefix.split("/")[-1] if "/" in prefix else prefix
                secrets.append((obj, f"{path}/{prefix}", _classify_key(key)))

        _walk(data)
        return secrets
    except ImportError:
        # No PyYAML -- fall back to basic key: value parsing
        return _extract_lines(path)


def _extract_ini(path: str) -> list[tuple[str, str, str]]:
    """Extract values from INI/CFG/CONF files."""
    secrets = []
    config = configparser.ConfigParser()
    config.read(path)
    for section in config.sections():
        for key, value in config.items(section):
            if len(value) >= KnownSecretsStore.MIN_SECRET_LEN:
                secrets.append((value, f"{path}/{section}/{key}", _classify_key(key)))
    return secrets


def _extract_lines(path: str) -> list[tuple[str, str, str]]:
    """
    Fall back: extract non-trivial lines as potential secrets.
    For config files with KEY=VALUE or KEY: VALUE patterns.
    Skips comments, blank lines, and lines shorter than threshold.
    """
    secrets = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//") or line.startswith(";"):
                continue
            # Try KEY=VALUE or KEY: VALUE
            for sep in ("=", ":"):
                if sep in line:
                    key, _, value = line.partition(sep)
                    value = value.strip().strip("'\"")
                    if len(value) >= KnownSecretsStore.MIN_SECRET_LEN:
                        secrets.append((value, f"{path}:{lineno}/{key.strip()}", _classify_key(key.strip())))
                    break
            else:
                # Whole line as a secret (for files like authorized_keys, token files)
                if len(line) >= KnownSecretsStore.MIN_SECRET_LEN:
                    secrets.append((line, f"{path}:{lineno}", "SECRET"))
    return secrets


def load_protected_files() -> list[tuple[str, str, str]]:
    """
    Load all protected files from PROTECT_FILES env var.
    Returns list of (value, label, type) extracted from all matching files.
    """
    if not PROTECT_FILES:
        return []

    all_secrets = []
    patterns = [p.strip() for p in PROTECT_FILES.split(",") if p.strip()]

    for pattern in patterns:
        # Expand ~ and env vars
        expanded = os.path.expanduser(os.path.expandvars(pattern))
        # Glob expand
        matched = _glob_mod.glob(expanded, recursive=True)
        if not matched:
            print(f"[protect] No files match: {pattern}", file=sys.stderr)
            continue
        for fpath in sorted(matched):
            if not os.path.isfile(fpath):
                continue
            extracted = _extract_secrets_from_file(fpath)
            if extracted:
                print(f"[protect] {fpath}: extracted {len(extracted)} values")
                all_secrets.extend(extracted)

    return all_secrets


# Module-level secrets store — loaded once at startup, refreshed periodically
_KNOWN_SECRETS = KnownSecretsStore()


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
    """Luhn algorithm check to validate credit card numbers.
    Accepts full match string -- strips non-digits internally."""
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

    # --- Issue #20: IPv6 addresses ---
    patterns.append((
        "ipv6",
        re.compile(
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'           # full form
            r'|'
            r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b'                         # trailing ::
            r'|'
            r'\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b'       # leading ::
            r'|'
            r'\b(?:[0-9a-fA-F]{1,4}:){1,5}:[0-9a-fA-F]{1,4}\b',       # mixed ::
        ),
        "IPV6_ADDRESS",
        None,
    ))

    # --- Issue #20: MAC addresses ---
    patterns.append((
        "mac_address",
        re.compile(
            r'\b(?:[0-9a-fA-F]{2}[:\-]){5}[0-9a-fA-F]{2}\b',
        ),
        "MAC_ADDRESS",
        None,
    ))

    # --- Issue #16: PEM private keys (multiline header detection) ---
    patterns.append((
        "pem_private_key",
        re.compile(
            r'-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----'
            r'[\s\S]*?'
            r'-----END\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+|ENCRYPTED\s+)?PRIVATE\s+KEY-----',
        ),
        "PRIVATE_KEY",
        None,
    ))

    # --- Issue #16: PEM certificates ---
    patterns.append((
        "pem_certificate",
        re.compile(
            r'-----BEGIN\s+CERTIFICATE-----[\s\S]*?-----END\s+CERTIFICATE-----',
        ),
        "CERTIFICATE",
        None,
    ))

    # --- Issue #16: JWT tokens ---
    def _jwt_valid(token):
        parts = token.split(".")
        return len(parts) == 3 and all(len(p) > 4 for p in parts)

    patterns.append((
        "jwt",
        re.compile(
            r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b',
        ),
        "JWT",
        _jwt_valid,
    ))

    # --- Issue #16: Slack tokens ---
    patterns.append((
        "slack_token",
        re.compile(
            r'\bxox[bpsar]-[A-Za-z0-9\-]{10,}\b',
        ),
        "API_KEY",
        None,
    ))

    # --- Issue #16: Stripe keys ---
    patterns.append((
        "stripe_key",
        re.compile(
            r'\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b',
        ),
        "API_KEY",
        None,
    ))

    # --- Issue #16: Twilio SID ---
    patterns.append((
        "twilio_key",
        re.compile(
            r'\bSK[0-9a-fA-F]{32}\b',
        ),
        "API_KEY",
        None,
    ))

    # --- Issue #16: SendGrid key ---
    patterns.append((
        "sendgrid_key",
        re.compile(
            r'\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b',
        ),
        "API_KEY",
        None,
    ))

    # --- Issue #16: Generic secrets (context-required) ---
    patterns.append((
        "generic_secret",
        re.compile(
            r'(?i)(?:password|passwd|secret|api_key|apikey|access_token|auth_token)'
            r'\s*[=:]\s*["\']?([A-Za-z0-9._\-/+]{8,})["\']?',
        ),
        "SECRET",
        None,
    ))

    # --- Issue #18: Bitcoin addresses ---
    patterns.append((
        "btc_p2pkh",
        re.compile(
            r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
        ),
        "CRYPTO_WALLET",
        None,
    ))
    patterns.append((
        "btc_bech32",
        re.compile(
            r'\bbc1[a-z0-9]{39,59}\b',
        ),
        "CRYPTO_WALLET",
        None,
    ))

    # --- Issue #18: Ethereum addresses ---
    patterns.append((
        "eth_address",
        re.compile(
            r'\b0x[0-9a-fA-F]{40}\b',
        ),
        "CRYPTO_WALLET",
        None,
    ))

    # --- Issue #15: US Passport numbers (context-required) ---
    patterns.append((
        "us_passport",
        re.compile(
            r'(?i)(?:passport)\s*(?:number|no|#|num)?\s*[:#]?\s*([A-Z]?\d{8,9})\b',
        ),
        "PASSPORT",
        None,
    ))

    # --- Issue #15: IBAN (International Bank Account Number) ---
    patterns.append((
        "iban",
        re.compile(
            r'\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?(?:[A-Z0-9]{4}\s?){2,7}[A-Z0-9]{1,4}\b',
        ),
        "IBAN",
        None,
    ))

    # --- Issue #15: US EIN (Employer Identification Number) ---
    patterns.append((
        "ein",
        re.compile(
            r'(?i)(?:ein|employer\s+id(?:entification)?(?:\s+number)?)\s*[:#]?\s*\d{2}-\d{7}\b',
        ),
        "EIN",
        None,
    ))

    # --- Issue #15: Drivers license (context-required) ---
    patterns.append((
        "drivers_license",
        re.compile(
            r'(?i)(?:driver\'?s?\s*(?:license|licence|lic)|DL)\s*(?:number|no|#|num)?\s*[:#]?\s*'
            r'([A-Z0-9]{5,15})\b',
        ),
        "DRIVERS_LICENSE",
        None,
    ))

    # --- Issue #23: Date of birth (context-required) ---
    patterns.append((
        "date_of_birth",
        re.compile(
            r'(?i)(?:d\.?o\.?b\.?|date\s+of\s+birth|birth\s*date|born\s+on|birthday)\s*[:#]?\s*'
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})',
        ),
        "DATE_OF_BIRTH",
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


def _load_synthetic_pools():
    """Load synthetic PII pools from JSON file (or use minimal fallback)."""
    pool_path = os.path.join(os.path.dirname(__file__) or ".", "synthetic_pools.json")
    try:
        with open(pool_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[pii] WARNING: Could not load {pool_path}: {e}. Using minimal fallback.", file=sys.stderr)
        return {}

_POOLS = _load_synthetic_pools()


class SyntheticPIIGenerator:
    """
    Generates realistic-looking fake PII values per type.
    Deterministic: same seed index always produces the same fake value.
    The LLM sees natural text instead of opaque tokens.
    Pools loaded from synthetic_pools.json (~200 names, ~180 surnames, 80 orgs, 64 cities).
    """

    _FIRST_NAMES = _POOLS.get("first_names", ["James", "Maria", "David", "Sarah", "Michael"])
    _LAST_NAMES = _POOLS.get("last_names", ["Miller", "Wilson", "Anderson", "Taylor", "Thomas"])
    _DOMAINS = _POOLS.get("domains", ["example.com", "example.net", "example.org"])
    _STREETS = _POOLS.get("streets", ["Oak", "Maple", "Cedar", "Pine", "Elm", "Main"])
    _SUFFIXES = _POOLS.get("street_suffixes", ["St", "Ave", "Blvd", "Dr", "Rd", "Ln", "Ct", "Way"])

    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)

    def generate(self, pii_type: str, original: str) -> str:
        """Generate a synthetic replacement for the given PII type."""
        idx = self._counters[pii_type]
        self._counters[pii_type] += 1
        # Use hash of original for stable sub-selections within the pools
        h = int(hashlib.md5(original.encode()).hexdigest(), 16)
        gen = getattr(self, f"_gen_{pii_type.lower()}", None)
        if gen:
            return gen(idx, h, original)
        # Fallback for unknown types: scramble to same length
        fake = hashlib.md5(original.encode()).hexdigest()
        return fake[:max(len(original), 8)]

    def _gen_email(self, idx, h, original):
        first = self._FIRST_NAMES[h % len(self._FIRST_NAMES)].lower()
        last = self._LAST_NAMES[(h >> 8) % len(self._LAST_NAMES)].lower()
        domain = self._DOMAINS[(h >> 16) % len(self._DOMAINS)]
        suffix = f"{idx}" if idx > 0 else ""
        return f"{first}.{last}{suffix}@{domain}"

    def _gen_phone(self, idx, h, original):
        # Preserve format shape: detect if original has parens, dashes, dots, +1 prefix
        area = 200 + (h % 800)
        mid = 100 + ((h >> 10) % 900)
        last4 = 1000 + ((h >> 20) % 9000) + idx
        if original.startswith("+1"):
            return f"+1-{area}-{mid}-{last4}"
        if original.startswith("("):
            return f"({area}) {mid}-{last4}"
        return f"{area}-{mid}-{last4}"

    def _gen_ssn(self, idx, h, original):
        # Keep the SSN keyword context from original, replace just the number part
        # Extract just the digits portion
        a1 = 100 + (h % 899)
        a2 = 10 + ((h >> 10) % 89)
        a3 = 1000 + ((h >> 20) % 8999) + idx
        fake_ssn = f"{a1}-{a2}-{a3}"
        # Find the number in the original and replace it
        import re as _re
        m = _re.search(r'\d{3}-\d{2}-\d{4}', original)
        if m:
            return original[:m.start()] + fake_ssn + original[m.end():]
        return fake_ssn

    def _gen_credit_card(self, idx, h, original):
        # Generate a Luhn-valid 16-digit fake card starting with 4 (Visa-like)
        prefix = f"400000{(h % 100000000):08d}{idx % 10}"[:15]
        digits = [int(c) for c in prefix]
        # Compute Luhn check digit (for the final position)
        total = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 0:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        check = (10 - (total % 10)) % 10
        return prefix + str(check)

    def _gen_ip_address(self, idx, h, original):
        o1 = 10
        o2 = (h % 250) + 1
        o3 = ((h >> 8) % 250) + 1
        o4 = ((h >> 16) % 250) + 1 + idx
        return f"{o1}.{o2}.{o3}.{o4}"

    def _gen_api_key(self, idx, h, original):
        # Generate realistic-looking fake keys that preserve the prefix format
        fake_suffix = hashlib.md5(f"{original}{idx}".encode()).hexdigest()
        if "sk-ant-api03" in original:
            return f"sk-ant-api03-{fake_suffix[:40]}"
        if "sk-ant-oat" in original:
            return f"sk-ant-oat01-{fake_suffix[:40]}"
        if "sk-ant-" in original:
            return f"sk-ant-api03-{fake_suffix[:40]}"
        if original.startswith("ghp_") or original.startswith("ghs_") or original.startswith("ghr_"):
            prefix = original[:4]
            return f"{prefix}{fake_suffix[:36]}"
        if original.startswith("AKIA"):
            return f"AKIA{fake_suffix[:16].upper()}"
        # Generic key-shaped string
        return f"tk_{fake_suffix[:32]}"

    def _gen_bearer_token(self, idx, h, original):
        fake_suffix = hashlib.md5(f"{original}{idx}".encode()).hexdigest()
        return f"Bearer {fake_suffix[:40]}"

    # --- NER entity generators (Presidio Phase 2) ---

    def _gen_person(self, idx, h, original):
        first = self._FIRST_NAMES[(h + idx) % len(self._FIRST_NAMES)]
        last = self._LAST_NAMES[((h >> 8) + idx) % len(self._LAST_NAMES)]
        # Preserve honorifics/titles if present
        prefixes = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Rev."]
        for p in prefixes:
            if original.startswith(p):
                return f"{p} {first} {last}"
        # Match word count of original name
        parts = original.split()
        if len(parts) == 1:
            return last
        if len(parts) >= 3:
            mid = self._FIRST_NAMES[((h >> 16) + idx) % len(self._FIRST_NAMES)]
            return f"{first} {mid[0]}. {last}"
        return f"{first} {last}"

    _ORG_PREFIXES = _POOLS.get("org_prefixes", ["Apex", "Summit", "Pinnacle", "Horizon"])
    _ORG_SUFFIXES = _POOLS.get("org_suffixes", ["Industries", "Corp", "Solutions", "Partners"])

    def _gen_organization(self, idx, h, original):
        prefix = self._ORG_PREFIXES[(h + idx) % len(self._ORG_PREFIXES)]
        suffix = self._ORG_SUFFIXES[((h >> 8) + idx) % len(self._ORG_SUFFIXES)]
        # Preserve Inc/LLC/Ltd if present
        for tag in [" Inc.", " Inc", " LLC", " Ltd.", " Ltd", " PLC", " GmbH", " S.A."]:
            if original.endswith(tag):
                return f"{prefix} {suffix}{tag}"
        return f"{prefix} {suffix}"

    _CITIES = _POOLS.get("cities", ["Riverside", "Fairview", "Springfield", "Greenville"])
    _STATES = _POOLS.get("states", ["CA", "TX", "NY", "FL", "IL", "PA"])

    def _gen_location(self, idx, h, original):
        city = self._CITIES[(h + idx) % len(self._CITIES)]
        state = self._STATES[((h >> 8) + idx) % len(self._STATES)]
        # If original looks like a full address (has digits), generate one
        if any(c.isdigit() for c in original):
            num = 100 + (h % 9900)
            street = self._STREETS[((h >> 12) + idx) % len(self._STREETS)]
            suffix = self._SUFFIXES[((h >> 16) + idx) % len(self._SUFFIXES)]
            return f"{num} {street} {suffix}, {city}, {state}"
        # If it has a comma (city, state pattern)
        if "," in original:
            return f"{city}, {state}"
        # Bare location name
        return city

    _MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    def _gen_date_time(self, idx, h, original):
        month = self._MONTHS[(h + idx) % 12]
        day = 1 + (h % 28)
        year = 1950 + ((h >> 8) % 60)
        # Try to match the format of the original
        import re as _re
        if _re.search(r'\d{4}-\d{2}-\d{2}', original):
            return f"{year}-{((h % 12) + 1):02d}-{day:02d}"
        if _re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', original):
            return f"{((h % 12) + 1)}/{day}/{year}"
        return f"{month} {day}, {year}"

    _NATIONALITIES = _POOLS.get("nationalities", ["Canadian", "Australian", "British", "German"])

    def _gen_nrp(self, idx, h, original):
        return self._NATIONALITIES[(h + idx) % len(self._NATIONALITIES)]

    def _gen_medical_license(self, idx, h, original):
        fake = hashlib.md5(f"{original}{idx}".encode()).hexdigest()[:8].upper()
        return f"ML-{fake}"

    def _gen_password(self, idx, h, original):
        # Generate a plausible password matching the original's length and complexity
        has_upper = any(c.isupper() for c in original)
        has_digit = any(c.isdigit() for c in original)
        has_special = any(c in "!@#$%^&*()-_=+[]{}|;:,.<>?" for c in original)
        pool = "abcdefghjkmnpqrstuvwxyz"
        if has_upper:
            pool += "ABCDEFGHJKLMNPQRSTUVWXYZ"
        if has_digit:
            pool += "23456789"
        if has_special:
            pool += "!@#$%&"
        length = len(original)
        return "".join(pool[(h >> (i % 32) + i * 7) % len(pool)] for i in range(length))

    def _gen_username(self, idx, h, original):
        first = self._FIRST_NAMES[(h + idx) % len(self._FIRST_NAMES)].lower()
        last = self._LAST_NAMES[((h >> 8) + idx) % len(self._LAST_NAMES)].lower()
        return f"{first}.{last}"

    def _gen_ipv6_address(self, idx, h, original):
        segs = [(h >> (i * 4)) & 0xFFFF for i in range(8)]
        segs[0] = (segs[0] + idx) & 0xFFFF
        return ":".join(f"{s:04x}" for s in segs)

    def _gen_mac_address(self, idx, h, original):
        octets = [(h >> (i * 8)) & 0xFF for i in range(6)]
        octets[0] = (octets[0] + idx) & 0xFF
        sep = ":" if ":" in original else "-"
        return sep.join(f"{o:02x}" for o in octets)

    def _gen_private_key(self, idx, h, original):
        fake_body = hashlib.md5(f"{original}{idx}".encode()).hexdigest() * 4
        return f"-----BEGIN PRIVATE KEY-----\n{fake_body[:64]}\n{fake_body[64:128]}\n-----END PRIVATE KEY-----"

    def _gen_certificate(self, idx, h, original):
        fake_body = hashlib.md5(f"{original}{idx}".encode()).hexdigest() * 4
        return f"-----BEGIN CERTIFICATE-----\n{fake_body[:64]}\n{fake_body[64:128]}\n-----END CERTIFICATE-----"

    def _gen_jwt(self, idx, h, original):
        import base64
        header = base64.urlsafe_b64encode(f'{{"alg":"HS256","idx":{idx}}}'.encode()).rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(f'{{"sub":"{h & 0xFFFFFFFF}","iat":1700000000}}'.encode()).rstrip(b"=").decode()
        sig = hashlib.md5(f"{original}{idx}".encode()).hexdigest()[:43]
        return f"eyJ{header}.eyJ{payload}.{sig}"

    def _gen_secret(self, idx, h, original):
        # Preserve the key= prefix, replace the value
        import re as _re
        m = _re.match(r'(?i)((?:password|passwd|secret|api_key|apikey|access_token|auth_token)\s*[=:]\s*["\']?)', original)
        prefix = m.group(1) if m else ""
        fake_val = hashlib.md5(f"{original}{idx}".encode()).hexdigest()[:max(len(original) - len(prefix), 12)]
        return f"{prefix}{fake_val}" if prefix else fake_val

    _HOST_PREFIXES = [
        "app", "web", "api", "db", "cache", "worker", "queue", "proxy",
        "monitor", "log", "build", "deploy", "test", "stage", "dev",
    ]
    _HOST_ENVS = ["prod", "staging", "dev", "test", "uat", "qa", "demo"]
    _HOST_DOMAINS = [
        "internal.net", "corp.local", "infra.io", "services.net",
        "cloud.local", "systems.io", "platform.net", "cluster.local",
    ]

    def _gen_url(self, idx, h, original):
        """Generate realistic URL/hostname/connection string synthetics."""
        import re as _re

        # Connection strings: protocol://[user]:[pass]@host[:port][/path]
        m = _re.match(r'^(\w+://)([^:]*):([^@]+)@([^:/]+)(:\d+)?(/.*)?\s*$', original)
        if m:
            proto, user, passwd, host, port, path = m.groups()
            port = port or ""
            path = path or ""
            fake_pass = hashlib.md5(f"{passwd}{idx}".encode()).hexdigest()[:max(len(passwd), 8)]
            fake_host = self._gen_hostname(idx, h, host)
            if user:
                fake_user = self._FIRST_NAMES[(h + idx) % len(self._FIRST_NAMES)].lower()
                return f"{proto}{fake_user}:{fake_pass}@{fake_host}{port}{path}"
            return f"{proto}:{fake_pass}@{fake_host}{port}{path}"

        # Full URLs with paths: https://hooks.slack.com/services/xxx
        m = _re.match(r'^(https?://)([^/]+)(/.*)$', original)
        if m:
            proto, domain, path = m.groups()
            fake_domain = self._gen_hostname(idx, h, domain)
            fake_path_parts = []
            for part in path.split("/"):
                if part and len(part) > 3:
                    fake_path_parts.append(hashlib.md5(f"{part}{idx}".encode()).hexdigest()[:len(part)])
                else:
                    fake_path_parts.append(part)
            return f"{proto}{fake_domain}{'/'.join(fake_path_parts)}"

        # URLs without paths: https://example.com
        m = _re.match(r'^(https?://)(.+)$', original)
        if m:
            proto, domain = m.groups()
            return f"{proto}{self._gen_hostname(idx, h, domain)}"

        # Bare hostnames
        return self._gen_hostname(idx, h, original)

    def _gen_hostname(self, idx, h, original):
        """Generate a realistic-looking hostname."""
        prefix = self._HOST_PREFIXES[(h + idx) % len(self._HOST_PREFIXES)]
        env = self._HOST_ENVS[((h >> 4) + idx) % len(self._HOST_ENVS)]
        num = (h % 10)
        domain = self._HOST_DOMAINS[((h >> 8) + idx) % len(self._HOST_DOMAINS)]

        # Try to match the shape of the original
        parts = original.replace("-", ".").split(".")
        if len(parts) >= 3:
            return f"{prefix}-{env}-{num:02d}.{domain}"
        if len(parts) == 2:
            return f"{prefix}-{env}.{domain.split('.')[0]}.net"
        return f"{prefix}-{env}-{num:02d}"

    def _gen_crypto_wallet(self, idx, h, original):
        if original.startswith("0x"):
            return f"0x{hashlib.md5(f'{original}{idx}'.encode()).hexdigest()[:40]}"
        if original.startswith("bc1"):
            chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"
            fake = "".join(chars[(h >> i) % len(chars)] for i in range(42))
            return f"bc1{fake}"
        # P2PKH/P2SH style
        chars = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        start = "1" if original.startswith("1") else "3"
        body = "".join(chars[(h >> i) % len(chars)] for i in range(33))
        return f"{start}{body}"

    def _gen_passport(self, idx, h, original):
        import re as _re
        m = _re.search(r'[A-Z]?\d{8,9}', original)
        if m:
            fake_num = f"{(h % 900000000) + 100000000}"
            return original[:m.start()] + fake_num + original[m.end():]
        return f"passport: {(h % 900000000) + 100000000}"

    def _gen_iban(self, idx, h, original):
        country = original[:2] if original[:2].isalpha() else "GB"
        check = f"{(h % 90) + 10:02d}"
        bank = f"{(h >> 8) % 10000:04d}"
        acct = f"{(h >> 20) % 100000000:08d}"
        return f"{country.upper()}{check} {bank} {acct[:4]} {acct[4:]}"

    def _gen_ein(self, idx, h, original):
        import re as _re
        m = _re.search(r'\d{2}-\d{7}', original)
        if m:
            fake = f"{(h % 90) + 10}-{(h >> 8) % 10000000:07d}"
            return original[:m.start()] + fake + original[m.end():]
        return f"{(h % 90) + 10}-{(h >> 8) % 10000000:07d}"

    def _gen_drivers_license(self, idx, h, original):
        import re as _re
        m = _re.search(r'[A-Z0-9]{5,15}', original)
        if m:
            fake = hashlib.md5(f"{original}{idx}".encode()).hexdigest()[:len(m.group(0))].upper()
            return original[:m.start()] + fake + original[m.end():]
        return hashlib.md5(f"{original}{idx}".encode()).hexdigest()[:10].upper()

    def _gen_date_of_birth(self, idx, h, original):
        import re as _re
        m = _re.search(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}|\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2}', original)
        if m:
            month = (h % 12) + 1
            day = (h >> 4) % 28 + 1
            year = 1950 + ((h >> 8) % 60)
            date_str = m.group(0)
            sep = "/" if "/" in date_str else "-" if "-" in date_str else "."
            if date_str[0:4].isdigit() and len(date_str) >= 10:
                fake = f"{year}{sep}{month:02d}{sep}{day:02d}"
            else:
                fake = f"{month:02d}{sep}{day:02d}{sep}{year}"
            return original[:m.start()] + fake + original[m.end():]
        return original


# ---------------------------------------------------------------------------
# Global PII mapping — persists across requests for session continuity
# ---------------------------------------------------------------------------
# Since synthetic generation is deterministic (md5-based), the same original
# always maps to the same synthetic. This global reverse lookup accumulates
# all mappings so that responses referencing PII from earlier turns can
# always be restored, even if the current request doesn't contain that PII.
#
# Storage backend: Redis (when ANTHROPIC_LB_REDIS set) or file-based (default).

PII_VAULT_FILE = os.environ.get("ANTHROPIC_LB_PII_VAULT", "./pii_vault.json")


class PIIVaultStore:
    """Abstract interface for the global PII reverse-lookup store."""

    def get(self, synthetic: str) -> str | None:
        raise NotImplementedError

    def get_type(self, synthetic: str) -> str | None:
        raise NotImplementedError

    def set(self, synthetic: str, original: str, pii_type: str):
        raise NotImplementedError

    def get_all(self) -> dict[str, str]:
        """Return dict of synthetic -> original for all entries."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def contains(self, synthetic: str) -> bool:
        raise NotImplementedError

    def bulk_set(self, mappings: dict[str, tuple[str, str]]):
        """Bulk insert: {synthetic: (original, pii_type)}."""
        for synthetic, (original, pii_type) in mappings.items():
            self.set(synthetic, original, pii_type)


class FilePIIVaultStore(PIIVaultStore):
    """File-backed PII vault store (JSON on disk). Default backend."""

    def __init__(self, vault_file: str):
        self._file = vault_file
        self._rev: dict[str, str] = {}       # synthetic -> original
        self._types: dict[str, str] = {}     # synthetic -> pii_type
        self._load()

    def _load(self):
        if not os.path.exists(self._file):
            return
        try:
            with open(self._file) as f:
                data = json.load(f)
            self._rev = data.get("rev", {})
            self._types = data.get("types", {})
            logger.info("[PII-VAULT] Loaded %d persisted mappings from %s", len(self._rev), self._file)
        except Exception as e:
            logger.warning("[PII-VAULT] Failed to load %s: %s (starting fresh)", self._file, e)

    def _save(self):
        try:
            with open(self._file, "w") as f:
                json.dump({"rev": self._rev, "types": self._types}, f)
        except Exception as e:
            logger.warning("[PII-VAULT] Failed to save %s: %s", self._file, e)

    def get(self, synthetic: str) -> str | None:
        return self._rev.get(synthetic)

    def get_type(self, synthetic: str) -> str | None:
        return self._types.get(synthetic)

    def set(self, synthetic: str, original: str, pii_type: str):
        is_new = synthetic not in self._rev
        self._rev[synthetic] = original
        self._types[synthetic] = pii_type
        if is_new:
            self._save()

    def get_all(self) -> dict[str, str]:
        return dict(self._rev)

    def count(self) -> int:
        return len(self._rev)

    def contains(self, synthetic: str) -> bool:
        return synthetic in self._rev

    def bulk_set(self, mappings: dict[str, tuple[str, str]]):
        changed = False
        for synthetic, (original, pii_type) in mappings.items():
            if synthetic not in self._rev:
                self._rev[synthetic] = original
                self._types[synthetic] = pii_type
                changed = True
        if changed:
            self._save()


class RedisPIIVaultStore(PIIVaultStore):
    """Redis-backed PII vault store. Uses hash keys pii:rev and pii:types."""

    def __init__(self, redis_url: str):
        self._client = _redis_mod.from_url(redis_url, decode_responses=True)
        self._rev_key = "pii:rev"
        self._types_key = "pii:types"
        try:
            self._client.ping()
            count = self._client.hlen(self._rev_key)
            logger.info("[PII-VAULT] Redis connected (%s), %d persisted mappings", redis_url, count)
        except Exception as e:
            raise ConnectionError(f"Redis connection failed ({redis_url}): {e}")

    def get(self, synthetic: str) -> str | None:
        return self._client.hget(self._rev_key, synthetic)

    def get_type(self, synthetic: str) -> str | None:
        return self._client.hget(self._types_key, synthetic)

    def set(self, synthetic: str, original: str, pii_type: str):
        pipe = self._client.pipeline()
        pipe.hset(self._rev_key, synthetic, original)
        pipe.hset(self._types_key, synthetic, pii_type)
        pipe.execute()

    def get_all(self) -> dict[str, str]:
        return self._client.hgetall(self._rev_key)

    def count(self) -> int:
        return self._client.hlen(self._rev_key)

    def contains(self, synthetic: str) -> bool:
        return self._client.hexists(self._rev_key, synthetic)

    def bulk_set(self, mappings: dict[str, tuple[str, str]]):
        if not mappings:
            return
        pipe = self._client.pipeline()
        for synthetic, (original, pii_type) in mappings.items():
            pipe.hset(self._rev_key, synthetic, original)
            pipe.hset(self._types_key, synthetic, pii_type)
        pipe.execute()


def _init_vault_store() -> PIIVaultStore:
    """Initialize the global PII vault store based on configuration."""
    if REDIS_URL:
        if not REDIS_AVAILABLE:
            logger.warning("[PII-VAULT] ANTHROPIC_LB_REDIS set but redis package not installed, "
                           "falling back to file-based vault")
        else:
            try:
                return RedisPIIVaultStore(REDIS_URL)
            except Exception as e:
                logger.warning("[PII-VAULT] Redis init failed (%s), falling back to file-based vault", e)
    return FilePIIVaultStore(PII_VAULT_FILE)


_vault_store: PIIVaultStore = _init_vault_store()


class PIIVault:
    """
    Per-request PII tokenization vault with synthetic replacement.

    Generates realistic-looking fake PII so the LLM can reason naturally
    about the data without knowing it's synthetic.

    Session-aware: each per-request vault merges its mappings into a global
    reverse lookup on flush. Detokenization checks both the per-request vault
    AND the global history, so responses referencing PII from earlier turns
    in the conversation are always restored correctly.

    Deterministic: same input value → same synthetic, always.
    """

    def __init__(self):
        # original_value → synthetic_value
        self._fwd: dict[str, str] = {}
        # synthetic_value → original_value  (for detokenization)
        self._rev: dict[str, str] = {}
        # synthetic_value → pii_type (for introspection / audit)
        self._types: dict[str, str] = {}
        self.count = 0
        self.by_type: dict[str, int] = defaultdict(int)
        self._generator = SyntheticPIIGenerator()

    def tokenize(self, value: str, pii_type: str) -> str:
        """Return (or create) a stable synthetic replacement for this value."""
        if value in self._fwd:
            return self._fwd[value]
        synthetic = self._generator.generate(pii_type, value)
        # Guard against collision (two different originals mapping to same synthetic)
        if synthetic in self._rev or _vault_store.contains(synthetic):
            existing_original = self._rev.get(synthetic) or _vault_store.get(synthetic)
            if existing_original != value:
                h = hashlib.md5(f"{value}_collision".encode()).hexdigest()
                synthetic = self._generator.generate(pii_type, f"{value}_{h[:8]}")
        self._fwd[value] = synthetic
        self._rev[synthetic] = value
        self._types[synthetic] = pii_type
        self.count += 1
        self.by_type[pii_type] += 1
        return synthetic

    def _build_full_rev(self) -> dict[str, str]:
        """Merge per-request + global reverse lookups. Per-request wins on conflict."""
        merged = _vault_store.get_all()
        merged.update(self._rev)  # per-request overrides global
        return merged

    def detokenize(self, text: str) -> str:
        """Replace all synthetic values in text with their originals.
        Checks both per-request vault and global history."""
        full_rev = self._build_full_rev()
        for synthetic, original in sorted(full_rev.items(), key=lambda x: -len(x[0])):
            text = text.replace(synthetic, original)
        return text

    def detokenize_bytes(self, data: bytes) -> bytes:
        """Byte-safe detokenization (for response bodies)."""
        full_rev = self._build_full_rev()
        for synthetic, original in sorted(full_rev.items(), key=lambda x: -len(x[0])):
            data = data.replace(synthetic.encode(), original.encode())
        return data

    def flush_to_global_stats(self):
        """Merge per-request counts into global PII statistics and global reverse lookup."""
        PII_GLOBAL_STATS["total_redacted"] += self.count
        for pii_type, cnt in self.by_type.items():
            PII_GLOBAL_STATS["by_type"][pii_type] += cnt
        # Persist mappings to global vault store for cross-request continuity
        if self._rev:
            new_mappings = {}
            for synthetic, original in self._rev.items():
                if not _vault_store.contains(synthetic):
                    pii_type = self._types.get(synthetic, "UNKNOWN")
                    new_mappings[synthetic] = (original, pii_type)
            if new_mappings:
                _vault_store.bulk_set(new_mappings)

    def audit_log(self, account: str, path: str):
        """Write a structured audit log entry if audit logging is enabled."""
        if not _pii_audit_logger or self.count == 0:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        types_summary = ", ".join(f"{t}:{c}" for t, c in sorted(self.by_type.items(), key=lambda x: -x[1]))
        entries = []
        for original, synthetic in self._fwd.items():
            pii_type = self._types.get(synthetic, "UNKNOWN")
            # Show first 3 and last 3 chars of original, mask the rest
            if len(original) > 8:
                masked = original[:3] + "*" * (len(original) - 6) + original[-3:]
            else:
                masked = original[:2] + "*" * max(len(original) - 2, 0)
            entries.append(f"    [{pii_type}] {masked} -> {synthetic}")
        detail = "\n".join(entries)
        _pii_audit_logger.info(
            f"[{ts}] account={account} path={path} redacted={self.count} types=[{types_summary}]\n{detail}"
        )


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
    # Step 0: Known secrets — exact match from password manager (highest priority)
    # ------------------------------------------------------------------
    secret_matches = []
    if _KNOWN_SECRETS.count > 0:
        for start, end, stype, value, label in _KNOWN_SECRETS.find_in_text(text):
            secret_matches.append((start, end, stype, value))

    # Build span coverage from known secrets (they take priority over regex/NER)
    secret_spans = [(s, e) for s, e, _, _ in secret_matches]

    def _in_secret_span(start: int, end: int) -> bool:
        for ss, se in secret_spans:
            if start < se and end > ss:
                return True
        return False

    # ------------------------------------------------------------------
    # Step 1: Collect regex matches on the original text
    # ------------------------------------------------------------------
    regex_matches = []  # list of (start, end, pii_type, original_value)
    for name, regex, pii_type, validator in patterns:
        for m in regex.finditer(text):
            # Skip if this span is already covered by a known secret
            if _in_secret_span(m.start(), m.end()):
                continue
            full = m.group(0)
            if validator is not None:
                if not validator(full):
                    continue
            regex_matches.append((m.start(), m.end(), pii_type, full))

    # Build a list of (start, end) spans already covered by regex+secrets
    all_covered = secret_spans + [(s, e) for s, e, _, _ in regex_matches]

    def _overlaps(start: int, end: int) -> bool:
        """Return True if (start, end) overlaps with any covered span."""
        for rs, re_ in all_covered:
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
    all_matches = secret_matches + regex_matches + ner_matches
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
                if not validator(full):
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
        # Step 1: restore any synthetic PII the LLM echoed back
        # Use full reverse lookup (per-request + global history)
        full_rev = self.vault._build_full_rev()
        if full_rev:
            for synthetic, original in sorted(full_rev.items(), key=lambda x: -len(x[0])):
                text = text.replace(synthetic, original)

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
# Sliding window rate limit tracker (Issue #6)
# ---------------------------------------------------------------------------

class SlidingWindowTracker:
    """
    Per-account sliding-window tracker for RPM and TPM over the last 60 seconds.
    Used to estimate capacity when Anthropic rate limit headers are unavailable.
    """
    WINDOW_SECONDS = 60

    def __init__(self, account_names):
        self._requests = {name: deque() for name in account_names}
        self._tokens   = {name: deque() for name in account_names}

    def _prune(self, account):
        cutoff = time.time() - self.WINDOW_SECONDS
        while self._requests[account] and self._requests[account][0] < cutoff:
            self._requests[account].popleft()
        while self._tokens[account] and self._tokens[account][0][0] < cutoff:
            self._tokens[account].popleft()

    def record_request(self, account, tokens_used=0):
        """Record a completed request for this account."""
        now = time.time()
        self._prune(account)
        self._requests[account].append(now)
        self._tokens[account].append((now, tokens_used))

    def get_rpm(self, account):
        """Return number of requests in the last 60 seconds."""
        self._prune(account)
        return len(self._requests[account])

    def get_tpm(self, account):
        """Return total tokens consumed in the last 60 seconds."""
        self._prune(account)
        return sum(t for _, t in self._tokens[account])


# ---------------------------------------------------------------------------
# Auth header helper (shared by initial request + cascade retry)
# ---------------------------------------------------------------------------

def _set_auth_headers(headers: dict, api_key: str):
    """Set authentication headers based on key type (OAuth vs API key)."""
    if api_key.startswith("sk-ant-oat"):
        headers["Authorization"] = f"Bearer {api_key}"
        headers.pop("x-api-key", None)
        existing_beta = headers.get("anthropic-beta", "")
        if "oauth-2025-04-20" not in existing_beta:
            beta_parts = [b for b in existing_beta.split(",") if b.strip()] + ["oauth-2025-04-20"]
            headers["anthropic-beta"] = ",".join(beta_parts)
    else:
        headers["x-api-key"] = api_key
        headers.pop("authorization", None)
        headers.pop("Authorization", None)


# ---------------------------------------------------------------------------
# Key loading & account management
# ---------------------------------------------------------------------------

def load_keys():
    """Load API keys from keys.json or ANTHROPIC_KEY_N environment variables.

    Supports two formats in keys.json (Issue #7):

    Simple format (backwards compatible):
        {"account-name": "sk-ant-..."}

    Extended format (with rate limit configuration and consumer affinity):
        {
            "account-name": {
                "key": "sk-ant-...",
                "limits": {"rpm": 50, "tpm": 40000, "tpd": 1000000},
                "affinity": ["cursor", "interactive"],
                "priority": 1
            }
        }

    Consumer affinity (Issue #15): when set, an account is preferred for
    requests whose X-LB-Consumer header matches one of the affinity tags.
    Priority (1=highest) controls preference order within affinity matches.

    Returns:
        (keys_dict, limits_dict, affinity_dict, priority_dict)
    """
    keys = {}
    limits = {}
    affinity = {}
    priority = {}

    if os.path.exists(KEYS_FILE):
        with open(KEYS_FILE) as f:
            raw = json.load(f)
        if raw:
            for name, value in raw.items():
                if name.startswith("_"):
                    continue
                if isinstance(value, str):
                    keys[name] = value
                elif isinstance(value, dict):
                    key = value.get("key", "")
                    if key:
                        keys[name] = key
                    if "limits" in value:
                        limits[name] = value["limits"]
                    if "affinity" in value:
                        affinity[name] = value["affinity"]
                    if "priority" in value:
                        priority[name] = value["priority"]
            if keys:
                if affinity:
                    logger.info("[CONFIG] consumer_affinity=%s", affinity)
                return keys, limits, affinity, priority

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

    return keys, limits, affinity, priority


KEYS, ACCOUNT_LIMITS, CONSUMER_AFFINITY, ACCOUNT_PRIORITY = load_keys()
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
        # Rate limits — standard API key accounts (x-api-key)
        "rate_requests_limit": None,
        "rate_requests_remaining": None,
        "rate_requests_reset": None,
        "rate_tokens_limit": None,
        "rate_tokens_remaining": None,
        "rate_tokens_reset": None,
        # Unified rate limits — OAuth/Max accounts (sk-ant-oat* keys)
        # Uses time-window utilization model instead of requests/tokens remaining
        "unified_status": None,              # 'allowed' | 'allowed_warning' | 'rejected'
        "unified_5h_utilization": None,      # 0.0-1.0 (five-hour session window)
        "unified_5h_reset": None,            # unix timestamp
        "unified_7d_utilization": None,      # 0.0-1.0 (seven-day weekly window)
        "unified_7d_reset": None,            # unix timestamp
        "unified_fallback": None,            # 'available' or None
        "unified_representative_claim": None,  # '5h' | '7d' | 'overage'
        # Routing scores — updated by pick_account_least_loaded() (Issues #12-14)
        "capacity_score": None,              # final score in [0, 1]
        "score_breakdown": None,             # dict with leniency, weights, velocity
        # 429 tracking
        "rate_limited_until": 0,
        "rate_limit_hits": 0,
    }


# Sliding window tracker — instantiated after KEY_NAMES is ready (Issue #6)
_window_tracker = SlidingWindowTracker(KEY_NAMES)

# Header audit counters — log all response headers for first 3 requests per account (Issue #5)
_HEADER_AUDIT_COUNT = {name: 0 for name in KEY_NAMES}


def update_rate_limits(name, headers):
    """Extract Anthropic rate limit headers and update account stats.

    Handles two distinct header schemes:
    - Standard API keys (x-api-key): anthropic-ratelimit-requests-* / tokens-*
    - OAuth/Max accounts (sk-ant-oat*): anthropic-ratelimit-unified-* (utilization model)
    """
    s = STATS[name]
    h = lambda k: headers.get(k)

    # ------------------------------------------------------------------
    # Standard API key headers (requests + tokens remaining)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Unified headers — OAuth/Max accounts (sk-ant-oat* keys)
    # Source: leaked Claude Code src/services/claudeAiLimits.ts
    # ------------------------------------------------------------------
    unified_status = h("anthropic-ratelimit-unified-status")
    if unified_status:
        s["unified_status"] = unified_status

    util_5h = h("anthropic-ratelimit-unified-5h-utilization")
    if util_5h is not None:
        s["unified_5h_utilization"] = float(util_5h)
    reset_5h = h("anthropic-ratelimit-unified-5h-reset")
    if reset_5h:
        s["unified_5h_reset"] = float(reset_5h)

    util_7d = h("anthropic-ratelimit-unified-7d-utilization")
    if util_7d is not None:
        s["unified_7d_utilization"] = float(util_7d)
    reset_7d = h("anthropic-ratelimit-unified-7d-reset")
    if reset_7d:
        s["unified_7d_reset"] = float(reset_7d)

    fallback = h("anthropic-ratelimit-unified-fallback")
    if fallback:
        s["unified_fallback"] = fallback
    claim = h("anthropic-ratelimit-unified-representative-claim")
    if claim:
        s["unified_representative_claim"] = claim

    if unified_status:
        logger.debug(
            "[UNIFIED-LIMITS] account=%s status=%s 5h_util=%.2f 7d_util=%.2f claim=%s",
            name, unified_status,
            s["unified_5h_utilization"] or 0.0,
            s["unified_7d_utilization"] or 0.0,
            claim or "?",
        )


def update_token_usage(name, body_bytes):
    """Extract token usage from response body (non-streaming only)."""
    try:
        data = json.loads(body_bytes)
        usage = data.get("usage", {})
        s = STATS[name]
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_write = usage.get("cache_creation_input_tokens", 0)
        s["tokens_in"] += inp
        s["tokens_out"] += out
        s["tokens_cache_read"] += cache_read
        s["tokens_cache_write"] += cache_write
        _window_tracker.record_request(name, inp + out)
        _log_cache_ratio(name, inp, out, cache_read, cache_write)
    except (json.JSONDecodeError, AttributeError):
        pass


def update_stream_usage(name, chunks):
    """Extract token usage from SSE stream (message_delta event has usage)."""
    total_tokens = 0
    stream_inp = 0
    stream_out = 0
    stream_cache_read = 0
    stream_cache_write = 0
    try:
        for chunk in chunks:
            if b'"type":"message_delta"' in chunk or b'"type": "message_delta"' in chunk:
                for line in chunk.split(b"\n"):
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        usage = data.get("usage", {})
                        s = STATS[name]
                        out = usage.get("output_tokens", 0)
                        s["tokens_out"] += out
                        stream_out += out
                        total_tokens += out
            if b'"type":"message_start"' in chunk or b'"type": "message_start"' in chunk:
                for line in chunk.split(b"\n"):
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:])
                        usage = data.get("message", {}).get("usage", {})
                        s = STATS[name]
                        inp = usage.get("input_tokens", 0)
                        cr = usage.get("cache_read_input_tokens", 0)
                        cw = usage.get("cache_creation_input_tokens", 0)
                        s["tokens_in"] += inp
                        s["tokens_cache_read"] += cr
                        s["tokens_cache_write"] += cw
                        stream_inp += inp
                        stream_cache_read += cr
                        stream_cache_write += cw
                        total_tokens += inp
    except (json.JSONDecodeError, AttributeError, KeyError):
        pass
    if total_tokens > 0:
        _window_tracker.record_request(name, total_tokens)
        _log_cache_ratio(name, stream_inp, stream_out, stream_cache_read, stream_cache_write)


# ---------------------------------------------------------------------------
# Cache ratio logging + peak-hour detection
# ---------------------------------------------------------------------------

def _log_cache_ratio(name: str, inp: int, out: int, cache_read: int, cache_write: int):
    """Log cache_read vs actual I/O ratio to detect cache inflation (Issue #15)."""
    io_total = inp + out
    if io_total == 0 and cache_read == 0:
        return
    all_tokens = io_total + cache_read + cache_write
    if all_tokens == 0:
        return
    cache_pct = (cache_read / all_tokens) * 100.0 if all_tokens else 0.0
    if cache_pct > 90.0:
        logger.warning(
            "[CACHE-RATIO] account=%s cache_read=%d io=%d cache_write=%d ratio=%.1f%% "
            "ALERT: cache reads dominating quota",
            name, cache_read, io_total, cache_write, cache_pct,
        )
    elif cache_read > 0:
        logger.info(
            "[CACHE-RATIO] account=%s cache_read=%d io=%d cache_write=%d ratio=%.1f%%",
            name, cache_read, io_total, cache_write, cache_pct,
        )


def _is_peak_hour() -> bool:
    """Return True if current time is within Anthropic peak hours (PT)."""
    pt_hour = datetime.now(_TZ_PT).hour
    return PEAK_HOUR_START <= pt_hour < PEAK_HOUR_END


# ---------------------------------------------------------------------------
# Account selection
# ---------------------------------------------------------------------------

def _compute_unified_score(name: str, s: dict, now: float) -> tuple[float, dict] | None:
    """Compute capacity score for an OAuth/unified-header account (Issues #12, #13, #14).

    Returns (score, breakdown) or None if account should be hard-skipped.

    Algorithm:
      1. Hard skip on rejected status
      2. TTL leniency: penalty decays to zero as window approaches reset
      3. Weighted combination: W_7D=2x because 7d window recovers ~34x slower
      4. allowed_warning → 30% penalty
      5. Velocity drag (only if rpm explicitly configured; OAuth accounts default to 0)
      6. Soft ceiling: score < 0.05 → 0.001 sentinel (routes, signals backpressure)
    """
    if s["unified_status"] == "rejected":
        logger.debug("[ROUTE] skip account=%s reason=unified_rejected", name)
        return None

    util_5h = s["unified_5h_utilization"] or 0.0
    util_7d = s["unified_7d_utilization"] or 0.0

    # --- TTL leniency (Issue #14) ---
    # Reduces penalty as a window approaches its reset timestamp.
    # If timestamps are absent, use representative_claim as a mid-window proxy.
    claim = s.get("unified_representative_claim")

    reset_5h = s.get("unified_5h_reset")
    if reset_5h:
        ttl_5h = max(0.0, reset_5h - now)
    else:
        # Proxy: five_hour claim → assume mid-window (2.5h left); else full penalty
        ttl_5h = 9000.0 if claim == "five_hour" else ROUTING_TTL_5H_FULL

    reset_7d = s.get("unified_7d_reset")
    if reset_7d:
        ttl_7d = max(0.0, reset_7d - now)
    else:
        # Proxy: seven_day claim → assume mid-window (3.5d left); else full penalty
        ttl_7d = 302400.0 if claim == "seven_day" else ROUTING_TTL_7D_FULL

    leniency_5h = min(1.0, ttl_5h / ROUTING_TTL_5H_FULL)
    leniency_7d = min(1.0, ttl_7d / ROUTING_TTL_7D_FULL)

    effective_5h = util_5h * leniency_5h
    effective_7d = util_7d * leniency_7d

    # --- Weighted combination (Issue #12) ---
    w_total = ROUTING_W_5H + ROUTING_W_7D
    weighted_util = (effective_5h * ROUTING_W_5H + effective_7d * ROUTING_W_7D) / w_total
    base_score = 1.0 - weighted_util

    # --- Status modifier ---
    if s["unified_status"] == "allowed_warning":
        base_score *= 0.7

    # --- Budget threshold (Issue #15) ---
    # Steep penalty when 5h utilization exceeds the budget threshold.
    # This prevents routing to accounts that are on track to exhaust their window.
    if util_5h >= BUDGET_5H_THRESHOLD:
        overage = (util_5h - BUDGET_5H_THRESHOLD) / (1.0 - BUDGET_5H_THRESHOLD)
        budget_penalty = overage * 0.5  # up to 50% penalty as util approaches 1.0
        base_score *= (1.0 - budget_penalty)
        logger.debug(
            "[ROUTE] budget_penalty account=%s util_5h=%.2f threshold=%.2f penalty=%.2f",
            name, util_5h, BUDGET_5H_THRESHOLD, budget_penalty,
        )

    # --- Peak-hour penalty (Issue #15) ---
    # Anthropic shrinks quotas during 05:00-11:00 PT. Apply multiplier to
    # conserve capacity when we know the window is effectively smaller.
    peak = _is_peak_hour()
    if peak:
        base_score *= PEAK_HOUR_PENALTY

    # --- Velocity penalty (Issue #13) ---
    # Only apply if rpm explicitly configured (OAuth accounts have no hard RPM cap)
    lims = ACCOUNT_LIMITS.get(name, {})
    configured_rpm = lims.get("rpm", 0)
    velocity_penalty = 0.0
    if configured_rpm > 0:
        recent_rpm = _window_tracker.get_rpm(name)
        velocity_ratio = recent_rpm / configured_rpm
        velocity_penalty = min(0.15, velocity_ratio * 0.15)
        base_score -= velocity_penalty

    # --- Soft ceiling ---
    # Don't hard-skip — return 0.001 sentinel so backpressure logic can detect
    # "all accounts marginal" vs "all accounts rejected"
    if base_score <= 0.05:
        if base_score <= 0.0:
            logger.debug(
                "[ROUTE] skip account=%s reason=unified_saturated 5h=%.2f 7d=%.2f score=%.3f",
                name, util_5h, util_7d, base_score,
            )
            return None
        base_score = 0.001  # sentinel

    breakdown = {
        "util_5h": util_5h,
        "util_7d": util_7d,
        "leniency_5h": round(leniency_5h, 3),
        "leniency_7d": round(leniency_7d, 3),
        "effective_5h": round(effective_5h, 3),
        "effective_7d": round(effective_7d, 3),
        "weighted_util": round(weighted_util, 3),
        "velocity_penalty": round(velocity_penalty, 3),
        "budget_5h_threshold": BUDGET_5H_THRESHOLD,
        "budget_over": util_5h >= BUDGET_5H_THRESHOLD,
        "peak_hour": peak,
        "peak_penalty": PEAK_HOUR_PENALTY if peak else 1.0,
        "w_5h": ROUTING_W_5H,
        "w_7d": ROUTING_W_7D,
    }

    logger.debug(
        "[ROUTE] score account=%s score=%.3f 5h=%.2f(x%.2f) 7d=%.2f(x%.2f) vel=%.3f peak=%s status=%s",
        name, base_score, util_5h, leniency_5h, util_7d, leniency_7d,
        velocity_penalty, peak, s["unified_status"],
    )
    return max(0.0, base_score), breakdown


def pick_account_least_loaded(consumer: str = ""):
    """Pick the account with the highest capacity score.

    Scoring priority:
    1. Skip accounts in 429 cooldown
    2. OAuth accounts: weighted multi-window score (Issues #12-14)
    3. Standard API key with headers: normalize remaining/limit
    4. Configured limits in keys.json: headroom from sliding window
    5. Fallback to least-used heuristic when no data exists
    6. Consumer affinity boost for accounts tagged for this consumer (Issue #15)
    """
    now = time.time()
    available = [n for n in KEY_NAMES if STATS[n]["rate_limited_until"] <= now]

    if not available:
        # All rate-limited: pick the one that resets soonest
        soonest = min(KEY_NAMES, key=lambda n: STATS[n]["rate_limited_until"])
        return soonest, KEYS[soonest]

    has_header_data = any(STATS[n]["rate_tokens_remaining"] is not None for n in available)

    scored = []
    score_breakdowns = {}
    for name in available:
        s = STATS[name]
        lims = ACCOUNT_LIMITS.get(name, {})
        is_oauth = KEYS[name].startswith("sk-ant-oat")

        if is_oauth and s["unified_status"] is not None:
            # OAuth/Max path: weighted multi-window score (Issues #12-14)
            result = _compute_unified_score(name, s, now)
            if result is None:
                continue
            score, breakdown = result
            score_breakdowns[name] = breakdown

        elif has_header_data and s["rate_tokens_remaining"] is not None:
            # Standard API key path: use real headers, normalize to [0, 1]
            token_limit = s["rate_tokens_limit"] or 100_000
            req_limit   = s["rate_requests_limit"] or 1_000
            token_score = (s["rate_tokens_remaining"] or 0) / token_limit
            req_score   = (s["rate_requests_remaining"] or 0) / req_limit
            score = min(token_score, req_score)

        elif lims:
            # Secondary path: sliding window + declared limits
            rpm_limit = lims.get("rpm", 0)
            tpm_limit = lims.get("tpm", 0)
            rpm_used  = _window_tracker.get_rpm(name)
            tpm_used  = _window_tracker.get_tpm(name)
            rpm_headroom = (rpm_limit - rpm_used) if rpm_limit else 1
            tpm_headroom = (tpm_limit - tpm_used) if tpm_limit else 1

            # Skip saturated accounts
            if rpm_limit and rpm_headroom <= 0:
                continue
            if tpm_limit and tpm_headroom <= 0:
                continue

            rpm_score = (rpm_headroom / rpm_limit) if rpm_limit else 1.0
            tpm_score = (tpm_headroom / tpm_limit) if tpm_limit else 1.0
            score = min(rpm_score, tpm_score)

        else:
            # Tertiary path: least-used heuristic
            total = s["tokens_in"] + s["tokens_out"]
            max_total = max(
                (STATS[n]["tokens_in"] + STATS[n]["tokens_out"]) for n in available
            ) or 1
            score = 1.0 - (total / max_total)

        scored.append((name, score))

    if not scored:
        # All accounts appear saturated — pick the one with lowest utilization
        def _util_key(n):
            s = STATS[n]
            u5 = s.get("unified_5h_utilization") or 0.0
            u7 = s.get("unified_7d_utilization") or 0.0
            return (u5 * ROUTING_W_5H + u7 * ROUTING_W_7D) / (ROUTING_W_5H + ROUTING_W_7D)
        best = min(available, key=_util_key) if available else KEY_NAMES[0]
        logger.warning(
            "[ROUTE] all_saturated fallback account=%s util_5h=%.2f util_7d=%.2f",
            best,
            STATS[best].get("unified_5h_utilization") or 0.0,
            STATS[best].get("unified_7d_utilization") or 0.0,
        )
        return best, KEYS[best]

    # --- Consumer affinity boost (Issue #15) ---
    # If caller identified itself, boost scores for accounts with matching affinity.
    if consumer and CONSUMER_AFFINITY:
        boosted = []
        for acct_name, score in scored:
            tags = CONSUMER_AFFINITY.get(acct_name, [])
            if consumer in tags:
                prio = ACCOUNT_PRIORITY.get(acct_name, 5)
                boost = max(0.01, 0.2 / prio)
                boosted.append((acct_name, min(1.0, score + boost)))
                logger.debug(
                    "[ROUTE] affinity_boost account=%s consumer=%s boost=+%.3f",
                    acct_name, consumer, boost,
                )
            else:
                boosted.append((acct_name, score))
        scored = boosted

    best_name, best_score = max(scored, key=lambda x: x[1])
    lims = ACCOUNT_LIMITS.get(best_name, {})

    # Store capacity scores and breakdowns in STATS for /status endpoint (step-4)
    for name, score in scored:
        STATS[name]["capacity_score"] = round(score, 4)
        if name in score_breakdowns:
            STATS[name]["score_breakdown"] = score_breakdowns[name]
    # Zero out scores for accounts not in this round (rate-limited or skipped)
    for name in KEY_NAMES:
        if name not in [n for n, _ in scored]:
            STATS[name]["capacity_score"] = 0.0

    logger.debug(
        "[ROUTE] selected=%s score=%.3f rpm_used=%s/%s tpm_used=%s/%s",
        best_name, best_score,
        _window_tracker.get_rpm(best_name), lims.get("rpm", "?"),
        _window_tracker.get_tpm(best_name), lims.get("tpm", "?"),
    )
    return best_name, KEYS[best_name]


def pick_account_round_robin():
    """Simple round-robin selection."""
    idx = next(KEY_CYCLE)
    name = KEY_NAMES[idx]
    return name, KEYS[name]


# ---------------------------------------------------------------------------
# OAuth usage poller — proactively fetch /api/oauth/usage for OAuth accounts
# ---------------------------------------------------------------------------

async def _poll_oauth_usage(session: ClientSession, name: str, api_key: str):
    """
    Fetch /api/oauth/usage for an OAuth account and update STATS with
    pre-request utilization data.

    Endpoint: GET {UPSTREAM}/api/oauth/usage
    Response: {five_hour, seven_day, seven_day_oauth_apps, seven_day_opus,
               seven_day_sonnet, extra_usage}
    Each window: {utilization: 0-100 (percent), resets_at: ISO8601}

    Source: leaked Claude Code src/services/api/usage.ts
    """
    if not api_key.startswith("sk-ant-oat"):
        return

    url = f"{UPSTREAM}/api/oauth/usage"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "anthropic-beta": "oauth-2025-04-20",
        "Content-Type": "application/json",
    }
    try:
        async with session.get(url, headers=headers, timeout=ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.debug("[OAUTH-USAGE] account=%s status=%s", name, resp.status)
                return
            data = await resp.json()
            s = STATS[name]

            # The API returns utilization as 0-100 percent; normalise to 0.0-1.0
            five_hour = data.get("five_hour") or {}
            seven_day = data.get("seven_day") or {}

            if five_hour.get("utilization") is not None:
                s["unified_5h_utilization"] = five_hour["utilization"] / 100.0
            if seven_day.get("utilization") is not None:
                s["unified_7d_utilization"] = seven_day["utilization"] / 100.0

            # Seed unified_status from utilization if not already set by headers
            if s["unified_status"] is None:
                worst = max(
                    s["unified_5h_utilization"] or 0.0,
                    s["unified_7d_utilization"] or 0.0,
                )
                s["unified_status"] = "rejected" if worst >= 1.0 else "allowed"

            logger.info(
                "[OAUTH-USAGE] account=%s 5h=%.1f%% 7d=%.1f%%",
                name,
                (s["unified_5h_utilization"] or 0.0) * 100,
                (s["unified_7d_utilization"] or 0.0) * 100,
            )
    except Exception as e:
        logger.debug("[OAUTH-USAGE] account=%s error=%s", name, e)


async def _oauth_usage_poll_loop():
    """Background task: poll /api/oauth/usage every 60s for all OAuth accounts."""
    oauth_accounts = [(n, k) for n, k in KEYS.items() if k.startswith("sk-ant-oat")]
    if not oauth_accounts:
        return
    async with ClientSession() as session:
        while True:
            for name, api_key in oauth_accounts:
                await _poll_oauth_usage(session, name, api_key)
            await asyncio.sleep(60)


def pick_account(consumer: str = ""):
    if STRATEGY == "round-robin":
        return pick_account_round_robin()
    return pick_account_least_loaded(consumer=consumer)


def _all_accounts_above_threshold() -> bool:
    """Return True if every account is above the backpressure utilization threshold."""
    if BACKPRESSURE_THRESHOLD > 1.0:
        return False  # Backpressure disabled
    now = time.time()
    for name in KEY_NAMES:
        s = STATS[name]
        # Skip accounts in hard 429 cooldown
        if s["rate_limited_until"] > now:
            continue
        # Check unified utilization (OAuth accounts)
        util_5h = s["unified_5h_utilization"] or 0.0
        util_7d = s["unified_7d_utilization"] or 0.0
        worst = max(util_5h, util_7d)
        if worst < BACKPRESSURE_THRESHOLD:
            return False
    return True


async def _backpressure_wait() -> bool:
    """Wait for capacity to free up. Returns True if capacity found, False on timeout."""
    global _backpressure_queue_depth, _backpressure_total_queued, _backpressure_total_timeouts
    _backpressure_queue_depth += 1
    _backpressure_total_queued += 1
    start = time.time()
    poll_interval = 1.0  # Start at 1s, backoff up to 4s
    try:
        while time.time() - start < BACKPRESSURE_QUEUE_TIMEOUT:
            await asyncio.sleep(poll_interval)
            if not _all_accounts_above_threshold():
                logger.info(
                    "[BACKPRESSURE] released after %.1fs queue_depth=%d",
                    time.time() - start, _backpressure_queue_depth,
                )
                return True
            poll_interval = min(poll_interval * 1.5, 4.0)
        # Timeout
        _backpressure_total_timeouts += 1
        logger.warning(
            "[BACKPRESSURE] timeout after %.1fs queue_depth=%d",
            time.time() - start, _backpressure_queue_depth,
        )
        return False
    finally:
        _backpressure_queue_depth -= 1


# ---------------------------------------------------------------------------
# Main proxy handler — now with PII redaction/detokenization
# ---------------------------------------------------------------------------

async def proxy_handler(request: web.Request) -> web.StreamResponse:
    # ------------------------------------------------------------------
    # Backpressure — queue request if all accounts are hot (Issue #10)
    # ------------------------------------------------------------------
    queued_seconds = 0.0
    if _all_accounts_above_threshold():
        bp_start = time.time()
        capacity_found = await _backpressure_wait()
        queued_seconds = time.time() - bp_start
        if not capacity_found:
            return web.Response(
                body=json.dumps({
                    "error": "all_accounts_saturated",
                    "message": "All accounts above utilization threshold. Request queued but timed out.",
                    "queued_seconds": round(queued_seconds, 1),
                    "queue_timeout": BACKPRESSURE_QUEUE_TIMEOUT,
                }).encode(),
                status=503,
                content_type="application/json",
                headers={
                    "X-LB-Queued": f"{queued_seconds:.1f}",
                    "Retry-After": "30",
                },
            )
    # ------------------------------------------------------------------

    consumer = request.headers.get("X-LB-Consumer", "").lower().strip()
    name, api_key = pick_account(consumer=consumer)
    STATS[name]["requests"] += 1
    STATS[name]["last_used"] = time.time()

    path = request.path
    if request.query_string:
        path += f"?{request.query_string}"

    headers = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("host", "content-length", "transfer-encoding", "x-lb-consumer"):
            continue
        headers[k] = v
    # Set auth using shared helper
    _set_auth_headers(headers, api_key)

    body = await request.read()

    # ------------------------------------------------------------------
    # PII Redaction -- tokenize PII in request body before forwarding
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

        # ------------------------------------------------------------------
        # Header audit -- log all response headers for first 3 requests
        # per account so we can see what Anthropic actually sends (Issue #5)
        # ------------------------------------------------------------------
        if _HEADER_AUDIT_COUNT.get(name, 0) < 3:
            _HEADER_AUDIT_COUNT[name] = _HEADER_AUDIT_COUNT.get(name, 0) + 1
            logger.debug(
                "[HEADER-AUDIT] account=%s status=%s headers=%s",
                name, resp.status, dict(resp.headers),
            )

        # ------------------------------------------------------------------
        # Cascade retry on 429 -- try ALL accounts before giving up (Issue #9)
        # ------------------------------------------------------------------
        if resp.status == 429:
            tried_accounts = {name}
            last_retry_after = 10

            while True:
                ra_hdr = resp.headers.get("retry-after")
                raw_cooldown = int(ra_hdr) if ra_hdr and ra_hdr.isdigit() else 60

                # For OAuth accounts: cap cooldown based on unified utilization.
                # If the account has low utilization, the 429 was a transient burst
                # limit -- don't lock it out for 48 minutes.
                is_oauth_acct = KEYS[name].startswith("sk-ant-oat")
                util_5h = STATS[name].get("unified_5h_utilization") or 0.0
                if is_oauth_acct and util_5h < 0.9:
                    # Low/moderate utilization: cap at 30s (transient burst)
                    cooldown = min(raw_cooldown, 30)
                elif is_oauth_acct and util_5h < 1.0:
                    # High but not maxed: cap at 120s
                    cooldown = min(raw_cooldown, 120)
                else:
                    # Non-OAuth or fully saturated: respect header, but cap at 5 min
                    cooldown = min(raw_cooldown, 300)

                last_retry_after = cooldown
                STATS[name]["errors"] += 1
                STATS[name]["rate_limit_hits"] += 1
                STATS[name]["rate_limited_until"] = time.time() + cooldown
                STATS[name]["rate_tokens_remaining"] = 0
                logger.warning(
                    "[RETRY] 429 account=%s cooldown=%ss remaining_accounts=%s",
                    name, cooldown, len(KEY_NAMES) - len(tried_accounts),
                )
                await resp.release()

                untried = [
                    n for n in KEY_NAMES
                    if n not in tried_accounts and STATS[n]["rate_limited_until"] <= time.time()
                ]
                if not untried:
                    # All accounts exhausted -- wait then try once more
                    logger.warning(
                        "[RETRY] all_accounts_saturated waiting=%ss then retrying", last_retry_after,
                    )
                    await asyncio.sleep(last_retry_after)
                    name, _ = pick_account()
                    STATS[name]["requests"] += 1
                    STATS[name]["last_used"] = time.time()
                    _set_auth_headers(headers, KEYS[name])
                    resp = await session.request(request.method, url, headers=headers, data=body)
                    if resp.status == 429:
                        await resp.release()
                        await session.close()
                        return web.Response(
                            body=json.dumps({
                                "error": "all_accounts_saturated",
                                "retry_after": last_retry_after,
                            }).encode(),
                            status=429,
                            content_type="application/json",
                            headers={
                                "X-LB-Account": name,
                                "retry-after": str(last_retry_after),
                            },
                        )
                    break

                name = untried[0]
                tried_accounts.add(name)
                STATS[name]["requests"] += 1
                STATS[name]["last_used"] = time.time()
                _set_auth_headers(headers, KEYS[name])
                logger.debug(
                    "[RETRY] attempt=%s account=%s remaining_accounts=%s",
                    len(tried_accounts), name, len(KEY_NAMES) - len(tried_accounts),
                )
                resp = await session.request(request.method, url, headers=headers, data=body)
                if resp.status != 429:
                    break

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
        if queued_seconds > 0:
            resp_headers["X-LB-Queued"] = f"{queued_seconds:.1f}"

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
                vault.audit_log(name, path)
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
            vault.audit_log(name, path)
            await resp.release()
            await session.close()
            return web.Response(
                body=response_body, status=resp.status, headers=resp_headers
            )
    except Exception as e:
        STATS[name]["errors"] += 1
        vault.flush_to_global_stats()
        vault.audit_log(name, path)
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
                "cache_ratio_pct": round(
                    (s["tokens_cache_read"] /
                     max(1, s["tokens_in"] + s["tokens_out"] + s["tokens_cache_read"] + s["tokens_cache_write"])
                    ) * 100, 1
                ),
            },
            "rate_limits": {
                # Standard API key fields
                "requests_remaining": s["rate_requests_remaining"],
                "requests_limit": s["rate_requests_limit"],
                "tokens_remaining": s["rate_tokens_remaining"],
                "tokens_limit": s["rate_tokens_limit"],
                "tokens_reset": s["rate_tokens_reset"],
                "rate_limited": rate_limited,
                "rate_limited_for": f"{s['rate_limited_until'] - now:.0f}s" if rate_limited else None,
                "total_429s": s["rate_limit_hits"],
                # Sliding window estimates
                "window_rpm": _window_tracker.get_rpm(name),
                "window_tpm": _window_tracker.get_tpm(name),
                "configured_limits": ACCOUNT_LIMITS.get(name, {}),
                # Unified fields — OAuth/Max accounts
                "unified_status": s["unified_status"],
                "unified_5h_utilization": s["unified_5h_utilization"],
                "unified_7d_utilization": s["unified_7d_utilization"],
                "unified_capacity": round(
                    1.0 - max(s["unified_5h_utilization"] or 0.0, s["unified_7d_utilization"] or 0.0), 3
                ) if s["unified_status"] is not None else None,
                "unified_representative_claim": s["unified_representative_claim"],
                # Routing scores — Issues #12-14
                "capacity_score": s["capacity_score"],
                "score_breakdown": s["score_breakdown"],
                "routing_weights": {"w_5h": ROUTING_W_5H, "w_7d": ROUTING_W_7D},
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
        "backpressure": {
            "enabled": BACKPRESSURE_THRESHOLD <= 1.0,
            "threshold": BACKPRESSURE_THRESHOLD if BACKPRESSURE_THRESHOLD <= 1.0 else None,
            "queue_timeout_seconds": BACKPRESSURE_QUEUE_TIMEOUT,
            "current_queue_depth": _backpressure_queue_depth,
            "total_queued": _backpressure_total_queued,
            "total_timeouts": _backpressure_total_timeouts,
        },
        "budget": {
            "threshold_5h": BUDGET_5H_THRESHOLD,
            "peak_hour_penalty": PEAK_HOUR_PENALTY,
            "peak_hours_pt": f"{PEAK_HOUR_START:02d}:00-{PEAK_HOUR_END:02d}:00",
            "is_peak_now": _is_peak_hour(),
        },
        "consumer_affinity": CONSUMER_AFFINITY or None,
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
            "global_mappings": _vault_store.count(),
            "vault_backend": "redis" if isinstance(_vault_store, RedisPIIVaultStore) else "file",
            "known_secrets": _KNOWN_SECRETS.count,
            "secrets_source": SECRETS_SOURCE or "off",
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


async def test_pii_handler(request: web.Request) -> web.Response:
    """
    Test endpoint: POST /test-pii with {"text": "..."} or GET /test-pii?text=...
    Shows exactly what the proxy does — original, tokenized (Anthropic sees), and restored (client sees).
    """
    if request.method == "POST":
        try:
            data = await request.json()
            text = data.get("text", "")
        except Exception:
            text = (await request.text()).strip()
    else:
        text = request.query.get("text", "")

    if not text:
        return web.json_response({
            "error": "Provide text via POST body {\"text\": \"...\"} or GET ?text=...",
            "example": "curl -X POST http://localhost:8891/test-pii -d '{\"text\": \"Email john@acme.com at 555-123-4567\"}'",
        }, status=400)

    vault = PIIVault()
    patterns = _get_all_patterns()
    tokenized = _redact_text(text, vault, patterns)
    restored = vault.detokenize(tokenized)

    detections = []
    for original, synthetic in vault._fwd.items():
        pii_type = vault._types.get(synthetic, "UNKNOWN")
        detections.append({
            "original": original,
            "token": synthetic,
            "type": pii_type,
        })

    return web.json_response({
        "pii_mode": PII_MODE,
        "original": text,
        "anthropic_sees": tokenized,
        "client_sees": restored,
        "pii_detected": len(detections),
        "detections": detections,
        "round_trip_ok": restored == text,
    }, dumps=lambda obj: json.dumps(obj, indent=2))


async def _on_startup(app):
    """Start background tasks on app startup."""
    asyncio.create_task(_oauth_usage_poll_loop())


app = web.Application()
app.on_startup.append(_on_startup)
app.router.add_get("/status", status_handler)
app.router.add_get("/health", health_handler)
app.router.add_route("*", "/test-pii", test_pii_handler)
app.router.add_route("*", "/{path:.*}", proxy_handler)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logger.info(
        "[CONFIG] PII_MODE=%s STRATEGY=%s accounts=%s upstream=%s",
        PII_MODE, STRATEGY, len(KEYS), UPSTREAM,
    )
    print(f"anthropic-lb starting on :{PORT}")
    print(f"  accounts: {', '.join(KEY_NAMES)} ({len(KEYS)} keys)")
    print(f"  upstream: {UPSTREAM}")
    print(f"  strategy: {STRATEGY}")
    _vault_backend = "redis" if isinstance(_vault_store, RedisPIIVaultStore) else "file"
    print(f"  pii vault: {_vault_backend} ({_vault_store.count()} persisted mappings)")
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
    # Load protected files first (feeds into known secrets store)
    if PROTECT_FILES:
        file_secrets = load_protected_files()
        if file_secrets:
            # Merge into known secrets store (before password manager, so PM overrides)
            _KNOWN_SECRETS._secrets.extend(file_secrets)
            print(f"  protected files: {len(file_secrets)} values from {PROTECT_FILES}")

    # Load known secrets from password manager (merges with file-protected values)
    if SECRETS_SOURCE and SECRETS_SOURCE != "off":
        existing = list(_KNOWN_SECRETS._secrets)  # preserve file-protected values
        _KNOWN_SECRETS.load()
        _KNOWN_SECRETS._secrets.extend(existing)  # re-add file values
        # Rebuild automaton with merged values
        if hasattr(_KNOWN_SECRETS, '_build_automaton'):
            _KNOWN_SECRETS._build_automaton()
        print(f"  secrets: {SECRETS_SOURCE} + files ({_KNOWN_SECRETS.count} total values)")
    elif PROTECT_FILES and _KNOWN_SECRETS._secrets:
        # No password manager but we have file-protected values -- deduplicate
        seen = set()
        unique = []
        for val, label, stype in _KNOWN_SECRETS._secrets:
            if val not in seen and len(val) >= KnownSecretsStore.MIN_SECRET_LEN:
                seen.add(val)
                unique.append((val, label, stype))
        _KNOWN_SECRETS._secrets = unique
        _KNOWN_SECRETS._loaded = True
        _KNOWN_SECRETS._last_refresh = time.time()
        print(f"  secrets: files-only ({_KNOWN_SECRETS.count} known values)")

    # Pre-seed global vault store with all known secrets mappings
    if _KNOWN_SECRETS.count > 0:
        generator = SyntheticPIIGenerator()
        pre_seed_batch = {}
        for value, label, stype in _KNOWN_SECRETS._secrets:
            synthetic = generator.generate(stype, value)
            if not _vault_store.contains(synthetic):
                pre_seed_batch[synthetic] = (value, stype)
        if pre_seed_batch:
            _vault_store.bulk_set(pre_seed_batch)
        print(f"  vault pre-seeded: {len(pre_seed_batch)} new mappings ({_vault_store.count()} total)")

    web.run_app(app, port=PORT, print=None)
