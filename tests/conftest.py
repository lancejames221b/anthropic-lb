"""
Shared fixtures for the anthropic-lb test suite.

Provides sample PII texts, mock accounts, pattern references, and
temporary file builders for .env/.json/.yaml/.ini extraction tests.

Author: Lance James, Unit 221B, Inc.
"""

import json
import os
import sys

import pytest

# Make proxy.py importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proxy import (
    _build_builtin_patterns,
    _luhn_valid,
    SyntheticPIIGenerator,
    PIIVault,
    KnownSecretsStore,
    FilePIIVaultStore,
    _classify_key,
    _extract_env,
    _extract_json,
    _extract_yaml,
    _extract_ini,
    _extract_lines,
    _redact_text,
    _redact_content_block,
    _set_auth_headers,
)


# ---------------------------------------------------------------------------
# Pattern references
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def builtin_patterns():
    """All 27 built-in regex patterns from _build_builtin_patterns()."""
    return _build_builtin_patterns()


@pytest.fixture(scope="session")
def pattern_dict(builtin_patterns):
    """Dict mapping pattern name -> (name, regex, type_label, validator)."""
    return {p[0]: p for p in builtin_patterns}


# ---------------------------------------------------------------------------
# Generator & vault instances
# ---------------------------------------------------------------------------

@pytest.fixture
def generator():
    """Fresh SyntheticPIIGenerator per test."""
    return SyntheticPIIGenerator()


@pytest.fixture
def vault():
    """Fresh PIIVault per test (uses module-level _vault_store)."""
    return PIIVault()


# ---------------------------------------------------------------------------
# Sample PII texts -- labeled for pattern testing
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_texts():
    """Dict of PII type -> list of sample values that SHOULD be detected."""
    return {
        "EMAIL": [
            "user@example.com",
            "first.last+tag@sub.domain.co",
            "test_email@company.org",
        ],
        "SSN": [
            "ssn: 123-45-6789",
            "Social Security Number: 456-78-9012",
            "SSN# 321-54-9876",
        ],
        "CREDIT_CARD": [
            "4111111111111111",  # Visa
            "5500000000000004",  # MC
            "378282246310005",   # Amex
        ],
        "PHONE": [
            "+1-555-123-4567",
            "(555) 987-6543",
            "555-123-4567",
        ],
        "AWS_KEY": [
            "AKIAIOSFODNN7EXAMPLE",
        ],
        "GITHUB_TOKEN": [
            "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh1234",
        ],
        "ANTHROPIC_KEY": [
            "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
            "sk-ant-oat01-abcdefghijklmnopqrstuvwxyz",
        ],
        "IPV4": [
            "192.168.1.1",
            "10.0.0.1",
            "255.255.255.0",
        ],
    }


@pytest.fixture(scope="session")
def sample_non_pii():
    """Values that SHOULD NOT match any PII pattern."""
    return [
        "Hello world",
        "12345",
        "test",
        "The quick brown fox",
        "version 2.3.4",
        "192.168",
        "not-an-email",
        "sk-ant-wrong-format",
    ]


# ---------------------------------------------------------------------------
# Mock account fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_accounts():
    """Simulated keys.json entries for routing tests."""
    return {
        "api-key-account": "sk-ant-api03-aaabbbcccdddeeefffggghhhiii",
        "oauth-account": "sk-ant-oat01-aaabbbcccdddeeefffggghhhiii",
        "second-api": "sk-ant-api03-111222333444555666777888999",
    }


@pytest.fixture
def mock_stats(mock_accounts):
    """Blank stats structure matching proxy.py STATS layout."""
    stats = {}
    for name in mock_accounts:
        stats[name] = {
            "requests": 0,
            "errors": 0,
            "last_used": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "tokens_cache_read": 0,
            "tokens_cache_write": 0,
            "rate_requests_limit": None,
            "rate_requests_remaining": None,
            "rate_requests_reset": None,
            "rate_tokens_limit": None,
            "rate_tokens_remaining": None,
            "rate_tokens_reset": None,
            "unified_status": None,
            "unified_5h_utilization": None,
            "unified_5h_reset": None,
            "unified_7d_utilization": None,
            "unified_7d_reset": None,
            "unified_fallback": None,
            "unified_representative_claim": None,
            "capacity_score": None,
            "score_breakdown": None,
            "rate_limited_until": 0,
            "rate_limit_hits": 0,
        }
    return stats


# ---------------------------------------------------------------------------
# Temporary file builders for file extraction tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_env_file(tmp_path):
    """Create a sample .env file and return its path."""
    content = """# Database config
DB_HOST=db.internal.corp
DB_PASSWORD=SuperSecret123!
API_KEY=sk-proj-abc123xyz789test
EMPTY_VAL=
SHORT=yes
# A comment line
SMTP_HOST=mail.company.com
"""
    path = tmp_path / ".env"
    path.write_text(content)
    return str(path)


@pytest.fixture
def sample_json_file(tmp_path):
    """Create a sample secrets JSON file and return its path."""
    data = {
        "database": {
            "host": "db.prod.internal",
            "password": "MyDBPassword!99",
            "port": 5432,
        },
        "api_key": "sk-test-0123456789abcdef",
        "short": "abc",
        "nested": {
            "deep": {
                "token": "super-long-nested-secret-value"
            }
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_ini_file(tmp_path):
    """Create a sample .ini file and return its path."""
    content = """[database]
host = db.prod.internal
password = IniSecretPass!42
port = 5432

[api]
token = long-api-token-value-here
"""
    path = tmp_path / "config.ini"
    path.write_text(content)
    return str(path)


@pytest.fixture
def sample_lines_file(tmp_path):
    """Create a plain text config file and return its path."""
    content = """# SSH authorized key
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC3 user@host
TOKEN=some-secret-token-value
; ini-style comment
bare-line-that-is-a-secret-value
short
"""
    path = tmp_path / "tokens.txt"
    path.write_text(content)
    return str(path)


@pytest.fixture
def sample_yaml_file(tmp_path):
    """Create a sample YAML file and return its path."""
    content = """database:
  host: db.yaml.internal
  password: YamlPassword!77
api:
  key: yaml-api-key-long-value
  short: abc
"""
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return str(path)


# ---------------------------------------------------------------------------
# Known secrets fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def secrets_list_file(tmp_path):
    """JSON file in list format for KnownSecretsStore._load_file()."""
    data = [
        {"label": "AWS Key", "value": "AKIAIOSFODNN7EXAMPLE", "type": "API_KEY"},
        {"label": "DB Password", "value": "SuperSecretDBPass!42", "type": "PASSWORD"},
        {"label": "Short Val", "value": "abc", "type": "SECRET"},
    ]
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def secrets_dict_file(tmp_path):
    """JSON file in dict format for KnownSecretsStore._load_file()."""
    data = {
        "aws_key": "AKIAIOSFODNN7EXAMPLE",
        "db_pass": "SuperSecretDBPass!42",
        "short": "abc",
    }
    path = tmp_path / "secrets_dict.json"
    path.write_text(json.dumps(data))
    return str(path)


# ---------------------------------------------------------------------------
# Anthropic API message fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_anthropic_request():
    """Sample Anthropic Messages API request body with embedded PII."""
    return {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "system": "You are a helpful assistant. Contact admin@internal.corp for help.",
        "messages": [
            {
                "role": "user",
                "content": "My email is john.doe@company.com and my phone is 555-123-4567."
            },
            {
                "role": "assistant",
                "content": "I understand you shared your contact information."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "My SSN is ssn: 123-45-6789"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_abc",
                        "content": "User email: test@example.com"
                    },
                ]
            },
        ],
    }


@pytest.fixture
def sample_tool_use_block():
    """Content block with tool_use containing PII in input."""
    return {
        "type": "tool_use",
        "id": "tool_123",
        "name": "lookup",
        "input": {
            "query": "Find user admin@corp.com",
            "ip": "192.168.1.100",
        },
    }
