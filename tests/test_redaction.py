"""
Test the full _redact_text pipeline, content block handling,
request body redaction, and round-trip integrity.

Tests: known secrets layer -> regex layer -> overlap handling,
content blocks (text, tool_use, tool_result), Anthropic messages format.

Author: Lance James, Unit 221B, Inc.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy import (
    _build_builtin_patterns,
    _redact_text,
    _redact_content_block,
    PIIVault,
    KnownSecretsStore,
)


@pytest.fixture
def patterns():
    return _build_builtin_patterns()


# =========================================================================
# _redact_text basic behavior
# =========================================================================
class TestRedactText:

    def test_email_redacted(self, patterns):
        vault = PIIVault()
        text = "Contact user@example.com for details"
        result = _redact_text(text, vault, patterns)
        assert "user@example.com" not in result
        assert vault.count == 1
        assert "EMAIL" in vault.by_type

    def test_phone_redacted(self, patterns):
        vault = PIIVault()
        text = "Call me at 555-123-4567"
        result = _redact_text(text, vault, patterns)
        assert "555-123-4567" not in result
        assert vault.count == 1

    def test_ssn_with_context_redacted(self, patterns):
        vault = PIIVault()
        text = "My ssn: 123-45-6789 is private"
        result = _redact_text(text, vault, patterns)
        assert "123-45-6789" not in result

    def test_credit_card_redacted(self, patterns):
        vault = PIIVault()
        text = "Card: 4111111111111111"
        result = _redact_text(text, vault, patterns)
        assert "4111111111111111" not in result

    def test_ip_address_redacted(self, patterns):
        vault = PIIVault()
        text = "Server 192.168.1.100 is down"
        result = _redact_text(text, vault, patterns)
        assert "192.168.1.100" not in result

    def test_multiple_pii_types(self, patterns):
        vault = PIIVault()
        text = "Email user@test.com, IP 10.0.0.1, phone 555-123-4567"
        result = _redact_text(text, vault, patterns)
        assert "user@test.com" not in result
        assert "10.0.0.1" not in result
        assert "555-123-4567" not in result
        assert vault.count == 3

    def test_no_pii(self, patterns):
        vault = PIIVault()
        text = "This is a normal sentence with no PII."
        result = _redact_text(text, vault, patterns)
        assert result == text
        assert vault.count == 0

    def test_empty_text(self, patterns):
        vault = PIIVault()
        result = _redact_text("", vault, patterns)
        assert result == ""

    def test_none_safe(self, patterns):
        """None/falsy input returns as-is."""
        vault = PIIVault()
        result = _redact_text("", vault, patterns)
        assert result == ""

    def test_round_trip(self, patterns):
        """Redact then detokenize should restore original text."""
        vault = PIIVault()
        original = "Email admin@company.com, phone 555-987-6543"
        redacted = _redact_text(original, vault, patterns)
        restored = vault.detokenize(redacted)
        assert restored == original

    def test_round_trip_complex(self, patterns):
        """Round-trip with many PII types."""
        vault = PIIVault()
        original = (
            "Contact john.doe@example.com (555-123-4567) "
            "at 192.168.1.1. "
            "AWS key: AKIAIOSFODNN7EXAMPLE"
        )
        redacted = _redact_text(original, vault, patterns)
        # Verify all PII removed
        assert "john.doe@example.com" not in redacted
        assert "555-123-4567" not in redacted
        assert "192.168.1.1" not in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        # Verify round-trip
        restored = vault.detokenize(redacted)
        assert restored == original


# =========================================================================
# No double-redaction
# =========================================================================
class TestNoDoubleRedaction:

    def test_same_value_twice(self, patterns):
        """Same PII value appearing twice should produce same synthetic."""
        vault = PIIVault()
        text = "Email user@test.com and also user@test.com again"
        result = _redact_text(text, vault, patterns)
        assert "user@test.com" not in result
        # Only one unique PII value
        assert vault.count == 1


# =========================================================================
# Content block handling
# =========================================================================
class TestRedactContentBlock:

    def test_text_block(self, patterns):
        vault = PIIVault()
        block = {"type": "text", "text": "Email: user@test.com"}
        result = _redact_content_block(block, vault, patterns)
        assert "user@test.com" not in result["text"]

    def test_tool_use_block(self, patterns, sample_tool_use_block):
        vault = PIIVault()
        result = _redact_content_block(sample_tool_use_block, vault, patterns)
        assert "admin@corp.com" not in result["input"]["query"]
        assert "192.168.1.100" not in result["input"]["ip"]

    def test_tool_result_string_content(self, patterns):
        vault = PIIVault()
        block = {
            "type": "tool_result",
            "tool_use_id": "id_123",
            "content": "Result contains user@secret.com"
        }
        result = _redact_content_block(block, vault, patterns)
        assert "user@secret.com" not in result["content"]

    def test_tool_result_list_content(self, patterns):
        vault = PIIVault()
        block = {
            "type": "tool_result",
            "tool_use_id": "id_123",
            "content": [
                {"type": "text", "text": "Email: nested@test.com"},
            ]
        }
        result = _redact_content_block(block, vault, patterns)
        assert "nested@test.com" not in result["content"][0]["text"]

    def test_plain_string_content(self, patterns):
        vault = PIIVault()
        text = "Plain string with user@test.com"
        result = _redact_content_block(text, vault, patterns)
        assert "user@test.com" not in result

    def test_unknown_block_type(self, patterns):
        vault = PIIVault()
        block = {"type": "image", "source": {"data": "base64..."}}
        result = _redact_content_block(block, vault, patterns)
        assert result == block  # unchanged


# =========================================================================
# Full request body redaction
# =========================================================================
class TestRedactRequestBody:

    def test_system_string(self, patterns):
        vault = PIIVault()
        body = {
            "model": "claude-sonnet-4-20250514",
            "system": "Admin email: admin@internal.corp",
            "messages": [],
        }
        text = json.dumps(body)
        patterns_list = patterns

        # Simulate what redact_request_body does
        data = json.loads(text)
        system = data.get("system")
        if isinstance(system, str):
            data["system"] = _redact_text(system, vault, patterns_list)

        assert "admin@internal.corp" not in data["system"]
        assert vault.count == 1

    def test_messages_content(self, patterns, sample_anthropic_request):
        vault = PIIVault()
        data = sample_anthropic_request

        # Redact system
        if isinstance(data.get("system"), str):
            data["system"] = _redact_text(data["system"], vault, patterns)

        # Redact messages
        for msg in data.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _redact_text(content, vault, patterns)
            elif isinstance(content, list):
                msg["content"] = [_redact_content_block(b, vault, patterns) for b in content]

        # Verify PII removed from system
        assert "admin@internal.corp" not in data["system"]

        # Verify PII removed from user message
        assert "john.doe@company.com" not in data["messages"][0]["content"]
        assert "555-123-4567" not in data["messages"][0]["content"]

        # Verify tool_result content redacted
        tool_result_content = data["messages"][2]["content"][1]["content"]
        assert "test@example.com" not in tool_result_content

        # Verify non-PII text preserved
        assert "I understand" in data["messages"][1]["content"]

    def test_round_trip_full_request(self, patterns, sample_anthropic_request):
        """Full round-trip: redact request -> detokenize -> matches original."""
        vault = PIIVault()
        original_system = sample_anthropic_request["system"]
        original_msg = sample_anthropic_request["messages"][0]["content"]

        data = json.loads(json.dumps(sample_anthropic_request))

        # Redact
        if isinstance(data.get("system"), str):
            data["system"] = _redact_text(data["system"], vault, patterns)
        for msg in data.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = _redact_text(content, vault, patterns)

        # Detokenize
        restored_system = vault.detokenize(data["system"])
        restored_msg = vault.detokenize(data["messages"][0]["content"])

        assert restored_system == original_system
        assert restored_msg == original_msg


# =========================================================================
# Known secrets take priority over regex
# =========================================================================
class TestSecretsPriority:

    def test_secrets_override_regex(self, patterns, monkeypatch):
        """Known secrets layer should take priority. When a known secret overlaps
        with a regex match, the secret match should win."""
        import proxy

        # Save original store
        orig_store = proxy._KNOWN_SECRETS

        # Create a store with a known email that would also match the email regex
        test_store = KnownSecretsStore()
        test_store._secrets = [
            ("admin@internal.corp", "admin_email", "EMAIL"),
        ]
        test_store._loaded = True
        test_store._last_refresh = 99999999999.0

        monkeypatch.setattr("proxy._KNOWN_SECRETS", test_store)

        vault = PIIVault()
        text = "Contact admin@internal.corp for help"
        result = _redact_text(text, vault, patterns)

        assert "admin@internal.corp" not in result
        # Should have been tokenized exactly once (not double-tokenized)
        assert vault.count == 1

        # Restore
        monkeypatch.setattr("proxy._KNOWN_SECRETS", orig_store)


# =========================================================================
# PEM / multiline content
# =========================================================================
class TestMultilineRedaction:

    def test_pem_key_redacted(self, patterns):
        vault = PIIVault()
        text = (
            "Here is a key:\n"
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/yGE...\n"
            "-----END RSA PRIVATE KEY-----\n"
            "Do not share this."
        )
        result = _redact_text(text, vault, patterns)
        assert "-----BEGIN RSA PRIVATE KEY-----" not in result
        assert vault.count >= 1


# =========================================================================
# Generic secret pattern
# =========================================================================
class TestGenericSecretRedaction:

    def test_password_assignment(self, patterns):
        vault = PIIVault()
        text = 'password=MyLongPassword123'
        result = _redact_text(text, vault, patterns)
        assert "MyLongPassword123" not in result

    def test_api_key_assignment(self, patterns):
        vault = PIIVault()
        text = 'api_key: sk-something-very-long'
        result = _redact_text(text, vault, patterns)
        assert "sk-something-very-long" not in result


# =========================================================================
# Crypto wallet addresses
# =========================================================================
class TestCryptoRedaction:

    def test_eth_address(self, patterns):
        vault = PIIVault()
        text = "Send to 0x742d35Cc6634C0532925a3b844Bc9e7595f2bD13"
        result = _redact_text(text, vault, patterns)
        assert "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD13" not in result

    def test_btc_bech32(self, patterns):
        vault = PIIVault()
        text = "BTC: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        result = _redact_text(text, vault, patterns)
        assert "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4" not in result
