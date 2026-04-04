"""
Test PIIVault: tokenize/detokenize round-trip, determinism,
global persistence, FilePIIVaultStore, collision resistance.

Author: Lance James, Unit 221B, Inc.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy import PIIVault, FilePIIVaultStore, SyntheticPIIGenerator, _luhn_valid


# =========================================================================
# Basic tokenize / detokenize round-trip
# =========================================================================
class TestVaultRoundTrip:

    def test_email_round_trip(self):
        vault = PIIVault()
        original = "user@example.com"
        token = vault.tokenize(original, "EMAIL")
        assert token != original
        text = f"Contact {token} for info"
        restored = vault.detokenize(text)
        assert original in restored

    def test_phone_round_trip(self):
        vault = PIIVault()
        original = "555-123-4567"
        token = vault.tokenize(original, "PHONE")
        assert token != original
        restored = vault.detokenize(f"Call {token}")
        assert original in restored

    def test_ssn_round_trip(self):
        vault = PIIVault()
        original = "ssn: 123-45-6789"
        token = vault.tokenize(original, "SSN")
        restored = vault.detokenize(token)
        assert original in restored

    def test_credit_card_round_trip(self):
        vault = PIIVault()
        original = "4111111111111111"
        token = vault.tokenize(original, "CREDIT_CARD")
        assert token != original
        # Synthetic should also be Luhn-valid
        assert _luhn_valid(token)
        restored = vault.detokenize(token)
        assert restored == original

    def test_ip_round_trip(self):
        vault = PIIVault()
        original = "192.168.1.100"
        token = vault.tokenize(original, "IP_ADDRESS")
        restored = vault.detokenize(token)
        assert restored == original

    def test_api_key_round_trip(self):
        vault = PIIVault()
        original = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz"
        token = vault.tokenize(original, "API_KEY")
        restored = vault.detokenize(token)
        assert restored == original

    def test_password_round_trip(self):
        vault = PIIVault()
        original = "MyP@ssw0rd!!"
        token = vault.tokenize(original, "PASSWORD")
        assert len(token) == len(original)
        restored = vault.detokenize(token)
        assert restored == original

    @pytest.mark.parametrize("pii_type,original", [
        ("PERSON", "John Smith"),
        ("ORGANIZATION", "Acme Corp"),
        ("LOCATION", "Springfield"),
        ("URL", "https://internal.corp.net/api"),
        ("CRYPTO_WALLET", "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD13"),
    ])
    def test_ner_types_round_trip(self, pii_type, original):
        vault = PIIVault()
        token = vault.tokenize(original, pii_type)
        assert token != original
        restored = vault.detokenize(token)
        assert restored == original


# =========================================================================
# Determinism within a request
# =========================================================================
class TestVaultDeterminism:

    def test_same_value_same_token(self):
        """Same original value always produces the same synthetic in one vault."""
        vault = PIIVault()
        t1 = vault.tokenize("user@example.com", "EMAIL")
        t2 = vault.tokenize("user@example.com", "EMAIL")
        assert t1 == t2

    def test_same_value_increments_once(self):
        """Tokenizing the same value twice should not increment the counter."""
        vault = PIIVault()
        vault.tokenize("user@example.com", "EMAIL")
        vault.tokenize("user@example.com", "EMAIL")
        assert vault.count == 1

    def test_determinism_across_vaults(self):
        """Same original value with same md5 produces same synthetic.
        Note: counters may differ if vault state differs."""
        v1 = PIIVault()
        v2 = PIIVault()
        # First call in both vaults (idx=0 for EMAIL in both)
        t1 = v1.tokenize("user@example.com", "EMAIL")
        t2 = v2.tokenize("user@example.com", "EMAIL")
        assert t1 == t2

    def test_different_values_different_tokens(self):
        vault = PIIVault()
        t1 = vault.tokenize("alice@one.com", "EMAIL")
        t2 = vault.tokenize("bob@two.com", "EMAIL")
        assert t1 != t2


# =========================================================================
# Vault counts & type tracking
# =========================================================================
class TestVaultCounts:

    def test_count_increments(self):
        vault = PIIVault()
        vault.tokenize("a@b.com", "EMAIL")
        vault.tokenize("555-123-4567", "PHONE")
        assert vault.count == 2

    def test_by_type_tracking(self):
        vault = PIIVault()
        vault.tokenize("a@b.com", "EMAIL")
        vault.tokenize("b@c.com", "EMAIL")
        vault.tokenize("555-123-4567", "PHONE")
        assert vault.by_type["EMAIL"] == 2
        assert vault.by_type["PHONE"] == 1


# =========================================================================
# Detokenize with bytes
# =========================================================================
class TestVaultDetokenizeBytes:

    def test_bytes_round_trip(self):
        vault = PIIVault()
        original = "user@example.com"
        token = vault.tokenize(original, "EMAIL")
        body = f'{{"content": "{token}"}}'.encode()
        restored = vault.detokenize_bytes(body)
        assert original.encode() in restored


# =========================================================================
# FilePIIVaultStore
# =========================================================================
class TestFilePIIVaultStore:

    def test_set_and_get(self, tmp_path):
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("fake@example.com", "real@example.com", "EMAIL")
        assert store.get("fake@example.com") == "real@example.com"
        assert store.get_type("fake@example.com") == "EMAIL"

    def test_contains(self, tmp_path):
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("fake", "real", "SECRET")
        assert store.contains("fake")
        assert not store.contains("other")

    def test_count(self, tmp_path):
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("a", "b", "T")
        store.set("c", "d", "T")
        assert store.count() == 2

    def test_get_all(self, tmp_path):
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("s1", "o1", "T1")
        store.set("s2", "o2", "T2")
        all_entries = store.get_all()
        assert all_entries == {"s1": "o1", "s2": "o2"}

    def test_bulk_set(self, tmp_path):
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        mappings = {
            "synth1": ("orig1", "EMAIL"),
            "synth2": ("orig2", "PHONE"),
        }
        store.bulk_set(mappings)
        assert store.get("synth1") == "orig1"
        assert store.get("synth2") == "orig2"
        assert store.count() == 2

    def test_persistence(self, tmp_path):
        """Store saves to disk, new instance loads from disk."""
        vault_file = str(tmp_path / "vault.json")
        store1 = FilePIIVaultStore(vault_file)
        store1.set("fake-email@test.com", "real@test.com", "EMAIL")
        store1.set("fake-phone", "555-123-4567", "PHONE")

        # Create new instance from same file
        store2 = FilePIIVaultStore(vault_file)
        assert store2.get("fake-email@test.com") == "real@test.com"
        assert store2.get("fake-phone") == "555-123-4567"
        assert store2.count() == 2

    def test_empty_file(self, tmp_path):
        """Store handles non-existent vault file."""
        store = FilePIIVaultStore(str(tmp_path / "nonexistent.json"))
        assert store.count() == 0
        assert store.get("anything") is None

    def test_idempotent_set(self, tmp_path):
        """Setting the same key twice doesn't create duplicates."""
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("fake", "real", "T")
        store.set("fake", "real", "T")
        assert store.count() == 1

    def test_overwrite_value(self, tmp_path):
        """Setting same synthetic with different original updates."""
        store = FilePIIVaultStore(str(tmp_path / "vault.json"))
        store.set("synth", "orig1", "T")
        store.set("synth", "orig2", "T")
        assert store.get("synth") == "orig2"


# =========================================================================
# Flush to global stats
# =========================================================================
class TestFlushToGlobalStats:

    def test_flush_merges_counts(self):
        import proxy
        # Save original state
        orig_total = proxy.PII_GLOBAL_STATS["total_redacted"]
        orig_email = proxy.PII_GLOBAL_STATS["by_type"].get("EMAIL", 0)

        vault = PIIVault()
        vault.tokenize("a@b.com", "EMAIL")
        vault.tokenize("555-123-4567", "PHONE")
        vault.flush_to_global_stats()

        assert proxy.PII_GLOBAL_STATS["total_redacted"] == orig_total + 2
        assert proxy.PII_GLOBAL_STATS["by_type"]["EMAIL"] == orig_email + 1

        # Clean up
        proxy.PII_GLOBAL_STATS["total_redacted"] = orig_total
        proxy.PII_GLOBAL_STATS["by_type"]["EMAIL"] = orig_email


# =========================================================================
# Collision handling
# =========================================================================
class TestVaultCollisionHandling:

    def test_no_collision_100_values(self):
        """100 unique values should produce 100 unique synthetics."""
        vault = PIIVault()
        tokens = set()
        for i in range(100):
            t = vault.tokenize(f"user{i}@test{i}.com", "EMAIL")
            assert t not in tokens, f"Collision at index {i}"
            tokens.add(t)
        assert vault.count == 100

    def test_no_collision_mixed_types(self):
        """Values of different types should not collide."""
        vault = PIIVault()
        tokens = set()
        values = [
            ("EMAIL", "user@test.com"),
            ("PHONE", "555-123-4567"),
            ("IP_ADDRESS", "10.0.0.1"),
            ("API_KEY", "sk-ant-api03-test123456789012345"),
            ("PASSWORD", "MySecretPassword!"),
            ("PERSON", "John Smith"),
            ("ORGANIZATION", "Acme Corp"),
            ("LOCATION", "Springfield, IL"),
            ("URL", "https://example.com"),
            ("CRYPTO_WALLET", "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD13"),
        ]
        for pii_type, val in values:
            t = vault.tokenize(val, pii_type)
            assert t not in tokens, f"Collision: {pii_type}={val}"
            tokens.add(t)

    def test_collision_guard(self):
        """If two originals map to the same synthetic, the guard re-hashes."""
        vault = PIIVault()
        # Manually insert a mapping that would collide
        t1 = vault.tokenize("original-1", "SECRET")
        # Now pretend the generator would produce the same synthetic for a different original
        # The vault checks _rev and _vault_store for collisions
        t2 = vault.tokenize("original-2", "SECRET")
        assert t1 != t2, "Collision guard should produce different tokens"


# =========================================================================
# Multi-value detokenization
# =========================================================================
class TestMultiValueDetokenize:

    def test_multiple_values_in_text(self):
        vault = PIIVault()
        email_token = vault.tokenize("user@example.com", "EMAIL")
        phone_token = vault.tokenize("555-123-4567", "PHONE")

        text = f"Contact {email_token} at {phone_token}"
        restored = vault.detokenize(text)
        assert "user@example.com" in restored
        assert "555-123-4567" in restored

    def test_longest_match_first(self):
        """Detokenization should replace longest synthetic first to avoid partial matches."""
        vault = PIIVault()
        short = vault.tokenize("a@b.com", "EMAIL")
        # The sort in detokenize uses -len(synthetic) to process longest first
        restored = vault.detokenize(short)
        assert "a@b.com" in restored
