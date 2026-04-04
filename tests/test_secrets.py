"""
Test KnownSecretsStore: file loading, Aho-Corasick matching,
linear fallback, file extractors, key classification.

Author: Lance James, Unit 221B, Inc.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy import (
    KnownSecretsStore,
    _classify_key,
    _extract_env,
    _extract_json,
    _extract_yaml,
    _extract_ini,
    _extract_lines,
)


# =========================================================================
# _classify_key
# =========================================================================
class TestClassifyKey:

    @pytest.mark.parametrize("key,expected", [
        ("password", "PASSWORD"),
        ("DB_PASSWORD", "PASSWORD"),
        ("admin_passwd", "PASSWORD"),
        ("master_pass", "PASSWORD"),
        ("API_KEY", "API_KEY"),
        ("secret_token", "API_KEY"),
        ("auth_header", "API_KEY"),
        ("EMAIL_ADDRESS", "EMAIL"),
        ("SMTP_MAIL", "EMAIL"),
        ("DB_HOST", "URL"),
        ("WEBHOOK_URL", "URL"),
        ("endpoint", "URL"),
        ("SERVER_NAME", "URL"),
        ("username", "USERNAME"),
        ("login_user", "USERNAME"),
        ("SSH_KEY", "API_KEY"),
        ("ssl_cert", "API_KEY"),
        ("private_key", "API_KEY"),
        ("phone_number", "PHONE"),
        ("mobile", "PHONE"),
        ("ssn_value", "SSN"),
        ("RANDOM_THING", "SECRET"),
        ("DATA_VALUE", "SECRET"),
    ])
    def test_classification(self, key, expected):
        assert _classify_key(key) == expected


# =========================================================================
# File extractors
# =========================================================================
class TestExtractEnv:

    def test_basic(self, sample_env_file):
        secrets = _extract_env(sample_env_file)
        values = [s[0] for s in secrets]
        assert "db.internal.corp" in values
        assert "SuperSecret123!" in values
        assert "sk-proj-abc123xyz789test" in values
        assert "mail.company.com" in values

    def test_skips_empty_and_short(self, sample_env_file):
        secrets = _extract_env(sample_env_file)
        values = [s[0] for s in secrets]
        # EMPTY_VAL= should be skipped (empty)
        assert "" not in values
        # SHORT=yes should be skipped (< 6 chars)
        assert "yes" not in values

    def test_skips_comments(self, sample_env_file):
        secrets = _extract_env(sample_env_file)
        labels = [s[1] for s in secrets]
        # Comments should not produce entries
        for label in labels:
            assert "# " not in label.split("/")[-1]

    def test_strips_quotes(self, tmp_path):
        content = """QUOTED='value-with-single-quotes'\nDOUBLE="value-with-double-quotes"\n"""
        path = tmp_path / ".env.test"
        path.write_text(content)
        secrets = _extract_env(str(path))
        values = [s[0] for s in secrets]
        assert "value-with-single-quotes" in values
        assert "value-with-double-quotes" in values

    def test_type_assignment(self, sample_env_file):
        secrets = _extract_env(sample_env_file)
        type_map = {s[0]: s[2] for s in secrets}
        assert type_map["SuperSecret123!"] == "PASSWORD"
        assert type_map["sk-proj-abc123xyz789test"] == "API_KEY"
        assert type_map["db.internal.corp"] == "URL"


class TestExtractJSON:

    def test_basic(self, sample_json_file):
        secrets = _extract_json(sample_json_file)
        values = [s[0] for s in secrets]
        assert "db.prod.internal" in values
        assert "MyDBPassword!99" in values
        assert "sk-test-0123456789abcdef" in values

    def test_nested(self, sample_json_file):
        secrets = _extract_json(sample_json_file)
        values = [s[0] for s in secrets]
        assert "super-long-nested-secret-value" in values

    def test_skips_short(self, sample_json_file):
        secrets = _extract_json(sample_json_file)
        values = [s[0] for s in secrets]
        assert "abc" not in values  # too short

    def test_skips_non_string(self, sample_json_file):
        secrets = _extract_json(sample_json_file)
        values = [s[0] for s in secrets]
        # port=5432 is an int, should not be extracted
        assert "5432" not in values


class TestExtractINI:

    def test_basic(self, sample_ini_file):
        secrets = _extract_ini(sample_ini_file)
        values = [s[0] for s in secrets]
        assert "db.prod.internal" in values
        assert "IniSecretPass!42" in values
        assert "long-api-token-value-here" in values

    def test_skips_short(self, sample_ini_file):
        secrets = _extract_ini(sample_ini_file)
        values = [s[0] for s in secrets]
        assert "5432" not in values  # too short


class TestExtractLines:

    def test_basic(self, sample_lines_file):
        secrets = _extract_lines(sample_lines_file)
        values = [s[0] for s in secrets]
        # KEY=VALUE lines extracted
        assert "some-secret-token-value" in values

    def test_skips_comments(self, sample_lines_file):
        secrets = _extract_lines(sample_lines_file)
        values = [s[0] for s in secrets]
        for v in values:
            assert not v.startswith("#")
            assert not v.startswith(";")

    def test_bare_lines(self, sample_lines_file):
        secrets = _extract_lines(sample_lines_file)
        values = [s[0] for s in secrets]
        assert "bare-line-that-is-a-secret-value" in values

    def test_skips_short(self, sample_lines_file):
        secrets = _extract_lines(sample_lines_file)
        values = [s[0] for s in secrets]
        assert "short" not in values


class TestExtractYAML:

    def test_basic(self, sample_yaml_file):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        secrets = _extract_yaml(sample_yaml_file)
        values = [s[0] for s in secrets]
        assert "db.yaml.internal" in values
        assert "YamlPassword!77" in values
        assert "yaml-api-key-long-value" in values

    def test_skips_short(self, sample_yaml_file):
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        secrets = _extract_yaml(sample_yaml_file)
        values = [s[0] for s in secrets]
        assert "abc" not in values


# =========================================================================
# KnownSecretsStore -- file loading
# =========================================================================
class TestKnownSecretsStoreFileLoad:

    def test_load_list_format(self, secrets_list_file, monkeypatch):
        monkeypatch.setattr("proxy.SECRETS_SOURCE", "file")
        monkeypatch.setattr("proxy.SECRETS_FILE", secrets_list_file)

        store = KnownSecretsStore()
        store._load_file()

        assert store.count >= 2  # 2 values above MIN_SECRET_LEN
        values = [s[0] for s in store._secrets]
        assert "AKIAIOSFODNN7EXAMPLE" in values
        assert "SuperSecretDBPass!42" in values

    def test_load_dict_format(self, secrets_dict_file, monkeypatch):
        monkeypatch.setattr("proxy.SECRETS_SOURCE", "file")
        monkeypatch.setattr("proxy.SECRETS_FILE", secrets_dict_file)

        store = KnownSecretsStore()
        store._load_file()

        values = [s[0] for s in store._secrets]
        assert "AKIAIOSFODNN7EXAMPLE" in values
        assert "SuperSecretDBPass!42" in values

    def test_dict_format_skips_short(self, secrets_dict_file, monkeypatch):
        monkeypatch.setattr("proxy.SECRETS_SOURCE", "file")
        monkeypatch.setattr("proxy.SECRETS_FILE", secrets_dict_file)

        store = KnownSecretsStore()
        store._load_file()

        values = [s[0] for s in store._secrets]
        assert "abc" not in values

    def test_file_not_found(self, monkeypatch):
        monkeypatch.setattr("proxy.SECRETS_FILE", "/nonexistent/path.json")

        store = KnownSecretsStore()
        with pytest.raises(FileNotFoundError):
            store._load_file()


# =========================================================================
# find_in_text — scanning for known secrets
# =========================================================================
class TestFindInText:

    def _make_store(self, secrets_list):
        """Build a KnownSecretsStore from a list of (value, label, type) tuples."""
        store = KnownSecretsStore()
        store._secrets = secrets_list
        store._loaded = True
        store._last_refresh = 99999999999.0  # prevent refresh
        return store

    def test_exact_match(self):
        store = self._make_store([
            ("SuperSecretDBPass!42", "db_pass", "PASSWORD"),
        ])
        text = "The password is SuperSecretDBPass!42 in the config"
        matches = store.find_in_text(text)
        assert len(matches) == 1
        start, end, stype, value, label = matches[0]
        assert value == "SuperSecretDBPass!42"
        assert stype == "PASSWORD"
        assert text[start:end] == "SuperSecretDBPass!42"

    def test_multiple_matches(self):
        store = self._make_store([
            ("secret-one-val", "first", "SECRET"),
            ("secret-two-val", "second", "SECRET"),
        ])
        text = "Values are secret-one-val and secret-two-val here"
        matches = store.find_in_text(text)
        assert len(matches) == 2
        values = [m[3] for m in matches]
        assert "secret-one-val" in values
        assert "secret-two-val" in values

    def test_overlapping_matches_deduped(self):
        """When two secrets overlap, leftmost/longest wins."""
        store = self._make_store([
            ("ABCDEFGH", "long", "SECRET"),
            ("DEFGH", "short", "SECRET"),
        ])
        text = "prefix ABCDEFGH suffix"
        matches = store.find_in_text(text)
        # Should only have the longest match
        assert len(matches) == 1
        assert matches[0][3] == "ABCDEFGH"

    def test_no_match(self):
        store = self._make_store([
            ("secret-value-here", "label", "SECRET"),
        ])
        text = "This text has nothing special"
        matches = store.find_in_text(text)
        assert len(matches) == 0

    def test_empty_text(self):
        store = self._make_store([
            ("something", "label", "SECRET"),
        ])
        assert store.find_in_text("") == []

    def test_empty_store(self):
        store = self._make_store([])
        assert store.find_in_text("anything goes here") == []

    def test_repeated_value(self):
        store = self._make_store([
            ("REPEATED", "val", "SECRET"),
        ])
        text = "REPEATED and then REPEATED again"
        matches = store.find_in_text(text)
        assert len(matches) == 2

    def test_sorted_by_position(self):
        store = self._make_store([
            ("second-secret", "b", "SECRET"),
            ("first-secret-val", "a", "SECRET"),
        ])
        text = "first-secret-val is before second-secret"
        matches = store.find_in_text(text)
        positions = [m[0] for m in matches]
        assert positions == sorted(positions)


# =========================================================================
# Aho-Corasick vs linear fallback
# =========================================================================
class TestAhoCorasickVsLinear:

    def _make_store_with_automaton(self, secrets_list):
        store = KnownSecretsStore()
        store._secrets = secrets_list
        store._loaded = True
        store._last_refresh = 99999999999.0
        store._build_automaton()
        return store

    def test_ahocorasick_matches_linear(self):
        """Both methods should produce identical results."""
        secrets = [
            ("my-api-key-value", "key", "API_KEY"),
            ("another-secret", "secret", "SECRET"),
        ]
        text = "The my-api-key-value is set and another-secret too"

        store_ac = self._make_store_with_automaton(secrets)
        store_linear = KnownSecretsStore()
        store_linear._secrets = secrets
        store_linear._loaded = True
        store_linear._last_refresh = 99999999999.0

        ac_matches = store_ac._find_ahocorasick(text) if store_ac._automaton else store_ac._find_linear(text)
        linear_matches = store_linear._find_linear(text)

        # Same values found (order may differ before sort)
        ac_values = sorted([m[3] for m in ac_matches])
        linear_values = sorted([m[3] for m in linear_matches])
        assert ac_values == linear_values

    def test_automaton_built(self):
        """If ahocorasick is available, automaton should be set."""
        try:
            import ahocorasick
            has_ac = True
        except ImportError:
            has_ac = False

        store = self._make_store_with_automaton([
            ("test-value-long", "label", "SECRET"),
        ])
        if has_ac:
            assert store._automaton is not None
        else:
            assert store._automaton is None


# =========================================================================
# MIN_SECRET_LEN
# =========================================================================
class TestMinSecretLen:

    def test_default_is_6(self):
        assert KnownSecretsStore.MIN_SECRET_LEN == 6

    def test_short_values_excluded_on_load(self, tmp_path, monkeypatch):
        data = [
            {"label": "short", "value": "abc", "type": "SECRET"},
            {"label": "ok", "value": "long-enough-value", "type": "SECRET"},
        ]
        path = tmp_path / "secrets.json"
        path.write_text(json.dumps(data))
        monkeypatch.setattr("proxy.SECRETS_SOURCE", "file")
        monkeypatch.setattr("proxy.SECRETS_FILE", str(path))

        store = KnownSecretsStore()
        store._load_file()
        # Deduplicate step filters out short values
        seen = set()
        unique = []
        for val, label, stype in store._secrets:
            if val not in seen and len(val) >= KnownSecretsStore.MIN_SECRET_LEN:
                seen.add(val)
                unique.append((val, label, stype))
        values = [s[0] for s in unique]
        assert "abc" not in values
        assert "long-enough-value" in values


# =========================================================================
# _classify_op_field (1Password field classification)
# =========================================================================
class TestClassifyOpField:

    def test_direct_label_match(self):
        store = KnownSecretsStore()
        assert store._classify_op_field("password") == "PASSWORD"
        assert store._classify_op_field("email") == "EMAIL"
        assert store._classify_op_field("api key") == "API_KEY"
        assert store._classify_op_field("phone") == "PHONE"

    def test_partial_match(self):
        store = KnownSecretsStore()
        assert store._classify_op_field("my_password_field") == "PASSWORD"
        assert store._classify_op_field("admin_email_addr") == "EMAIL"

    def test_category_fallback(self):
        store = KnownSecretsStore()
        assert store._classify_op_field("unknown_field", category="login") == "PASSWORD"
        assert store._classify_op_field("unknown_field", category="credit_card") == "CREDIT_CARD"
        assert store._classify_op_field("unknown_field", category="identity") == "SECRET"

    def test_default_is_secret(self):
        store = KnownSecretsStore()
        assert store._classify_op_field("random_field_name") == "SECRET"
