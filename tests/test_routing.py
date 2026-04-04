"""
Test account selection and routing logic.

Tests: pick_account_least_loaded, pick_account_round_robin,
429 cooldown, OAuth detection, API key detection, unified scoring.

Author: Lance James, Unit 221B, Inc.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy import (
    _set_auth_headers,
    _compute_unified_score,
    SlidingWindowTracker,
)


# =========================================================================
# _set_auth_headers
# =========================================================================
class TestSetAuthHeaders:

    def test_oauth_key_gets_bearer(self):
        """sk-ant-oat* keys should get Bearer auth + beta header."""
        headers = {}
        _set_auth_headers(headers, "sk-ant-oat01-testkey123456789")
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer sk-ant-oat01-testkey123456789"
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")

    def test_oauth_removes_xapi_key(self):
        """OAuth should remove x-api-key if present."""
        headers = {"x-api-key": "old-value"}
        _set_auth_headers(headers, "sk-ant-oat01-testkey123456789")
        assert "x-api-key" not in headers

    def test_api_key_gets_xapi_header(self):
        """sk-ant-api03-* keys should get x-api-key header."""
        headers = {}
        _set_auth_headers(headers, "sk-ant-api03-testkey123456789")
        assert headers["x-api-key"] == "sk-ant-api03-testkey123456789"
        assert "Authorization" not in headers

    def test_api_key_removes_authorization(self):
        """API key should remove Authorization if present."""
        headers = {"Authorization": "Bearer old-token"}
        _set_auth_headers(headers, "sk-ant-api03-testkey123456789")
        assert "Authorization" not in headers
        assert "authorization" not in headers

    def test_oauth_preserves_existing_beta(self):
        """OAuth should append to existing anthropic-beta, not replace."""
        headers = {"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
        _set_auth_headers(headers, "sk-ant-oat01-testkey123456789")
        beta = headers["anthropic-beta"]
        assert "max-tokens-3-5-sonnet-2024-07-15" in beta
        assert "oauth-2025-04-20" in beta

    def test_oauth_no_duplicate_beta(self):
        """Should not duplicate oauth beta if already present."""
        headers = {"anthropic-beta": "oauth-2025-04-20"}
        _set_auth_headers(headers, "sk-ant-oat01-testkey123456789")
        beta = headers["anthropic-beta"]
        assert beta.count("oauth-2025-04-20") == 1


# =========================================================================
# SlidingWindowTracker
# =========================================================================
class TestSlidingWindowTracker:

    def test_record_and_count(self):
        tracker = SlidingWindowTracker(["a", "b"])
        tracker.record_request("a", tokens_used=100)
        tracker.record_request("a", tokens_used=200)
        tracker.record_request("b", tokens_used=50)
        assert tracker.get_rpm("a") == 2
        assert tracker.get_rpm("b") == 1
        assert tracker.get_tpm("a") == 300
        assert tracker.get_tpm("b") == 50

    def test_window_pruning(self):
        """Entries older than 60s should be pruned."""
        tracker = SlidingWindowTracker(["a"])
        # Manually insert old entry
        old_time = time.time() - 70  # 70 seconds ago
        tracker._requests["a"].append(old_time)
        tracker._tokens["a"].append((old_time, 100))
        # New entry
        tracker.record_request("a", tokens_used=50)
        assert tracker.get_rpm("a") == 1  # old one pruned
        assert tracker.get_tpm("a") == 50

    def test_separate_accounts(self):
        tracker = SlidingWindowTracker(["x", "y"])
        tracker.record_request("x", tokens_used=100)
        assert tracker.get_rpm("x") == 1
        assert tracker.get_rpm("y") == 0


# =========================================================================
# _compute_unified_score
# =========================================================================
class TestComputeUnifiedScore:

    def _make_stats(self, **kwargs):
        """Build a stats dict with defaults for unified scoring."""
        defaults = {
            "unified_status": "allowed",
            "unified_5h_utilization": 0.0,
            "unified_7d_utilization": 0.0,
            "unified_5h_reset": None,
            "unified_7d_reset": None,
            "unified_fallback": None,
            "unified_representative_claim": None,
        }
        defaults.update(kwargs)
        return defaults

    def test_fresh_account_full_capacity(self):
        """Account with 0% utilization should score near 1.0."""
        s = self._make_stats(unified_5h_utilization=0.0, unified_7d_utilization=0.0)
        result = _compute_unified_score("test", s, time.time())
        assert result is not None
        score, breakdown = result
        assert score >= 0.9

    def test_rejected_returns_none(self):
        """Rejected status should return None (hard skip)."""
        s = self._make_stats(unified_status="rejected")
        result = _compute_unified_score("test", s, time.time())
        assert result is None

    def test_high_utilization_low_score(self):
        """90%+ utilization should produce a low score."""
        s = self._make_stats(unified_5h_utilization=0.95, unified_7d_utilization=0.90)
        result = _compute_unified_score("test", s, time.time())
        assert result is not None
        score, _ = result
        assert score < 0.15

    def test_allowed_warning_penalty(self):
        """allowed_warning should apply 30% penalty."""
        s_allowed = self._make_stats(
            unified_status="allowed",
            unified_5h_utilization=0.5,
            unified_7d_utilization=0.5,
        )
        s_warning = self._make_stats(
            unified_status="allowed_warning",
            unified_5h_utilization=0.5,
            unified_7d_utilization=0.5,
        )
        result_allowed = _compute_unified_score("test", s_allowed, time.time())
        result_warning = _compute_unified_score("test", s_warning, time.time())
        assert result_allowed is not None and result_warning is not None
        # Warning score should be ~70% of allowed score
        assert result_warning[0] < result_allowed[0]

    def test_ttl_leniency_with_reset(self):
        """When reset is imminent, penalty should be reduced."""
        now = time.time()
        # 5h reset in 5 minutes (low TTL = high leniency relief)
        s = self._make_stats(
            unified_5h_utilization=0.8,
            unified_7d_utilization=0.3,
            unified_5h_reset=now + 300,  # 5 min from now
        )
        result = _compute_unified_score("test", s, now)
        assert result is not None
        score_with_reset, breakdown = result

        # Compare with no reset info (full penalty)
        s2 = self._make_stats(
            unified_5h_utilization=0.8,
            unified_7d_utilization=0.3,
        )
        result2 = _compute_unified_score("test", s2, now)
        score_no_reset = result2[0]

        # With imminent reset, leniency should give higher score
        assert score_with_reset > score_no_reset

    def test_breakdown_has_expected_keys(self):
        s = self._make_stats(unified_5h_utilization=0.3, unified_7d_utilization=0.2)
        result = _compute_unified_score("test", s, time.time())
        assert result is not None
        _, breakdown = result
        expected_keys = {
            "util_5h", "util_7d", "leniency_5h", "leniency_7d",
            "effective_5h", "effective_7d", "weighted_util",
            "velocity_penalty", "w_5h", "w_7d",
        }
        assert expected_keys == set(breakdown.keys())

    def test_7d_weighted_heavier(self):
        """7d window should carry more weight (W_7D=2.0 vs W_5H=1.0 by default)."""
        s_5h_heavy = self._make_stats(unified_5h_utilization=0.8, unified_7d_utilization=0.1)
        s_7d_heavy = self._make_stats(unified_5h_utilization=0.1, unified_7d_utilization=0.8)

        r1 = _compute_unified_score("test", s_5h_heavy, time.time())
        r2 = _compute_unified_score("test", s_7d_heavy, time.time())

        assert r1 is not None and r2 is not None
        # 7d-heavy should score lower because 7d has double weight
        assert r2[0] < r1[0]

    def test_soft_ceiling_sentinel(self):
        """Score between 0 and 0.05 should be clamped to 0.001 sentinel."""
        s = self._make_stats(unified_5h_utilization=0.97, unified_7d_utilization=0.97)
        result = _compute_unified_score("test", s, time.time())
        if result is not None:
            score, _ = result
            # Either None (hard skip) or sentinel (0.001)
            assert score <= 0.05


# =========================================================================
# 429 cooldown behavior
# =========================================================================
class TestCooldownBehavior:

    def test_cooldown_excludes_account(self):
        """Accounts in cooldown should not be selected."""
        import proxy

        # Save originals
        orig_keys = proxy.KEY_NAMES
        orig_keys_dict = proxy.KEYS
        orig_stats = proxy.STATS
        orig_limits = proxy.ACCOUNT_LIMITS
        orig_tracker = proxy._window_tracker

        now = time.time()
        proxy.KEY_NAMES = ["acct-a", "acct-b"]
        proxy.KEYS = {
            "acct-a": "sk-ant-api03-aaaa",
            "acct-b": "sk-ant-api03-bbbb",
        }
        proxy.ACCOUNT_LIMITS = {}
        proxy._window_tracker = SlidingWindowTracker(proxy.KEY_NAMES)
        proxy.STATS = {
            "acct-a": {
                "rate_limited_until": now + 60,  # in cooldown
                "rate_tokens_remaining": None,
                "rate_tokens_limit": None,
                "rate_requests_remaining": None,
                "rate_requests_limit": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
            "acct-b": {
                "rate_limited_until": 0,  # available
                "rate_tokens_remaining": None,
                "rate_tokens_limit": None,
                "rate_requests_remaining": None,
                "rate_requests_limit": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
        }

        name, key = proxy.pick_account_least_loaded()
        assert name == "acct-b"  # acct-a should be excluded

        # Restore
        proxy.KEY_NAMES = orig_keys
        proxy.KEYS = orig_keys_dict
        proxy.STATS = orig_stats
        proxy.ACCOUNT_LIMITS = orig_limits
        proxy._window_tracker = orig_tracker

    def test_all_cooldown_picks_soonest(self):
        """When all accounts are rate-limited, pick the one resetting soonest."""
        import proxy

        orig_keys = proxy.KEY_NAMES
        orig_keys_dict = proxy.KEYS
        orig_stats = proxy.STATS
        orig_limits = proxy.ACCOUNT_LIMITS
        orig_tracker = proxy._window_tracker

        now = time.time()
        proxy.KEY_NAMES = ["acct-a", "acct-b"]
        proxy.KEYS = {
            "acct-a": "sk-ant-api03-aaaa",
            "acct-b": "sk-ant-api03-bbbb",
        }
        proxy.ACCOUNT_LIMITS = {}
        proxy._window_tracker = SlidingWindowTracker(proxy.KEY_NAMES)
        proxy.STATS = {
            "acct-a": {
                "rate_limited_until": now + 120,  # 2 min cooldown
                "rate_tokens_remaining": None,
                "rate_tokens_limit": None,
                "rate_requests_remaining": None,
                "rate_requests_limit": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
            "acct-b": {
                "rate_limited_until": now + 30,  # 30 sec cooldown (soonest)
                "rate_tokens_remaining": None,
                "rate_tokens_limit": None,
                "rate_requests_remaining": None,
                "rate_requests_limit": None,
                "tokens_in": 0,
                "tokens_out": 0,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
        }

        name, key = proxy.pick_account_least_loaded()
        assert name == "acct-b"  # soonest reset

        proxy.KEY_NAMES = orig_keys
        proxy.KEYS = orig_keys_dict
        proxy.STATS = orig_stats
        proxy.ACCOUNT_LIMITS = orig_limits
        proxy._window_tracker = orig_tracker


# =========================================================================
# Least-loaded selection with token data
# =========================================================================
class TestLeastLoadedSelection:

    def test_picks_most_remaining(self):
        """Should pick the account with the most remaining capacity."""
        import proxy

        orig_keys = proxy.KEY_NAMES
        orig_keys_dict = proxy.KEYS
        orig_stats = proxy.STATS
        orig_limits = proxy.ACCOUNT_LIMITS
        orig_tracker = proxy._window_tracker

        proxy.KEY_NAMES = ["low-cap", "high-cap"]
        proxy.KEYS = {
            "low-cap": "sk-ant-api03-lowcap1234567890",
            "high-cap": "sk-ant-api03-highcap123456789",
        }
        proxy.ACCOUNT_LIMITS = {}
        proxy._window_tracker = SlidingWindowTracker(proxy.KEY_NAMES)
        proxy.STATS = {
            "low-cap": {
                "rate_limited_until": 0,
                "rate_tokens_remaining": 100,
                "rate_tokens_limit": 10000,
                "rate_requests_remaining": 5,
                "rate_requests_limit": 100,
                "tokens_in": 5000,
                "tokens_out": 5000,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
            "high-cap": {
                "rate_limited_until": 0,
                "rate_tokens_remaining": 9000,
                "rate_tokens_limit": 10000,
                "rate_requests_remaining": 90,
                "rate_requests_limit": 100,
                "tokens_in": 500,
                "tokens_out": 500,
                "unified_status": None,
                "capacity_score": None,
                "score_breakdown": None,
                "rate_limit_hits": 0,
            },
        }

        name, key = proxy.pick_account_least_loaded()
        assert name == "high-cap"

        proxy.KEY_NAMES = orig_keys
        proxy.KEYS = orig_keys_dict
        proxy.STATS = orig_stats
        proxy.ACCOUNT_LIMITS = orig_limits
        proxy._window_tracker = orig_tracker


# =========================================================================
# Round-robin selection
# =========================================================================
class TestRoundRobinSelection:

    def test_rotates(self):
        """Round-robin should cycle through accounts."""
        import proxy
        import itertools

        orig_keys = proxy.KEY_NAMES
        orig_keys_dict = proxy.KEYS
        orig_cycle = proxy.KEY_CYCLE

        proxy.KEY_NAMES = ["a", "b", "c"]
        proxy.KEYS = {
            "a": "sk-ant-api03-aaa",
            "b": "sk-ant-api03-bbb",
            "c": "sk-ant-api03-ccc",
        }
        proxy.KEY_CYCLE = itertools.cycle(range(3))

        names = []
        for _ in range(6):
            name, _ = proxy.pick_account_round_robin()
            names.append(name)

        # Should cycle: a, b, c, a, b, c
        assert names == ["a", "b", "c", "a", "b", "c"]

        proxy.KEY_NAMES = orig_keys
        proxy.KEYS = orig_keys_dict
        proxy.KEY_CYCLE = orig_cycle


# =========================================================================
# OAuth vs API key detection
# =========================================================================
class TestKeyTypeDetection:

    @pytest.mark.parametrize("key,is_oauth", [
        ("sk-ant-oat01-test123456789", True),
        ("sk-ant-oat02-abc123def456", True),
        ("sk-ant-api03-test123456789", False),
        ("sk-ant-api03-abc123def456", False),
    ])
    def test_oauth_detection(self, key, is_oauth):
        headers = {}
        _set_auth_headers(headers, key)
        if is_oauth:
            assert "Authorization" in headers
            assert headers["Authorization"].startswith("Bearer ")
        else:
            assert "x-api-key" in headers
            assert headers["x-api-key"] == key
