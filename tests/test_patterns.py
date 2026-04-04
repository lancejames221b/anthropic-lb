"""
Test every regex pattern in _build_builtin_patterns() (27 patterns).

For each pattern: true positives, true negatives, and edge cases.
Tests the real compiled regex -- no mocking.

Author: Lance James, Unit 221B, Inc.
"""

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxy import _build_builtin_patterns, _luhn_valid


# Build patterns once for the module
PATTERNS = {p[0]: p for p in _build_builtin_patterns()}


def _match(pattern_name, text):
    """Return all matches for a named pattern in text."""
    _, regex, _, validator = PATTERNS[pattern_name]
    matches = []
    for m in regex.finditer(text):
        full = m.group(0)
        if validator is not None:
            digits_only = re.sub(r'\D', '', full)
            if not validator(digits_only):
                continue
        matches.append(full)
    return matches


def _has_match(pattern_name, text):
    """Return True if pattern_name matches anywhere in text."""
    return len(_match(pattern_name, text)) > 0


# =========================================================================
# EMAIL
# =========================================================================
class TestEmailPattern:

    @pytest.mark.parametrize("email", [
        "user@example.com",
        "first.last@domain.co.uk",
        "name+tag@sub.domain.org",
        "test_email@company.net",
        "a@b.co",
        "UPPER@CASE.COM",
        "mixed.Case@Domain.Org",
        "user123@test.io",
        "special-chars@my-domain.info",
        "dotted.user.name@long.subdomain.example.com",
    ])
    def test_true_positive(self, email):
        assert _has_match("email", email)

    @pytest.mark.parametrize("text", [
        "not-an-email",
        "missing@tld",
        "@no-user.com",
        "spaces in@email .com",
        "user@.com",
        "plaintext",
        "12345",
    ])
    def test_true_negative(self, text):
        assert not _has_match("email", text)

    def test_embedded_in_sentence(self):
        text = "Contact us at support@company.com for help."
        matches = _match("email", text)
        assert "support@company.com" in matches

    def test_multiple_emails(self):
        text = "Email a@b.com or c@d.org"
        matches = _match("email", text)
        assert len(matches) == 2


# =========================================================================
# SSN (context-required)
# =========================================================================
class TestSSNPattern:

    @pytest.mark.parametrize("text", [
        "ssn: 123-45-6789",
        "Social Security Number: 456-78-9012",
        "SSN# 321-54-9876",
        "SS# 234-56-7890",
        "soc sec: 567-89-0123",
        "social security: 678-90-1234",
        "SSN:234-56-7890",
    ])
    def test_true_positive(self, text):
        assert _has_match("ssn", text)

    @pytest.mark.parametrize("text", [
        "123-45-6789",             # no SSN context keyword
        "ssn: 000-12-3456",        # area 000 is invalid
        "ssn: 666-12-3456",        # area 666 is invalid
        "ssn: 900-12-3456",        # area 9xx is invalid
        "ssn: 123-00-6789",        # group 00 is invalid
        "ssn: 123-45-0000",        # serial 0000 is invalid
        "phone: 555-123-4567",
        "reference: 111-22-3333",  # no SSN keyword
    ])
    def test_true_negative(self, text):
        assert not _has_match("ssn", text)


# =========================================================================
# CREDIT_CARD (Luhn-validated)
# =========================================================================
class TestCreditCardPattern:

    @pytest.mark.parametrize("cc", [
        "4111111111111111",    # Visa
        "5500000000000004",    # MasterCard
        "378282246310005",     # Amex
        "6011111111111117",    # Discover
        "3530111333300000",    # JCB
    ])
    def test_true_positive(self, cc):
        assert _has_match("credit_card", cc)
        # Confirm Luhn validation passes
        assert _luhn_valid(cc)

    @pytest.mark.parametrize("cc", [
        "4111111111111112",    # Fails Luhn
        "1234567890123456",    # Not a valid prefix
        "41111111111111",      # Too short for Visa
        "000000000000",        # All zeros, too short
    ])
    def test_true_negative(self, cc):
        # The regex might match but Luhn validator should reject
        assert not _has_match("credit_card", cc)

    def test_luhn_algorithm(self):
        assert _luhn_valid("4111111111111111") is True
        assert _luhn_valid("4111111111111112") is False
        assert _luhn_valid("0000000000000000") is True  # all-zeros passes Luhn
        assert _luhn_valid("123") is False  # too short


# =========================================================================
# PHONE
# =========================================================================
class TestPhonePattern:

    @pytest.mark.parametrize("phone", [
        "+1-555-123-4567",
        "(555) 987-6543",
        "555-123-4567",
        "+1 555 123 4567",
        "(800) 555-1212",
        "+44-20-7946-0958",
        "555.123.4567",
    ])
    def test_true_positive(self, phone):
        assert _has_match("phone", phone)

    @pytest.mark.parametrize("text", [
        "12345",
        "555-12-4567",         # only 2 digits in middle group
        "phone",
        "1-800",
        "version 1.2.3",
    ])
    def test_true_negative(self, text):
        assert not _has_match("phone", text)


# =========================================================================
# AWS_KEY
# =========================================================================
class TestAWSKeyPattern:

    @pytest.mark.parametrize("key", [
        "AKIAIOSFODNN7EXAMPLE",
        "AKIAI44QH8DHBEXAMPLE",
    ])
    def test_true_positive(self, key):
        assert _has_match("aws_key", key)

    @pytest.mark.parametrize("text", [
        "AKIA123",               # too short
        "BKIAIOSFODNN7EXAMPLE",  # wrong prefix
        "akiaiosfodnn7example",  # lowercase
    ])
    def test_true_negative(self, text):
        assert not _has_match("aws_key", text)


# =========================================================================
# GITHUB_TOKEN
# =========================================================================
class TestGitHubTokenPattern:

    @pytest.mark.parametrize("token", [
        "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh1234",
        "ghs_abcdefghijklmnopqrstuvwxyz1234567890ab",
        "ghr_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",
        "gho_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",
        "ghu_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",
    ])
    def test_true_positive(self, token):
        assert _has_match("github_token", token)

    @pytest.mark.parametrize("text", [
        "ghp_short",              # too short
        "ghx_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl",  # invalid prefix
        "github_pat_short",       # pat needs 82 chars
    ])
    def test_true_negative(self, text):
        assert not _has_match("github_token", text)

    def test_github_pat_format(self):
        # github_pat_ needs exactly 82 chars after prefix
        pat = "github_pat_" + "A" * 82
        assert _has_match("github_token", pat)


# =========================================================================
# ANTHROPIC_KEY
# =========================================================================
class TestAnthropicKeyPattern:

    @pytest.mark.parametrize("key", [
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz",
        "sk-ant-oat01-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
    ])
    def test_true_positive(self, key):
        assert _has_match("anthropic_key", key)

    @pytest.mark.parametrize("text", [
        "sk-ant-wrong",
        "sk-ant-api03-short",     # too short (< 20 chars after prefix)
        "sk-other-api03-abcdefghijklmnopqrstuvwxyz",
    ])
    def test_true_negative(self, text):
        assert not _has_match("anthropic_key", text)


# =========================================================================
# BEARER_TOKEN
# =========================================================================
class TestBearerTokenPattern:

    @pytest.mark.parametrize("text", [
        "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "Bearer sk-ant-oat01-longtokenvalue1234",
    ])
    def test_true_positive(self, text):
        assert _has_match("bearer_token", text)

    @pytest.mark.parametrize("text", [
        "Bearer short",         # < 20 chars
        "bearer TOKEN123456789012345",   # lowercase bearer -- case matters in header
        "NotBearer eyJhbGciOi",
    ])
    def test_true_negative(self, text):
        assert not _has_match("bearer_token", text)


# =========================================================================
# IPV4
# =========================================================================
class TestIPv4Pattern:

    @pytest.mark.parametrize("ip", [
        "192.168.1.1",
        "10.0.0.1",
        "255.255.255.255",
        "0.0.0.0",
        "127.0.0.1",
        "8.8.8.8",
    ])
    def test_true_positive(self, ip):
        assert _has_match("ipv4", ip)

    @pytest.mark.parametrize("text", [
        "256.1.2.3",
        "192.168.1",
        "999.999.999.999",
    ])
    def test_true_negative(self, text):
        assert not _has_match("ipv4", text)

    def test_partial_match_in_longer_string(self):
        """The regex can match '1.2.3.4' inside '1.2.3.4.5' (expected behavior)."""
        matches = _match("ipv4", "1.2.3.4.5")
        # The regex sees a valid quad in there
        assert len(matches) >= 1

    def test_embedded(self):
        text = "Server at 10.0.0.42 is down"
        matches = _match("ipv4", text)
        assert "10.0.0.42" in matches


# =========================================================================
# IPV6
# =========================================================================
class TestIPv6Pattern:

    def test_full_form(self):
        """Full 8-segment IPv6 addresses match."""
        assert _has_match("ipv6", "2001:0db8:85a3:0000:0000:8a2e:0370:7334")
        assert _has_match("ipv6", "fe80:0000:0000:0000:0000:0000:0000:0001")

    def test_compressed_with_trailing_segments(self):
        """Compressed forms match when followed by hex segments (word boundary works)."""
        # fe80::1 matches the trailing :: branch as 'fe80::' because
        # the \b at end anchors against the digit '1'
        matches = _match("ipv6", "fe80::1")
        assert len(matches) >= 1

    def test_mixed_compression(self):
        """Mixed :: forms with segments on both sides match."""
        matches = _match("ipv6", "2001:db8::ff00:42:8329")
        assert len(matches) >= 1

    def test_bare_shortened_not_matched(self):
        """KNOWN LIMITATION: bare 'fe80::', '::1', '::ffff' do not match.
        The \\b word boundary at the end of :: has no word character to anchor
        against when :: is at the end of the string or before a space.
        This documents the current regex behavior."""
        assert not _has_match("ipv6", "fe80::")
        assert not _has_match("ipv6", "::1")
        assert not _has_match("ipv6", "::ffff")


# =========================================================================
# MAC_ADDRESS
# =========================================================================
class TestMACAddressPattern:

    @pytest.mark.parametrize("mac", [
        "00:1A:2B:3C:4D:5E",
        "aa:bb:cc:dd:ee:ff",
        "00-1A-2B-3C-4D-5E",
    ])
    def test_true_positive(self, mac):
        assert _has_match("mac_address", mac)

    @pytest.mark.parametrize("text", [
        "00:1A:2B:3C:4D",      # too short
        "GG:HH:II:JJ:KK:LL",  # not hex
        "001A2B3C4D5E",        # no separators
    ])
    def test_true_negative(self, text):
        assert not _has_match("mac_address", text)


# =========================================================================
# PEM_PRIVATE_KEY
# =========================================================================
class TestPEMPrivateKeyPattern:

    def test_rsa_key(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----"
        assert _has_match("pem_private_key", key)

    def test_ec_key(self):
        key = "-----BEGIN EC PRIVATE KEY-----\nMHQCAQEE...\n-----END EC PRIVATE KEY-----"
        assert _has_match("pem_private_key", key)

    def test_generic_key(self):
        key = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg...\n-----END PRIVATE KEY-----"
        assert _has_match("pem_private_key", key)

    def test_encrypted_key(self):
        key = "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHDBOBg...\n-----END ENCRYPTED PRIVATE KEY-----"
        assert _has_match("pem_private_key", key)

    def test_openssh_key(self):
        key = "-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn...\n-----END OPENSSH PRIVATE KEY-----"
        assert _has_match("pem_private_key", key)

    def test_not_certificate(self):
        cert = "-----BEGIN CERTIFICATE-----\nMIIC...\n-----END CERTIFICATE-----"
        assert not _has_match("pem_private_key", cert)


# =========================================================================
# PEM_CERTIFICATE
# =========================================================================
class TestCertificatePattern:

    def test_certificate(self):
        cert = "-----BEGIN CERTIFICATE-----\nMIICdTCCAd4CCQC...\n-----END CERTIFICATE-----"
        assert _has_match("pem_certificate", cert)

    def test_not_private_key(self):
        key = "-----BEGIN PRIVATE KEY-----\nMIIEvg...\n-----END PRIVATE KEY-----"
        assert not _has_match("pem_certificate", key)


# =========================================================================
# JWT
# =========================================================================
class TestJWTPattern:

    def test_regex_matches_jwt(self):
        """The JWT regex itself matches valid JWT tokens."""
        _, regex, _, _ = PATTERNS["jwt"]
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert regex.search(jwt) is not None

    def test_validator_receives_digits_only(self):
        """KNOWN ISSUE: _redact_text strips non-digits before calling validator,
        so the JWT _jwt_valid() validator always fails (it needs dots to split).
        This means the full _match pipeline rejects JWTs even though the regex matches.
        Documenting current behavior for regression tracking."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        # The _match helper simulates _redact_text which strips digits and calls validator
        # JWT validator splits on dots, but digits_only has no dots -> returns False
        assert not _has_match("jwt", jwt)

    def test_short_segments_rejected(self):
        jwt = "eyJ.eyJ.ab"
        assert not _has_match("jwt", jwt)

    def test_not_jwt(self):
        assert not _has_match("jwt", "not.a.jwt")
        assert not _has_match("jwt", "abc.def.ghi")


# =========================================================================
# SLACK_TOKEN
# =========================================================================
class TestSlackTokenPattern:

    @pytest.mark.parametrize("token", [
        "xoxb-1234567890-abcdefgh",
        "xoxp-1234567890-abcdefgh",
        "xoxa-1234567890-abcdefgh",
        "xoxr-1234567890-abcdefgh",
        "xoxs-1234567890-abcdefgh",
    ])
    def test_true_positive(self, token):
        assert _has_match("slack_token", token)

    def test_true_negative(self):
        assert not _has_match("slack_token", "xoxz-invalid")
        assert not _has_match("slack_token", "xoxb-short")


# =========================================================================
# STRIPE_KEY
# =========================================================================
class TestStripeKeyPattern:

    # Keys constructed at runtime to avoid GitHub push protection false positives
    _STRIPE_SUFFIX = "_test_51234567890abcdefghijklmnop"

    @pytest.mark.parametrize("prefix", ["sk", "pk", "rk"])
    def test_true_positive(self, prefix):
        key = prefix + self._STRIPE_SUFFIX
        assert _has_match("stripe_key", key)

    def test_true_negative(self):
        assert not _has_match("stripe_key", "sk_invalid_short")
        assert not _has_match("stripe_key", "xx_live_51234567890abcdef")


# =========================================================================
# TWILIO_KEY
# =========================================================================
class TestTwilioKeyPattern:

    def test_true_positive(self):
        key = "SK" + "a1b2c3d4" * 4  # 32 hex chars
        assert _has_match("twilio_key", key)

    def test_true_negative(self):
        assert not _has_match("twilio_key", "SK12345")        # too short
        assert not _has_match("twilio_key", "TK" + "ab" * 16)  # wrong prefix


# =========================================================================
# SENDGRID_KEY
# =========================================================================
class TestSendGridKeyPattern:

    def test_true_positive(self):
        key = "SG." + "A" * 22 + "." + "B" * 43
        assert _has_match("sendgrid_key", key)

    def test_true_negative(self):
        assert not _has_match("sendgrid_key", "SG.short.short")
        assert not _has_match("sendgrid_key", "XX." + "A" * 22 + "." + "B" * 43)


# =========================================================================
# GENERIC_SECRET (context-required)
# =========================================================================
class TestGenericSecretPattern:

    @pytest.mark.parametrize("text", [
        "password=MySecretPass123",
        "api_key: abcdefghij123456",
        'secret="long_secret_value_here"',
        "access_token=tokenvalue12345678",
        "auth_token: someauthtoken12345",
        "apikey=ABCDEFGHIJKLMNOP",
    ])
    def test_true_positive(self, text):
        assert _has_match("generic_secret", text)

    @pytest.mark.parametrize("text", [
        "password=short",          # too short (< 8 chars)
        "no_keyword=longvalue123",  # no matching keyword
        "just a plain sentence",
    ])
    def test_true_negative(self, text):
        assert not _has_match("generic_secret", text)


# =========================================================================
# BTC_P2PKH
# =========================================================================
class TestBTCP2PKHPattern:

    @pytest.mark.parametrize("addr", [
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",   # Satoshi's address
        "3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy",    # P2SH
    ])
    def test_true_positive(self, addr):
        assert _has_match("btc_p2pkh", addr)

    def test_true_negative(self):
        # Address starting with 2 is not valid
        assert not _has_match("btc_p2pkh", "2A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")


# =========================================================================
# BTC_BECH32
# =========================================================================
class TestBTCBech32Pattern:

    def test_true_positive(self):
        addr = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4"
        assert _has_match("btc_bech32", addr)

    def test_true_negative(self):
        assert not _has_match("btc_bech32", "bc1short")
        assert not _has_match("btc_bech32", "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx")  # testnet


# =========================================================================
# ETH_ADDRESS
# =========================================================================
class TestETHAddressPattern:

    def test_true_positive(self):
        addr = "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD13"
        assert _has_match("eth_address", addr)

    def test_true_negative(self):
        assert not _has_match("eth_address", "0xshort")
        assert not _has_match("eth_address", "0x" + "G" * 40)  # non-hex


# =========================================================================
# US_PASSPORT (context-required)
# =========================================================================
class TestPassportPattern:

    @pytest.mark.parametrize("text", [
        "passport number: 123456789",
        "Passport No: A12345678",
        "Passport# 987654321",
        "passport num: 012345678",
    ])
    def test_true_positive(self, text):
        assert _has_match("us_passport", text)

    @pytest.mark.parametrize("text", [
        "123456789",              # no passport keyword
        "passport number: 123",    # too short
    ])
    def test_true_negative(self, text):
        assert not _has_match("us_passport", text)


# =========================================================================
# IBAN
# =========================================================================
class TestIBANPattern:

    @pytest.mark.parametrize("iban", [
        "GB29 NWBK 6016 1331 9268 19",
        "DE89370400440532013000",
        "FR7630006000011234567890189",
    ])
    def test_true_positive(self, iban):
        assert _has_match("iban", iban)

    def test_true_negative(self):
        assert not _has_match("iban", "XX12")  # too short


# =========================================================================
# EIN (context-required)
# =========================================================================
class TestEINPattern:

    @pytest.mark.parametrize("text", [
        "EIN: 12-3456789",
        "employer identification number: 98-7654321",
        "ein# 55-1234567",
        "Employer ID: 12-3456789",
    ])
    def test_true_positive(self, text):
        assert _has_match("ein", text)

    @pytest.mark.parametrize("text", [
        "12-3456789",              # no EIN keyword
        "EIN: 123456789",          # no hyphen
    ])
    def test_true_negative(self, text):
        assert not _has_match("ein", text)


# =========================================================================
# DRIVERS_LICENSE (context-required)
# =========================================================================
class TestDriversLicensePattern:

    @pytest.mark.parametrize("text", [
        "driver's license: D12345678",
        "DL: AB123456",
        "Drivers License Number: F987654321",
        "drivers lic: X1234567890",
    ])
    def test_true_positive(self, text):
        assert _has_match("drivers_license", text)

    @pytest.mark.parametrize("text", [
        "D12345678",               # no keyword
        "driver's license: AB",     # too short
    ])
    def test_true_negative(self, text):
        assert not _has_match("drivers_license", text)


# =========================================================================
# DATE_OF_BIRTH (context-required)
# =========================================================================
class TestDateOfBirthPattern:

    @pytest.mark.parametrize("text", [
        "DOB: 01/15/1990",
        "date of birth: 1985-06-23",
        "birthday: 12/25/1980",
        "born on 03-14-1975",
        "d.o.b. 07.04.2001",
        "birthdate: 2000-01-01",
    ])
    def test_true_positive(self, text):
        assert _has_match("date_of_birth", text)

    @pytest.mark.parametrize("text", [
        "01/15/1990",              # no DOB keyword
        "DOB: unknown",
        "the date was 2023-01-01",  # no DOB keyword
    ])
    def test_true_negative(self, text):
        assert not _has_match("date_of_birth", text)


# =========================================================================
# Pattern count sanity
# =========================================================================
class TestPatternInventory:

    def test_27_patterns(self):
        """Verify we have all 27 builtin patterns."""
        patterns = _build_builtin_patterns()
        assert len(patterns) == 27

    def test_all_patterns_named(self):
        """Every pattern has a unique name."""
        patterns = _build_builtin_patterns()
        names = [p[0] for p in patterns]
        assert len(names) == len(set(names))

    def test_expected_pattern_names(self):
        """Verify the full list of expected pattern names."""
        expected = {
            "email", "ssn", "credit_card", "phone",
            "aws_key", "github_token", "anthropic_key", "bearer_token",
            "ipv4", "ipv6", "mac_address",
            "pem_private_key", "pem_certificate", "jwt",
            "slack_token", "stripe_key", "twilio_key", "sendgrid_key",
            "generic_secret",
            "btc_p2pkh", "btc_bech32", "eth_address",
            "us_passport", "iban", "ein", "drivers_license", "date_of_birth",
        }
        actual = {p[0] for p in _build_builtin_patterns()}
        assert actual == expected

    def test_all_patterns_have_type_labels(self):
        """Every pattern has a non-empty type label string."""
        for name, _, label, _ in _build_builtin_patterns():
            assert isinstance(label, str) and len(label) > 0, f"{name} missing type label"
