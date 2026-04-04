#!/bin/bash
# test-docker.sh — Spin up anthropic-lb in Docker and run smoke tests
# Uses port 9891 (proxy) and 9892 (dashboard) to avoid conflicting with live 8891

set -euo pipefail

LB_URL="http://localhost:9891"
DASH_URL="http://localhost:9892"

echo "=== anthropic-lb Docker smoke test ==="
echo ""

# 1. Health check
echo "[1/5] Health check..."
HEALTH=$(curl -sf "$LB_URL/health" 2>/dev/null) || { echo "FAIL: health endpoint unreachable"; exit 1; }
echo "  $HEALTH"

# 2. Status endpoint
echo "[2/5] Status endpoint..."
STATUS=$(curl -sf "$LB_URL/status" 2>/dev/null) || { echo "FAIL: status unreachable"; exit 1; }
ACCOUNTS=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('accounts',{})))")
STRATEGY=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('strategy','?'))")
PII_MODE=$(echo "$STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pii',{}).get('mode',d.get('pii_mode','?')))")
echo "  accounts=$ACCOUNTS strategy=$STRATEGY pii=$PII_MODE"

# 3. PII test endpoint
echo "[3/5] PII detection test..."
PII_RESULT=$(curl -sf -X POST "$LB_URL/test-pii" \
  -H "Content-Type: application/json" \
  -d '{"text":"Email john@acme.com, SSN: 123-45-6789, call 555-867-5309, card 4111111111111111"}' 2>/dev/null)
PII_COUNT=$(echo "$PII_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pii_detected',0))")
ROUNDTRIP=$(echo "$PII_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('round_trip_ok',False))")
echo "  detected=$PII_COUNT round_trip_ok=$ROUNDTRIP"
if [ "$PII_COUNT" -lt 3 ]; then
  echo "  WARN: expected at least 3 PII detections, got $PII_COUNT"
fi

# 4. Dashboard reachable
echo "[4/5] Dashboard..."
DASH=$(curl -sf -o /dev/null -w "%{http_code}" "$DASH_URL/" 2>/dev/null) || DASH="000"
if [ "$DASH" = "200" ]; then
  echo "  OK (HTTP 200) at $DASH_URL"
else
  echo "  WARN: dashboard returned HTTP $DASH"
fi

# 5. Proxy a real API call (only if keys are real)
echo "[5/5] Proxy API call (messages endpoint)..."
API_RESULT=$(curl -sf -X POST "$LB_URL/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: test" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 50,
    "messages": [{"role":"user","content":"Say hello in exactly 3 words."}]
  }' 2>/dev/null) || API_RESULT="ERROR"

if echo "$API_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('content') else 1)" 2>/dev/null; then
  ACCOUNT=$(curl -sI -X POST "$LB_URL/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: test" \
    -H "anthropic-version: 2023-06-01" \
    -d '{"model":"claude-haiku-4-5-20251001","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null \
    | grep -i "x-lb-account" | tr -d '\r' || echo "x-lb-account: unknown")
  echo "  OK — routed via $ACCOUNT"
else
  echo "  API call returned: $(echo "$API_RESULT" | head -c 200)"
  echo "  (this is expected if keys.json has placeholder keys)"
fi

echo ""
echo "=== Done ==="
echo "Dashboard: $DASH_URL"
echo "Proxy:     $LB_URL"
echo "Status:    $LB_URL/status"
echo "PII test:  curl -X POST $LB_URL/test-pii -H 'Content-Type: application/json' -d '{\"text\":\"test@email.com\"}'"
