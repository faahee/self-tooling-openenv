#!/usr/bin/env bash
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  echo "Usage: $0 <ping_url> [repo_dir]"
  exit 1
fi

echo "Step 1: pinging $PING_URL/reset"
HTTP_CODE=$(curl -s -o /tmp/openenv-reset.out -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{}' "$PING_URL/reset" --max-time 30 || printf "000")
if [ "$HTTP_CODE" != "200" ]; then
  echo "HF Space ping failed with code $HTTP_CODE"
  exit 1
fi

echo "Step 2: docker build"
docker build "$REPO_DIR"

echo "Step 3: openenv validate"
openenv validate

echo "Validation checks completed."
