#!/usr/bin/env bash
set -euo pipefail
source deploy.env
if ! aws ce get-cost-and-usage \
  --time-period Start="$(date -u -d '7 days ago' +%F)",End="$(date -u +%F)" \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output table; then
  printf '
Cost Explorer did not return data yet. This can happen when billing data is not ingested for the selected period or Cost Explorer was recently enabled. Try again later, or check the AWS Billing console.
' >&2
fi
