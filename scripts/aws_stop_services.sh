#!/usr/bin/env bash
set -euo pipefail
source deploy.env
if [[ -f deploy.runtime.env ]]; then
  source deploy.runtime.env
fi

FRONTEND_SERVICE="${FRONTEND_SERVICE:-regression-mlops-e2e-frontend-service}"

aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --desired-count 0 \
  --region "${AWS_REGION}" >/dev/null || true
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${FRONTEND_SERVICE}" \
  --desired-count 0 \
  --region "${AWS_REGION}" >/dev/null || true
printf 'Requested desired-count=0 for backend and frontend ECS services.
'
