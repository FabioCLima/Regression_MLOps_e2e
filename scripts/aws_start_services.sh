#!/usr/bin/env bash
set -euo pipefail
source deploy.env
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --desired-count 1 \
  --region "${AWS_REGION}" >/dev/null
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "regression-mlops-e2e-frontend-service" \
  --desired-count 1 \
  --region "${AWS_REGION}" >/dev/null
printf 'Requested desired-count=1 for backend and frontend ECS services.
'
