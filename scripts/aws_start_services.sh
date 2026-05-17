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
  --desired-count 1 \
  --region "${AWS_REGION}" >/dev/null || {
  printf 'Backend ECS service was not found. If demo infrastructure was torn down, run scripts/aws_recreate_demo_infra.sh.\n' >&2
  exit 1
}
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${FRONTEND_SERVICE}" \
  --desired-count 1 \
  --region "${AWS_REGION}" >/dev/null || {
  printf 'Frontend ECS service was not found. If demo infrastructure was torn down, run scripts/aws_recreate_demo_infra.sh.\n' >&2
  exit 1
}
printf 'Requested desired-count=1 for backend and frontend ECS services.
'
