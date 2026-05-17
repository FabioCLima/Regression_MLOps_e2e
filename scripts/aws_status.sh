#!/usr/bin/env bash
set -euo pipefail
source deploy.env
if [[ -f deploy.runtime.env ]]; then
  source deploy.runtime.env
fi

FRONTEND_SERVICE="${FRONTEND_SERVICE:-regression-mlops-e2e-frontend-service}"
FRONTEND_ALB_DNS="${FRONTEND_ALB_DNS:-reg-mlops-fe-alb-1328758802.us-east-1.elb.amazonaws.com}"

if ! aws ecs describe-services \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" "${FRONTEND_SERVICE}" \
  --region "${AWS_REGION}" \
  --query 'services[*].[serviceName,desiredCount,runningCount,pendingCount,deployments[0].rolloutState]' \
  --output table; then
  printf 'ECS services are not fully available. If you tore down demo infrastructure, run scripts/aws_recreate_demo_infra.sh.\n'
fi

if [[ -n "${ALB_DNS:-}" ]]; then
  printf '\nBackend health:\n'
  curl -fsS "http://${ALB_DNS}/health" || true
else
  printf '\nBackend health: ALB_DNS is not set.\n'
fi

if [[ -n "${FRONTEND_ALB_DNS:-}" ]]; then
  printf '\n\nFrontend health:\n'
  curl -fsS "http://${FRONTEND_ALB_DNS}/_stcore/health" || true
else
  printf '\n\nFrontend health: FRONTEND_ALB_DNS is not set.\n'
fi
printf '\n'
