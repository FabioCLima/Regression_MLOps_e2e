#!/usr/bin/env bash
set -euo pipefail
source deploy.env
aws ecs describe-services \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" "regression-mlops-e2e-frontend-service" \
  --region "${AWS_REGION}" \
  --query 'services[*].[serviceName,desiredCount,runningCount,pendingCount,deployments[0].rolloutState]' \
  --output table
printf '
Backend health:
'
curl -fsS "http://${ALB_DNS}/health" || true
printf '

Frontend health:
'
curl -fsS "http://reg-mlops-fe-alb-1328758802.us-east-1.elb.amazonaws.com/_stcore/health" || true
printf '
'
