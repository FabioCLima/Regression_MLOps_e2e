#!/usr/bin/env bash
set -euo pipefail

source deploy.env
if [[ -f deploy.runtime.env ]]; then
  source deploy.runtime.env
fi

FRONTEND_SERVICE="${FRONTEND_SERVICE:-regression-mlops-e2e-frontend-service}"

usage() {
  cat <<'EOF'
Usage:
  DEMO_TEARDOWN_CONFIRM=delete-demo-infra scripts/aws_teardown_demo_infra.sh

Deletes the demo ECS services, Application Load Balancers, and target groups.
Keeps S3, ECR, IAM roles, CloudWatch log groups, task definitions, and budgets.

Use scripts/aws_recreate_demo_infra.sh to recreate a public demo endpoint later.
EOF
}

if [[ "${DEMO_TEARDOWN_CONFIRM:-}" != "delete-demo-infra" ]]; then
  usage >&2
  exit 2
fi

service_exists() {
  local service_name="$1"
  aws ecs describe-services \
    --cluster "${ECS_CLUSTER}" \
    --services "${service_name}" \
    --region "${AWS_REGION}" \
    --query 'length(services[?status!=`INACTIVE`])' \
    --output text 2>/dev/null
}

delete_service() {
  local service_name="$1"
  if [[ "$(service_exists "${service_name}")" == "0" ]]; then
    printf 'ECS service already absent: %s\n' "${service_name}"
    return
  fi

  printf 'Stopping ECS service: %s\n' "${service_name}"
  aws ecs update-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${service_name}" \
    --desired-count 0 \
    --region "${AWS_REGION}" >/dev/null

  printf 'Deleting ECS service: %s\n' "${service_name}"
  aws ecs delete-service \
    --cluster "${ECS_CLUSTER}" \
    --service "${service_name}" \
    --force \
    --region "${AWS_REGION}" >/dev/null

  aws ecs wait services-inactive \
    --cluster "${ECS_CLUSTER}" \
    --services "${service_name}" \
    --region "${AWS_REGION}" || true
}

delete_load_balancer() {
  local lb_arn="${1:-}"
  local label="$2"
  if [[ -z "${lb_arn}" ]]; then
    printf 'No load balancer ARN set for %s.\n' "${label}"
    return
  fi

  if ! aws elbv2 describe-load-balancers \
    --load-balancer-arns "${lb_arn}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
    printf 'Load balancer already absent: %s\n' "${label}"
    return
  fi

  printf 'Deleting load balancer: %s\n' "${label}"
  aws elbv2 delete-load-balancer \
    --load-balancer-arn "${lb_arn}" \
    --region "${AWS_REGION}"

  aws elbv2 wait load-balancers-deleted \
    --load-balancer-arns "${lb_arn}" \
    --region "${AWS_REGION}"
}

delete_target_group() {
  local tg_arn="${1:-}"
  local label="$2"
  if [[ -z "${tg_arn}" ]]; then
    printf 'No target group ARN set for %s.\n' "${label}"
    return
  fi

  if ! aws elbv2 describe-target-groups \
    --target-group-arns "${tg_arn}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
    printf 'Target group already absent: %s\n' "${label}"
    return
  fi

  printf 'Deleting target group: %s\n' "${label}"
  for attempt in {1..12}; do
    if aws elbv2 delete-target-group \
      --target-group-arn "${tg_arn}" \
      --region "${AWS_REGION}"; then
      return
    fi
    printf 'Target group still in use, retrying (%s/12): %s\n' "${attempt}" "${label}"
    sleep 10
  done

  printf 'Could not delete target group after retries: %s\n' "${label}" >&2
  return 1
}

delete_service "${ECS_SERVICE}"
delete_service "${FRONTEND_SERVICE}"

delete_load_balancer "${ALB_ARN:-}" "backend"
delete_load_balancer "${FRONTEND_ALB_ARN:-}" "frontend"

delete_target_group "${TG_ARN:-}" "backend"
delete_target_group "${FRONTEND_TG_ARN:-}" "frontend"

printf '\nDemo infrastructure removed. Persistent assets kept: S3, ECR, task definitions, IAM roles, logs, and budget.\n'
