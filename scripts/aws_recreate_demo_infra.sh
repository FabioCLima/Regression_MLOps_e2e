#!/usr/bin/env bash
set -euo pipefail

source deploy.env

BACKEND_ALB_NAME="${BACKEND_ALB_NAME:-regression-mlops-e2e-alb}"
BACKEND_TG_NAME="${BACKEND_TG_NAME:-regression-mlops-e2e-tg}"
FRONTEND_ALB_NAME="${FRONTEND_ALB_NAME:-reg-mlops-fe-alb}"
FRONTEND_TG_NAME="${FRONTEND_TG_NAME:-reg-mlops-fe-tg}"
FRONTEND_SERVICE="${FRONTEND_SERVICE:-regression-mlops-e2e-frontend-service}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

require_command aws
require_command jq

ensure_log_group() {
  local log_group="$1"
  aws logs create-log-group \
    --log-group-name "${log_group}" \
    --region "${AWS_REGION}" >/dev/null 2>&1 || true
}

load_balancer_arn_by_name() {
  local name="$1"
  aws elbv2 describe-load-balancers \
    --names "${name}" \
    --region "${AWS_REGION}" \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text 2>/dev/null || true
}

target_group_arn_by_name() {
  local name="$1"
  aws elbv2 describe-target-groups \
    --names "${name}" \
    --region "${AWS_REGION}" \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text 2>/dev/null || true
}

ensure_load_balancer() {
  local name="$1"
  local security_group="$2"
  local arn
  arn="$(load_balancer_arn_by_name "${name}")"
  if [[ -z "${arn}" || "${arn}" == "None" ]]; then
    printf 'Creating load balancer: %s\n' "${name}" >&2
    arn="$(aws elbv2 create-load-balancer \
      --name "${name}" \
      --subnets "${SUBNET_1}" "${SUBNET_2}" \
      --security-groups "${security_group}" \
      --scheme internet-facing \
      --type application \
      --ip-address-type ipv4 \
      --region "${AWS_REGION}" \
      --query 'LoadBalancers[0].LoadBalancerArn' \
      --output text)"
    aws elbv2 wait load-balancer-available \
      --load-balancer-arns "${arn}" \
      --region "${AWS_REGION}"
  else
    printf 'Reusing load balancer: %s\n' "${name}" >&2
  fi
  printf '%s\n' "${arn}"
}

ensure_target_group() {
  local name="$1"
  local port="$2"
  local health_path="$3"
  local arn
  arn="$(target_group_arn_by_name "${name}")"
  if [[ -z "${arn}" || "${arn}" == "None" ]]; then
    printf 'Creating target group: %s\n' "${name}" >&2
    arn="$(aws elbv2 create-target-group \
      --name "${name}" \
      --protocol HTTP \
      --port "${port}" \
      --vpc-id "${VPC_ID}" \
      --target-type ip \
      --health-check-protocol HTTP \
      --health-check-path "${health_path}" \
      --health-check-interval-seconds 30 \
      --health-check-timeout-seconds 5 \
      --healthy-threshold-count 2 \
      --unhealthy-threshold-count 3 \
      --matcher HttpCode=200 \
      --region "${AWS_REGION}" \
      --query 'TargetGroups[0].TargetGroupArn' \
      --output text)"
  else
    printf 'Reusing target group: %s\n' "${name}" >&2
  fi
  printf '%s\n' "${arn}"
}

ensure_listener() {
  local lb_arn="$1"
  local tg_arn="$2"
  local existing
  existing="$(aws elbv2 describe-listeners \
    --load-balancer-arn "${lb_arn}" \
    --region "${AWS_REGION}" \
    --query 'Listeners[?Port==`80`].ListenerArn | [0]' \
    --output text 2>/dev/null || true)"

  if [[ -n "${existing}" && "${existing}" != "None" ]]; then
    printf 'Reusing HTTP listener: %s\n' "${existing}" >&2
    return
  fi

  printf 'Creating HTTP listener.\n' >&2
  aws elbv2 create-listener \
    --load-balancer-arn "${lb_arn}" \
    --protocol HTTP \
    --port 80 \
    --default-actions "Type=forward,TargetGroupArn=${tg_arn}" \
    --region "${AWS_REGION}" >/dev/null
}

load_balancer_dns() {
  local arn="$1"
  aws elbv2 describe-load-balancers \
    --load-balancer-arns "${arn}" \
    --region "${AWS_REGION}" \
    --query 'LoadBalancers[0].DNSName' \
    --output text
}

service_exists() {
  local service_name="$1"
  aws ecs describe-services \
    --cluster "${ECS_CLUSTER}" \
    --services "${service_name}" \
    --region "${AWS_REGION}" \
    --query 'length(services[?status!=`INACTIVE`])' \
    --output text 2>/dev/null
}

register_backend_task_definition() {
  aws ecs register-task-definition \
    --cli-input-json "file://infra/aws/task-definition.json" \
    --region "${AWS_REGION}" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text
}

register_frontend_task_definition() {
  local api_url="$1"
  local tmp_file
  tmp_file="$(mktemp)"
  trap 'rm -f "${tmp_file}"' RETURN
  jq --arg api_url "${api_url}" '
    .containerDefinitions[0].environment =
      ((.containerDefinitions[0].environment // [])
       | map(if .name == "API_BASE_URL" then .value = $api_url else . end))
  ' infra/aws/frontend-task-definition.json > "${tmp_file}"

  aws ecs register-task-definition \
    --cli-input-json "file://${tmp_file}" \
    --region "${AWS_REGION}" \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text

  rm -f "${tmp_file}"
  trap - RETURN
}

ensure_service() {
  local service_name="$1"
  local task_definition="$2"
  local target_group_arn="$3"
  local container_name="$4"
  local container_port="$5"
  local security_group="$6"

  if [[ "$(service_exists "${service_name}")" == "0" ]]; then
    printf 'Creating ECS service: %s\n' "${service_name}" >&2
    aws ecs create-service \
      --cluster "${ECS_CLUSTER}" \
      --service-name "${service_name}" \
      --task-definition "${task_definition}" \
      --desired-count 1 \
      --launch-type FARGATE \
      --network-configuration "awsvpcConfiguration={subnets=[${SUBNET_1},${SUBNET_2}],securityGroups=[${security_group}],assignPublicIp=ENABLED}" \
      --load-balancers "targetGroupArn=${target_group_arn},containerName=${container_name},containerPort=${container_port}" \
      --region "${AWS_REGION}" >/dev/null
  else
    printf 'Updating ECS service: %s\n' "${service_name}" >&2
    aws ecs update-service \
      --cluster "${ECS_CLUSTER}" \
      --service "${service_name}" \
      --task-definition "${task_definition}" \
      --desired-count 1 \
      --region "${AWS_REGION}" >/dev/null
  fi
}

ensure_log_group "/ecs/regression-mlops-e2e-api"
ensure_log_group "/ecs/regression-mlops-e2e-frontend"

BACKEND_TG_ARN="$(ensure_target_group "${BACKEND_TG_NAME}" "${CONTAINER_PORT}" "/health")"
FRONTEND_TG_ARN_NEW="$(ensure_target_group "${FRONTEND_TG_NAME}" "${FRONTEND_CONTAINER_PORT}" "/_stcore/health")"

BACKEND_ALB_ARN="$(ensure_load_balancer "${BACKEND_ALB_NAME}" "${ALB_SG_ID}")"
FRONTEND_ALB_ARN_NEW="$(ensure_load_balancer "${FRONTEND_ALB_NAME}" "${FRONTEND_ALB_SG_ID}")"

ensure_listener "${BACKEND_ALB_ARN}" "${BACKEND_TG_ARN}"
ensure_listener "${FRONTEND_ALB_ARN_NEW}" "${FRONTEND_TG_ARN_NEW}"

BACKEND_ALB_DNS="$(load_balancer_dns "${BACKEND_ALB_ARN}")"
FRONTEND_ALB_DNS_NEW="$(load_balancer_dns "${FRONTEND_ALB_ARN_NEW}")"

BACKEND_TASK_DEF_ARN="$(register_backend_task_definition)"
FRONTEND_TASK_DEF_ARN="$(register_frontend_task_definition "http://${BACKEND_ALB_DNS}")"

ensure_service \
  "${ECS_SERVICE}" \
  "${BACKEND_TASK_DEF_ARN}" \
  "${BACKEND_TG_ARN}" \
  "${CONTAINER_NAME}" \
  "${CONTAINER_PORT}" \
  "${ECS_SG_ID}"

ensure_service \
  "${FRONTEND_SERVICE}" \
  "${FRONTEND_TASK_DEF_ARN}" \
  "${FRONTEND_TG_ARN_NEW}" \
  "${FRONTEND_CONTAINER_NAME}" \
  "${FRONTEND_CONTAINER_PORT}" \
  "${FRONTEND_ECS_SG_ID}"

aws ecs wait services-stable \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" "${FRONTEND_SERVICE}" \
  --region "${AWS_REGION}"

cat > deploy.runtime.env <<EOF
# Generated by scripts/aws_recreate_demo_infra.sh.
# This file is intentionally gitignored because recreated ARNs and DNS names change.
export ALB_ARN=${BACKEND_ALB_ARN}
export TG_ARN=${BACKEND_TG_ARN}
export ALB_DNS=${BACKEND_ALB_DNS}
export FRONTEND_ALB_ARN=${FRONTEND_ALB_ARN_NEW}
export FRONTEND_TG_ARN=${FRONTEND_TG_ARN_NEW}
export FRONTEND_ALB_DNS=${FRONTEND_ALB_DNS_NEW}
export FRONTEND_SERVICE=${FRONTEND_SERVICE}
EOF

printf '\nDemo infrastructure is ready.\n'
printf 'Frontend: http://%s\n' "${FRONTEND_ALB_DNS_NEW}"
printf 'API docs: http://%s/docs\n' "${BACKEND_ALB_DNS}"
printf 'Health: http://%s/health\n' "${BACKEND_ALB_DNS}"
