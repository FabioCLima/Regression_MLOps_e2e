import time
import uuid

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start = time.perf_counter()

        with logger.contextualize(request_id=rid):
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "method={} path={} status={} latency_ms={}",
                request.method,
                request.url.path,
                response.status_code,
                latency_ms,
            )

        response.headers["X-Request-ID"] = rid
        return response
