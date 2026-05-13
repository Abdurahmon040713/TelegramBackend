import json
import logging
import time
from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "request_id": getattr(record, "request_id", None),
            },
            ensure_ascii=False,
        )


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    logging.getLogger("telethon.crypto.libssl").setLevel(logging.ERROR)
    logging.getLogger("telethon.network.mtprotosender").setLevel(logging.WARNING)


def mask_phone(phone: str) -> str:
    """998901234567 → 998*****567  (keeps PII out of log aggregators)."""
    if len(phone) < 6:
        return "***"
    return phone[:3] + "*" * (len(phone) - 6) + phone[-3:]


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())[:8]
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        logging.getLogger("access").info(
            "%s %s %s %.0fms",
            request.method,
            request.url.path,
            response.status_code,
            ms,
            extra={"request_id": request_id},
        )
        return response
