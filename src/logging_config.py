from collections.abc import Callable
from functools import wraps
import sys
from time import perf_counter
from typing import ParamSpec, TypeVar

from loguru import logger

from src.config import paths

P = ParamSpec("P")
R = TypeVar("R")

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
    "{name}:{function}:{line} | {message}"
)


def setup_logger(log_file: str = "pipeline.log", level: str = "INFO") -> None:
    """Configures Loguru for the whole ML pipeline."""
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()

    logger.add(
        sink=paths.logs_dir / log_file,
        level=level,
        format=LOG_FORMAT,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        sink=sys.stderr,
        level=level,
        format=LOG_FORMAT,
        filter=lambda record: not record["extra"].get("file_only", False),
        colorize=True,
    )


def log_step(func: Callable[P, R]) -> Callable[P, R]:
    """Logs the start, finish, duration, and failure of a pipeline step."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        step_name = func.__name__
        start_time = perf_counter()

        logger.info("Starting step: {}", step_name)

        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("Step failed: {}", step_name)
            raise

        elapsed_time = perf_counter() - start_time
        logger.info("Finished step: {} in {:.2f}s", step_name, elapsed_time)
        return result

    return wrapper
