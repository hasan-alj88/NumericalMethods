import functools
import traceback
from contextlib import contextmanager

from utils.log_config import get_logger


@contextmanager
def IgnoreException(exception_type, logger=get_logger(__name__)): # noqa
    try:
        yield
    except exception_type:
        logger.warning(f'Ignoring exception [{exception_type.__name__}]')
        pass


def ignore_exception(exception_type, logger=None):
    if logger is None:
        logger = get_logger(__name__)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logger.warning(f'Ignoring exception [{exception_type.__name__}\n{e}]')
                return None
        return wrapper

    return decorator

@contextmanager
def LogAndReraise(logger=get_logger(__name__), message:str = ''): # noqa
    try:
        yield
    except Exception as e:
        logger.error(f'Exception [{e.__class__.__name__}] raised: {message}\n{e}')
        logger.error(traceback.format_exc())
        raise

def log_and_reraise(logger=None):
    if logger is None:
        logger = get_logger(__name__)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f'Exception [{e.__class__.__name__}] raised: {e}')
                logger.error(traceback.format_exc())
                raise
        return wrapper

    return decorator
