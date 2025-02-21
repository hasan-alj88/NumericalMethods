import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import glob
from pathlib import Path

# Define log folder relative to script location
log_folder = Path(__file__).parent / "logs"
# Create logs directory if it doesn't exist
log_folder.mkdir(exist_ok=True)


class SequentialRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename, maxBytes, backupCount=0, encoding=None, delay=False):
        # Ensure we're using the log folder
        self.log_dir = log_folder

        # Find the next available number for the initial log file
        next_num = self._get_next_number()
        initial_filename = self.log_dir / f"log_{next_num}.log"

        super().__init__(
            filename=str(initial_filename),
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )

    def _get_next_number(self):
        existing_logs = list(self.log_dir.glob("log_*.log"))
        if not existing_logs:
            return 1

        numbers = []
        for log in existing_logs:
            try:
                num = int(log.stem.split('_')[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue

        return max(numbers) + 1 if numbers else 1

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        next_num = self._get_next_number()
        new_file = self.log_dir / f"log_{next_num}.log"

        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, str(new_file))

        if not self.delay:
            self.stream = self._open()


LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(funcName)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': '__main__.SequentialRotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'placeholder.log',  # This will be overridden by the handler's __init__
            'maxBytes': 3 * 1024 * 1024,  # 3MB
            'backupCount': 0
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': [
                'console',
                'file'
            ],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}


def get_logger(name):
    logging.SequentialRotatingFileHandler = SequentialRotatingFileHandler

    # Create logs directory if it doesn't exist
    log_folder.mkdir(exist_ok=True)

    if not logging.getLogger().handlers:
        # Update the class path in the config to match the current module
        LOGGING_CONFIG['handlers']['file']['class'] = f'{__name__}.SequentialRotatingFileHandler'

        # Apply the configuration
        logging.config.dictConfig(LOGGING_CONFIG)

    return logging.getLogger(name)