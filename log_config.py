import logging
import logging.config
from logging.handlers import RotatingFileHandler
import os
import glob


class SequentialRotatingFileHandler(RotatingFileHandler):
    def __init__(self, filename, maxBytes, backupCount=0, encoding=None, delay=False):
        # Find the next available number for the initial log file
        next_num = self._get_next_number()
        initial_filename = f"log_{next_num}.log"

        super().__init__(
            filename=initial_filename,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay
        )

    @staticmethod
    def _get_next_number():
        existing_logs = glob.glob("log_*.log")
        if not existing_logs:
            return 1

        numbers = []
        for log in existing_logs:
            try:
                num = int(log.split('_')[1].split('.')[0])
                numbers.append(num)
            except (IndexError, ValueError):
                continue

        return max(numbers) + 1 if numbers else 1

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        next_num = self._get_next_number()
        new_file = f"log_{next_num}.log"

        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, new_file)

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
            'level': 'INFO',
            'formatter': 'detailed',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': '__main__.SequentialRotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'log_1.log',  # This will be overridden by the handler's __init__
            'maxBytes': 3 * 1024 * 1024,  # 3MB
            'backupCount': 0
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': [
                # 'console',
                'file'
            ],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}


def get_logger(name):
    logging.SequentialRotatingFileHandler = SequentialRotatingFileHandler
    if not logging.getLogger().handlers:
        # Update the class path in the config to match the current module
        LOGGING_CONFIG['handlers']['file']['class'] = f'{__name__}.SequentialRotatingFileHandler'

        # Apply the configuration
        logging.config.dictConfig(LOGGING_CONFIG)

    return logging.getLogger(name)