from datetime import datetime
import logging
import os

from src.utils.settings import PathSettings


def get_custom_logger(
        name: str,
        log_dir: str = None,
        level: int = logging.INFO,
    ) -> logging.Logger:
    if log_dir is None:
        log_dir = PathSettings.LOG_DIR

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}.log")
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger 
