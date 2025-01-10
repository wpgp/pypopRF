# src/pypoprf/utils/logger.py
import logging
import sys
from typing import Optional, TextIO
from pathlib import Path
from datetime import datetime
from colorama import Fore, Style


class PopRFLogger:
    """
    Centralized logging system for pypopRF.

    This class provides logging functionality throughout the pypopRF package,
    supporting both file output and custom output streams.
    """

    def __init__(self,
                 log_file: Optional[str] = None,
                 log_level: int = logging.INFO,
                 output_stream: Optional[TextIO] = sys.stdout):
        """
        Initialize logger with optional file output.

        Args:
            log_file: Optional path to log file
            log_level: Logging level (default: INFO)
            output_stream: Optional custom output stream (default: sys.stdout)
        """
        self.logger = logging.getLogger('pypopRF')
        self.logger.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if output_stream:
            console_handler = logging.StreamHandler(output_stream)
            color_formatter = self._get_color_formatter()
            console_handler.setFormatter(color_formatter)
            self.logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.console_handler = console_handler if output_stream else None
        self.file_handler = file_handler if log_file else None

    def _get_color_formatter(self) -> logging.Formatter:
        class ColorFormatter(logging.Formatter):
            LEVEL_COLORS = {
                logging.DEBUG: Fore.BLUE,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
            }

            def format(self, record: logging.LogRecord) -> str:
                color = self.LEVEL_COLORS.get(record.levelno, Fore.WHITE)
                formatted_message = super().format(record)
                return f"{color}{formatted_message}{Style.RESET_ALL}"

        return ColorFormatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    def set_level(self, level: str) -> None:
        """
        Set logging level.

        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        log_level = level_map.get(level.upper(), logging.INFO)
        self.logger.setLevel(log_level)

    def set_output_stream(self, stream: TextIO) -> None:
        """
        Update or set output stream.

        Args:
            stream: New output stream
        """
        if self.console_handler:
            self.logger.removeHandler(self.console_handler)

        console_handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.console_handler = console_handler

    def set_log_file(self, log_file: str) -> None:
        """
        Add or update file handler.

        Args:
            log_file: Path to log file
        """
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.file_handler = file_handler

    def info(self, message: str) -> None:
        """Log info level message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning level message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error level message."""
        self.logger.error(message)

    def debug(self, message: str) -> None:
        """Log debug level message."""
        self.logger.debug(message)

    def critical(self, message: str) -> None:
        """Log critical level message."""
        self.logger.critical(message)


logger = PopRFLogger()


def get_logger(log_file: Optional[str] = None,
               output_stream: Optional[TextIO] = None) -> PopRFLogger:
    """
    Get or create logger instance.

    Args:
        log_file: Optional path to log file
        output_stream: Optional custom output stream

    Returns:
        PopRFLogger instance
    """
    global logger
    if log_file:
        logger.set_log_file(log_file)
    if output_stream:
        logger.set_output_stream(output_stream)
    return logger