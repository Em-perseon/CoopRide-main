"""
Log Module for Simulator
Provides logging functionality with different levels: info, debug, warning
"""

import os
from datetime import datetime


class Logger:
    """
    Logger class for logging messages with different levels.
    """
    
    # Global log mode configuration
    # Available modes: 'info', 'warning', 'info_daily', 'debug'
    # - 'info': prints info and warning
    # - 'warning': prints info and warning
    # - 'info_daily': prints info, warning, info_daily (logs to file only with indentation)
    # - 'debug': prints debug, info_daily, info, and warning (all levels)
    # LOG_MODE = 'info'
    LOG_MODE = 'info_daily'
    # LOG_MODE = 'debug'
    
    # Log file path with timestamp suffix
    @staticmethod
    def _get_log_file_path():
        """Get log file path with timestamp suffix."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        log_file = os.path.join(log_dir, f'simulator_{timestamp}.log')
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        return log_file
    
    LOG_FILE = None  # Will be set dynamically
    
    @staticmethod
    def _get_timestamp():
        """Get current timestamp in formatted string."""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def _should_log(level):
        """
        Check if the given log level should be printed based on current LOG_MODE.
        
        Args:
            level (str): Log level ('debug', 'info', 'warning', 'info_daily')
        
        Returns:
            bool: True if should log, False otherwise
        """
        if Logger.LOG_MODE == 'debug':
            return True
        elif Logger.LOG_MODE == 'info_daily':
            return level in ['info_daily', 'info', 'warning']
        elif Logger.LOG_MODE == 'info' or Logger.LOG_MODE == 'warning':
            return level in ['info', 'warning']
        return False
    
    @staticmethod
    def _write_to_log(level, message):
        """
        Write log message to log file.
        
        Args:
            level (str): Log level
            message (str): Log message
        """
        # Initialize LOG_FILE if not set
        if Logger.LOG_FILE is None:
            Logger.LOG_FILE = Logger._get_log_file_path()
        
        timestamp = Logger._get_timestamp()
        # Add indentation for info_daily level to distinguish from daily logs
        prefix = "    " if level == 'info_daily' else ""
        log_entry = f"[{timestamp}] [{level.upper()}] {prefix}{message}\n"
        
        # Append to log file
        with open(Logger.LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    @staticmethod
    def debug(message):
        """
        Log debug level message.
        
        Args:
            message (str): Debug message to log
        
        Usage:
            from simulator.log import Logger
            Logger.debug("This is a debug message")
        """
        if Logger._should_log('debug'):
            Logger._write_to_log('debug', message)
    
    @staticmethod
    def info(message):
        """
        Log info level message. Also prints to console.
        
        Args:
            message (str): Info message to log
        
        Usage:
            from simulator.log import Logger
            Logger.info("This is an info message")
        """
        if Logger._should_log('info'):
            Logger._write_to_log('info', message)
            print(f"[INFO] {message}")
    
    @staticmethod
    def warning(message):
        """
        Log warning level message. Also prints to console.
        
        Args:
            message (str): Warning message to log
        
        Usage:
            from simulator.log import Logger
            Logger.warning("This is a warning message")
        """
        if Logger._should_log('warning'):
            Logger._write_to_log('warning', message)
            print(f"[WARNING] {message}")
    
    @staticmethod
    def info_daily(message):
        """
        Log info_daily level message. Writes to log file only (no console output).
        Used for step-by-step simulation details.
        
        Args:
            message (str): Info daily message to log
        
        Usage:
            from simulator.log import Logger
            Logger.info_daily("Step details: 5 matched, 10 waiting")
        """
        if Logger._should_log('info_daily'):
            Logger._write_to_log('info_daily', message)
            print(f"[INFO_DAILY] {message}")
    
    @staticmethod
    def set_log_mode(mode):
        """
        Set the global log mode.
        
        Args:
            mode (str): Log mode ('info', 'warning', 'info_daily', 'debug')
        
        Usage:
            from simulator.log import Logger
            Logger.set_log_mode('debug')
        """
        if mode in ['info', 'warning', 'info_daily', 'debug']:
            Logger.LOG_MODE = mode
        else:
            raise ValueError(f"Invalid log mode: {mode}. Must be 'info', 'warning', 'info_daily', or 'debug'")
    
    @staticmethod
    def get_log_mode():
        """
        Get the current log mode.
        
        Returns:
            str: Current log mode
        """
        return Logger.LOG_MODE
    
    @staticmethod
    def clear_log():
        """
        Clear the log file content.
        
        Usage:
            from simulator.log import Logger
            Logger.clear_log()
        """
        # Initialize LOG_FILE if not set
        if Logger.LOG_FILE is None:
            Logger.LOG_FILE = Logger._get_log_file_path()
        with open(Logger.LOG_FILE, 'w', encoding='utf-8') as f:
            f.write('')


# Create a singleton instance for convenience
logger = Logger()