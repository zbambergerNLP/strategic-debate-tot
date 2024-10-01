import logging
import os
import datetime


def get_current_time():
    """
    get_current_time() -> str
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_logger(folder_path:str=None):
    """
    setup_logger(folder_path=None) -> logging.Logger
    This function sets up the logger for the application.
    """
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set the root logger level to INFO
    logging.root.setLevel(logging.INFO)
    
    # Suppress lower-level logs from specific libraries
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    
    # Ensure the log directory exists
    log_directory = folder_path if folder_path else 'run_logs'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    
    # Create file handler
    file_handler = logging.FileHandler(f'{log_directory}/{get_current_time()}.log')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Define formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and add handlers
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # **Suppress LiteLLM's INFO logs**
    lite_llm_logger = logging.getLogger('LiteLLM')
    lite_llm_logger.setLevel(logging.WARNING)  # Change to WARNING or higher as needed
    
    return logger