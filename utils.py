import time
import dspy
import logging
import os
import datetime
import json

def set_up_dspy(
    openai_key: str = None,
    openai_key_path: str = "openai_key.txt", 
    model_name: str = "gpt-4o",
    max_tokens: int = 1_000,
):
    """
    Set up the DSPY environment with the specified model.
    
    Parameters:
        openai_key_path (str): The path to the OpenAI key file.
        model_name (str): The name of the model to use.
    """
    if openai_key:
        openai_api_key = openai_key
    else:
        with open(openai_key_path, 'r') as file:
            openai_api_key = file.read().strip()
    
    os.environ['OPENAI_API_KEY'] = openai_api_key

    model = dspy.OpenAI(
        model=model_name,
        api_key=openai_api_key, 
        max_tokens=max_tokens,
    )
    dspy.settings.configure(lm=model)

def generate_name(name: str) -> str:
    """
    Generates a unique name for a file.

    Parameters:
        name (str): The base name for the file.

    Returns:
        str: The unique name for the file.
    """
    
    current_time = time.strftime("%m-%d-%H-%M-%S")
    
    if len(name.split()) <= 5:
        return '_'.join(name.lower().split())+'_'+current_time
    return '_'.join(name.lower().split()[-5:])+'_'+current_time

def get_current_time():
    """
    get_current_time() -> str
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def setup_logger(folder_path: str = None):
    """
    Set up the logger for the program.

    Parameters:
        folder_path (str, optional): The path to the folder where the logs will be stored. Defaults to None.
    """
    # return logging.getLogger(__name__)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.root.setLevel(logging.INFO)
    logging.getLogger('httpcore').setLevel(logging.CRITICAL)
    logging.getLogger('requests').setLevel(logging.CRITICAL)
    logging.getLogger('httpx').setLevel(logging.CRITICAL)
    
    if not os.path.exists('run_logs'):
        os.makedirs('run_logs')
    
    file_handler = logging.FileHandler(f'run_logs/{get_current_time()}.log')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
