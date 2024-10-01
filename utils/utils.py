import time
import dspy
import os
from typing import List, Dict, Any
import datetime
import json
import pickle


def set_up_dspy(
    openai_key_path: str,
    model_name: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    use_cache: bool = True,
):
    """
    Set up the DSPY environment with the specified model.
    
    Parameters:
        openai_key_path (str): The path to the OpenAI key file.
        model_name (str): The name of the model to use.
        max_tokens (int): The maximum number of tokens to use when generating responses.
        use_cache (bool): Whether to use the cache.
    """
    with open(openai_key_path, 'r') as file:
        openai_api_key = file.read().strip()
    os.environ["OPENAI_API_KEY"] = openai_api_key
    model = dspy.LM(model=f'openai/{model_name}', model_type='chat', max_tokens=max_tokens, cache=use_cache)
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
        return '_'.join(name.lower().split()) + '_' + current_time
    return '_'.join(name.lower().split()[-5:]) + '_' + current_time

