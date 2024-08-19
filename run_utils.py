"""Functionality for running the tree-of-thoughts script ('tree_of_thoughts.py')."""
from tree import State

def create_conversation_state(
    topic: str, 
    stance: str, 
    conversation_path: str,
) -> State:
    """
    Create a conversation state for the strategic debater.
    
    Parameters:
        topic (str): The topic of the debate.
        stance (str): The stance of the debator.
        conversation_path (str): The path to the conversation file.
    
    Returns:
        State: The conversation state for the strategic debater.
    """
    # Initialize Tree-of-Thoughts given a conversation in the provided file
    if conversation_path:
        with open(conversation_path, 'r') as file:
            conversation = file.readlines()
        conversation_state = State(
            topic=topic,
            stance=stance,
            conversation=conversation,
        )
    # Initialize Tree-of-Thoughts given an empty conversation
    else:  
        conversation_state = State(
            topic=topic,
            stance=stance,
            conversation=[],
        )
    return conversation_state
