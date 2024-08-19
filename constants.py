from enum import Enum

class Model(Enum):
    """
    The model to use for generating responses.

    GPT_3_5_TURBO: The GPT-3.5-turbo model.
    GPT_4O: The GPT-4o model.
    GPT_4O_MINI: The GPT-4o-mini model (cheaper than both GPT-4o and GPT-3.5-turbo, but performs only slightly worse than GPT-4o).
    """
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    GPT_4O = 'gpt-4o'
    GPT_4O_MINI = 'gpt-4o-mini'

class TreeOfThoughtsType(Enum):
    """
    The type of search algorithm to use in the Tree of Thoughts module.
    
    BFS: The breadth-first search algorithm. This algorithm is used to explore the tree of thoughts in a breadth-first manner.
        This involves generating child nodes for each node in the frontier of the tree, then scoring the child nodes (via
        scoring each node individually, or via voting), and then selecting the best child node for further expansion.
    """
    BFS = 'breadth_first_search'  # Breadth-first search, introduced in the original Tree of Thoughts paper.


class NodeSelectionStrategy(Enum):
    """
    The strategy for evaluating which nodes to select for further expansion in the (vanilla) Tree of Thoughts module.

    GREEDY: The greedy strategy. This strategy involves selecting the node with the highest score for expansion. In case of a tie,
        the nodes with smaller indices are selected.
    
    SAMPLE: The sample strategy. This strategy involves selecting a node for expansion at random, with weights proportional to the
        scores of the nodes. This strategy is useful for encouraging exploration in the tree of thoughts, especially when the scores
        of the nodes are close to each other.
    """
    GREEDY = 'greedy'
    SAMPLE = 'sample'


class EvaluationStrategy(Enum):
    """
    The strategy for evaluating the quality of a response.
    
    SCORE: The score-based strategy. This strategy involves evaluating the quality of a response based on a score assigned to the
        response by a model. The score is typically a float between 0 and 1, where 1 indicates a high-quality response and 0 indicates
        a low-quality response. Each response is evaluated in isolation. The top 'k' responses are considered high-quality responses,
        and the rest are pruned.
    
    VOTE: The vote-based strategy. This strategy involves evaluating the quality of a response based on votes from multiple agents.
        Each agent votes on the quality of the response, and the response is evaluated based on the majority vote (i.e., the responses
        with the top 'k' votes are considered high-quality responses, and the rest are pruned).
    """
    SCORE = 'score'
    VOTE = 'vote'

class Colors(Enum):
    """
    The colors to use for highlighting text in the console. 
    
    Different colors are used to distinguish between debators with different stances in a debate.

    BLUE: The color blue.
    RED: The color red.
    """
    BLUE = 'blue'
    RED = 'red'