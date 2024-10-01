# Response Strategies
from enum import Enum


class ReasoningType(Enum):
    """
    The type of reasoning used for Tree-of-Thoughts modules.

    ITERATIVE_DRAFTING: This type of reasoning involves generating argumentative responses that improve on previous drafts.
    PLAN_AND_EXECUTE: This type of reasoning involves creating an initial plan for the argument (e.g., claims to make). 
        At the i'th layer of the tree, the agent executes the i'th step of the plan.
    """
    ITERATIVE_DRAFTING = 'iterative_drafting'
    PLAN_AND_EXECUTE = 'plan_and_execute'

class ReasoningComponent(Enum):
    GENERATOR = 'generator'
    EVALUATOR = 'evaluator'

class ReasoningPhase(Enum):
    FIRST_STEP = 'first_step'
    SUBSEQUENT_STEP = 'subsequent_step'
    FINAL_STEP = 'final_step'

class Model(Enum):
    """
    The model to use for generating responses.

    GPT_4O: The GPT-4o model.
    GPT_4O_MINI: The GPT-4o-mini model (cheaper than both GPT-4o and GPT-3.5-turbo, but performs only slightly worse than GPT-4o).
    """
    GPT_4O = 'gpt-4o'
    GPT_4O_MINI = 'gpt-4o-mini'


class SearchAlgorithm(Enum):
    """
    The type of search algorithm to use in the Tree of Thoughts module.
    
    MINIMAX: The minimax algorithm. This algorithm assumes there are strictly 2 opposing players in an exchange, 
        and that they take turns making moves. The algorithm is used to determine the optimal move for a player in a 
        zero-sum game. This search type is only available in the RESPONSE granularity, since that granularity corresponds
        with messages between two opposing players (while DRAFT corresponds with a single player).
        
    MONTE_CARLO_TREE_SEARCH: The Monte Carlo Tree Search algorithm. This algorithm is used to determine the optimal move
        for a player in a game by simulating a large number of random games and selecting the move that leads to the
        highest win rate. 
    """
    MONTE_CARLO_TREE_SEARCH = 'monte_carlo'
    BEAM_SEARCH = 'beam_search'  # Known as Breadth-first search (BFS) within the original Tree of Thoughts paper.


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
        response by a model. The score is typically a float between 0 and 1, where 1 indicates a high-quality response and 0 
        indicates a low-quality response. Each response is evaluated in isolation. The top 'k' responses are considered 
        high-quality responses, and the rest are pruned.
    
    VOTE: The vote-based strategy. This strategy involves evaluating the quality of a response based on votes from multiple agents.
        Each agent votes on the quality of the response, and the response is evaluated based on the majority vote (i.e., the 
        responses with the top 'k' votes are considered high-quality responses, and the rest are pruned).
    """
    SCORE = 'score'
    VOTE = 'vote'

class Color(Enum):
    """
    The colors to use for highlighting text in the console. 
    
    Different colors are used to distinguish between debators with different stances in a debate.

    BLUE: The color blue.
    RED: The color red.
    """
    BLUE = 'blue'
    RED = 'red'
