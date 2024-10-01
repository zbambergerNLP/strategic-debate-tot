import abc
import dspy
import math
import typing
import numpy as np

from tree_of_thoughts import CombinedMeta
from abstractions.generator.generator import ResponseParameters
from abstractions.tree.tree import (
    State,
    MonteCarloNode,
    Tree,
)
from utils.search_parameters import MonteCarloTreeOfThoughtParameters
from utils import logger_config
from utils import visualize_tree


class MonteCarloTreeOfThought(dspy.Module, abc.ABC, metaclass=CombinedMeta):
    """
    A Tree-Of-Thought module that uses the monte carlo algorithm to plan the best response to a given conversation.
    """

    def _initialize_module(self, signature: dspy.Signature) -> dspy.Module:
        """
        Initialize a function based on the specified signature.

        Parameters:
            signature (dspy.Signature): The signature of the function to initialize.
        
        Returns:
            dspy.Module: The initialized function.
        """
        if self.use_chain_of_thoughts:
            return dspy.TypedChainOfThought(signature=signature)
        else:
            return dspy.TypedPredictor(signature=signature)
    
    @abc.abstractmethod
    def _get_evaluator(self, state: State) -> dspy.Module:
        """Retrieve the appropriate evaluator module based on the state.
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
        
        Returns:
            dspy.Module: The evaluator module.
        """
        pass
    
    @abc.abstractmethod
    def _get_generator(self, state: State) -> dspy.Module:
        """Retrieve the appropriate generator module based on the state.
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
        
        Returns:
            dspy.Module: The generator module.
        """
        pass
    
    @abc.abstractmethod
    def _initialize_tree(
        self,
        state: State,
        depth: int,
    ) -> Tree:
        """Initialize a tree of thoughts with nodes that represent states of intermediate planning/reasoning.
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            depth (int): The depth of the tree, which corresponds with the number of reasoning steps to execute. Advancing to a 
                new layer in the tree corresponds to taking a new reasoning step.
        """
        pass
    
    def __init__(
        self, 
        use_chain_of_thoughts: bool = True,
    ) -> None:
        """
        Initializes the Monte Carlo Tree of Thoughts module.

        Perform Monte Carlo Tree Search to plan/reason over the best response to a given conversation/query.
        Uses evaluation strategy `score` since MCTS uses rollout with branching factor 1.
        Uses UCB for node selection strategy, since that is the primary (most studied) exploration/exploitation method for MCTS.
        
        Parameters:
            use_chain_of_thoughts (bool): Whether to use the chain of thoughts or not.
        """
        super().__init__()
        self.use_chain_of_thoughts = use_chain_of_thoughts
        self.logger = logger_config.setup_logger(folder_path='./logs')
      
    
    def forward(
        self, 
        state: State, 
        response_parameters: ResponseParameters,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        do_visualize_tree: bool = False,
        do_save_tree: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Utilize Tree-Of-Thought to plan the best response to a given conversation.

        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought (e.g., generation_temperature, rollout_depth, etc...).
            do_visualize_tree (bool): Whether to visualize the tree.
            do_save_tree (bool): Whether to save the tree.
            verbose (bool): Whether to print verbose output.
        
        Returns:
            str: The best response to the conversation. Derived from reasoning steps explored by the MCTS algorithm.
        """
        tree = self._initialize_tree(state=state, depth=mcts_parameters.rollout_depth)
        for step in range(mcts_parameters.monte_carlo_iterations):
            self.logger.info(f'Performing Monte Carlo step {step+1} of {mcts_parameters.monte_carlo_iterations}')
            self.monte_carlo_step(tree.root, tree, step+1, mcts_parameters, response_parameters)
        response: str = self.generate_response(tree, mcts_parameters, response_parameters)
        if verbose:
            tree.log_tree(logger=self.logger)
        if do_save_tree:
            tree.save_to_file()
        if do_visualize_tree:
            visualize_tree.draw_graph(graph=visualize_tree.tree_to_graph(tree))
        return response
    
    def monte_carlo_step(
        self, 
        node: MonteCarloNode,
        tree: Tree,
        total_visits: int,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> float:
        """
        Recursively computes the scores for each node in the tree using the minimax algorithm.
        
        Parameters:
            node (MonteCarloNode): The node to compute the score for.
            tree (Tree): the tree.
            total_visits (int): The total number of visits to all nodes in the tree.
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought.
            rresponse_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).

        Returns:
            float: The score of the node.
        """
        self.logger.info(f'Performing Monte Carlo step for node {node.index}')
        # Base case: leaf node
        if not node.children_ids:
            if node.visits == 0:
                # Perform a rollout to estimate the score without creating new nodes in the tree.
                value: float = self.rollout(node, tree, mcts_parameters, response_parameters)
                self.logger.info(f'\tRollout score for node {node.index} is {value}')
                return value
            else:
                # Given a node with a score (obtained from a rollout), we choose to expand this node.
                self.expand_node(node, tree, mcts_parameters, response_parameters)
                child_node: MonteCarloNode = self.choose_best_child(node, tree, total_visits)
                value: float = self.monte_carlo_step(child_node, tree, total_visits, mcts_parameters, response_parameters)
                self.logger.info(f'\tExpanded node {node.index}, which recieved score {value}')
        else:
            child_node: MonteCarloNode = self.choose_best_child(node, tree, total_visits)
            value: float = self.monte_carlo_step(child_node, tree, total_visits, mcts_parameters, response_parameters)
            self.logger.info(f'\tChose child node {node.index} with score {value}')
        node.visits += 1
        node.score = value if node.score == 0 else (node.score + value).round(decimals=2)
        self.logger.info(f'\tUpdated score for node {node.index} (with {node.visits} visits) to {node.score}')
        return value

    def rollout(
        self, 
        node: MonteCarloNode,
        tree: Tree,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> float:
        """
        Estimates the score of a node using a rollout strategy.

        Parameters:
            node (MonteCarloNode): The node to compute the score to.
            tree (Tree): The search tree.
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought.
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).

        Returns:
            float: The score of the node.
        """
        new_node = node
        for layer in range(mcts_parameters.rollout_depth):
            # Since we are performing a rollout, we only want a **single** chain of responses. The nodes we create
            # here are **not** added to the tree.
            rollout_parameters: MonteCarloTreeOfThoughtParameters = mcts_parameters.model_copy(update={'n_samples_generation': 1})
            response, _ = self.get_response(new_node.state, rollout_parameters, response_parameters)
            response: typing.List[str]  # `response` is a list of one element.
            new_node = tree.create_child_node(index=-layer, state=new_node.state, output=response[0])
        node.score, node.reasoning = self.get_score(new_node.state, mcts_parameters)
        node.visits = 1
        return node.score

    def expand_node(
        self, 
        node: MonteCarloNode,
        tree: Tree,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> typing.List[MonteCarloNode]:
        """
        Expands a node in the search tree by generating possible augmentations (actions) that lead to child nodes.
        
        Parameters:
            node (Node): The node to expand.
            tree (Tree): The search tree.
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought.
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).
            
        Returns:
            List[MonteCarloNode]: The child nodes of the expanded node.
        """
        responses, reasonings = self.get_response(node.state, mcts_parameters, response_parameters)
        child_nodes = []
        for response, reasoning in zip(responses, reasonings):
            child_node: MonteCarloNode = tree.create_child_node(index=len(tree.nodes), state=node.state, output=response)
            tree.add_child_node(parent_node=node, child_node=child_node, response=response, expansion_reasoning=reasoning)
            child_nodes.append(child_node)
        tree.add_layer(child_nodes)
        return child_nodes

    def get_response(
        self, 
        state: State,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> typing.Tuple[typing.List[str], typing.List[str]]:
        """
        Generates a response to a conversation using the model.

        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought.
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).
        Returns:
            Tuple[List[str], List[str]]: A 2-tuple that consists of a list of candidate responses and a list of reasonings 
                behind those responses.
        """
        n_samples_generation: int = mcts_parameters.n_samples_generation
        generation_temperature: float = mcts_parameters.generation_temperature
        generator: dspy.Module = self._get_generator(state=state)
        completions = generator(
            **state.state_to_generator_input(response_parameters=response_parameters),
            config=dict(n=n_samples_generation, temperature=generation_temperature),
        ).completions
        output: typing.List[str] = [response for response in completions.response]
        return (output, completions.reasoning)
    
    def choose_best_child(
        self, 
        node: MonteCarloNode, 
        tree: Tree,
        total_visits: int,
    ) -> MonteCarloNode:
        """
        Chooses the best child node of a parent node based on the scores of the children.

        Parameters:
            node (Node): The parent node.
            tree (Tree): The search tree.
            total_visits (int): The total number of visits to all nodes in the tree.
        
        Returns: 
            Node: The best child node using UCB1.
        """
        assert len(node.children_ids) > 0, f"Node {node.index} has no children to choose from."
        children_nodes: typing.List[MonteCarloNode] = [tree.nodes[child_index] for child_index in node.children_ids]
        child_node_scores: typing.List[float] = []
        for node in children_nodes:
            if node.visits == 0:    # If the node is unvisited, set the score to infinity to ensure it is selected
                current_score = float('inf')
            else:                   # Use the UCB1 formula to select the best node
                current_score: float = round(
                    node.score / node.visits + 2 * math.sqrt(math.log(total_visits) / node.visits), 
                    ndigits=3,      # Round to 3 decimal places (avoid common python floating point errors)
                )
            child_node_scores.append(current_score)
        best_node: MonteCarloNode = children_nodes[np.argmax(child_node_scores)]
        return best_node
    
    @staticmethod
    def _wrap_judge_reasoning(reasoning: typing.List[str]) -> str:
        """
        Wrap the reasoning for the votes in an informative string.
        
        Parameters:
            reasoning (List[str]): The reasoning for the votes.
        
        Returns:
            str: The wrapped reasoning for the votes. The reasoning is separated by newlines.
        """
        result = ""
        for i, reason in enumerate(reasoning):
            result += f"Judge #{i + 1} reasoning:\n{reason}\n\n"
        return result.strip()
    
    def get_score(self, state: State, mcts_parameters: MonteCarloTreeOfThoughtParameters) -> typing.Tuple[float, str]:
        """
        Gets the score of a conversation using an evaluator.

        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over
                reasoning steps with MCTS Tree-Of-Thought.

        Returns:
            Tuple[float, str]: The score of the conversation and the reasoning behind the score.
        """
        n_samples_judge: int = mcts_parameters.n_samples_judge
        judge_temperature: float = mcts_parameters.judge_temperature
        judge: dspy.Module = self._get_evaluator(state=state)
        completions = judge(
            **state.state_to_evaluator_input(),
            config=dict(n=n_samples_judge, temperature=judge_temperature),
        ).completions
        score: float = np.mean(completions.score).round(decimals=2)     # Compute average score across `n_samples_judge` judges
        reasoning: str = self._wrap_judge_reasoning(reasoning=completions.reasoning) if self.use_chain_of_thoughts else "N/A"
        return score, reasoning


    @abc.abstractmethod
    def generate_response(
        self, 
        tree: Tree,
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> str:
        """
        Generate an response based on a tree of thoughts (obtained via MCTS).

        Parameters:
            tree (Tree): The search tree for the best response.
            mcts_parameters (MonteCarloTreeOfThoughtParameters): Parameters which define the behavior of the search over 
                reasoning steps with MCTS Tree-Of-Thought
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).

        Returns:
            str: The optimal response to the conversation.
        """
        pass
