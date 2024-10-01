"""Implementation of the Tree-Of-Thoughts module."""
import abc
import typing
import dspy
import numpy as np

# Local imports
import utils.visualize_tree as visualize_tree
from abstractions.tree.tree import State, Node, Tree
from abstractions.generator.generator import ResponseParameters
# Utilities
from utils import constants
from utils.utils import generate_name
from utils import logger_config
import utils.visualize_tree as visualize_tree
from utils.search_parameters import TreeOfThoughtsParameters

###############################
# To-Dos for Tree-Of-Thoughts #
###############################

# Generic to-dos for the 'tree_of_thoughts.py' file:

# TODO[P2](Zach): Move the code from the tree data structure files (i.e., files in `abstractions/tree/`) into
#  the main files within the root directory (e.g., move contents of `abstractions/tree/devils_advocate_tree.py` 
#   into 'devils_advocate_tree_of_thoughts.py').
# TODO[P2](Zach): Move 'visualize_tree' into the 'tree' module. Allow subclasses of 'Tree' to implement their own visualization.
#   or inherit a basic version from 'Tree'.
# TODO[P3](Zach): Creater wrappers to make logging intermediate reasoning (from generators and evaluators) more legible.
# TODO[P3](Zach): Create a function in 'tree' to format the reasoning for a node's score. 
#   This should differ based on the reasoning type in which the score is assigned.
# TODO[P3](Zach): Use DSPy's MultiChainComparison to determine judge scores based on the scores of an ensemble of judges.

# Specific to-dos for the 'devils_advocate_tree_of_thoughts.py' file:

# TODO[P1](Zach): Add a flag to allow the user to choose whether or not to use self reflection to generate the final response
#   (as opposed to returning the best child of the root node after minimax).
# TODO[P1](Zach): Self-Reflection should rely on the reasoning of judges to generate responses.
# TODO[P1](Zach): Modify the UCT formula so that when it is the adversary's turn, we choose the child node with the minimum
#   score (since the adversary is trying to minimize the score that we are trying to maximize).


# Create a metaclass that combines the ABCMeta and the type of the dspy.Module class.
# This allows us to naturally extend the dspy.Module class while enforcing subclasses to implement specific methods.
class CombinedMeta(abc.ABCMeta, type(dspy.Module)):
    pass

class TreeOfThoughts(dspy.Module, abc.ABC, metaclass=CombinedMeta):
    """
    Utilize the Tree-Of-Thoughts framework to plan and reason before generating a response.
    """
    
    @abc.abstractmethod
    def _get_evaluator(self, state: State) -> dspy.Module:
        """Retrieve the appropriate evaluator based on the state."""
        pass
    
    @abc.abstractmethod
    def _get_generator(self, state: State) -> dspy.Module:
        """Retrieve the appropriate generator based on the state."""
        pass
    
    @abc.abstractmethod
    def _initialize_tree(
        self,
        state: State,
        depth: int,
    ) -> Tree:
        pass

    @abc.abstractmethod
    def first_step(
        self, 
        tree: Tree, 
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: str,
    ) -> typing.List[Node]:
        pass

    @abc.abstractmethod
    def final_step(self, tree: Tree, tot_parameters: TreeOfThoughtsParameters, response_parameters: ResponseParameters) -> str:
        pass

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
    def _initialize_reasoning_modules(self) -> None:
        """Initialize DSPy modules for reasoning (i.e., generators and evaluators)."""
        pass
    
    def __init__(
        self,
        use_chain_of_thought: bool = False,
        node_selection_strategy: str = constants.NodeSelectionStrategy.GREEDY.value,
        evaluation_strategy: str = constants.EvaluationStrategy.SCORE.value,
        do_pruning: bool = True,
    ) -> None:
        """
        Initialize the Tree-Of-Thoughts module.
        
        Parameters:
            use_chain_of_thought (bool): Whether to use a chain of thought model for generating intermediate thoughts/reasoning.
            node_selection_strategy (str): The strategy to use for selecting the nodes in the tree. 
                Must be one of ["greedy", "sample"].
            evaluation_strategy (str): The strategy to use for evaluating the quality of a response.
                Must be one of ["score", "vote"].
            do_pruning (bool): Whether to prune the tree of thoughts at intermediate reasoning steps.
        """
        super().__init__()
        self.use_chain_of_thoughts = use_chain_of_thought
        self.node_selection_strategy = node_selection_strategy
        self.evaluation_strategy = evaluation_strategy
        self.do_pruning = do_pruning
        self.logger = logger_config.setup_logger(folder_path='./logs')
        self._initialize_reasoning_modules()
    
    def forward(
        self,
        state: State,
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: ResponseParameters,
        do_visualize_tree: bool = False,
        do_save_tree: bool = False,
        verbose: bool = False,
    ) -> str:
        """
        Utilize the Tree-Of-Thoughts framework to plan and reason before generating a response.
        
        Args:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module. Defines the number of steps 
                (depth) to take in the tree, the number of samples to use when judging the quality of intermediate thoughts, 
                the number of samples to generate for each reasoning step, and the temperatures to use when judging and 
                generating thoughts.
            do_visualize_tree (bool): Whether to visualize the tree of thoughts.
            do_save_tree (bool): Whether to save the tree of thoughts to a file.
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).
        
        Returns:
            str: The response generated by the Tree-Of-Thoughts module.
        """
        assert tot_parameters.depth > 0, "The number of steps (depth) for Tree-Of-Thoughts must be greater than 0."
        # Initialize the tree
        tree: Tree = self._initialize_tree(state=state, depth=tot_parameters.depth)
        # Conditionally perform a "first-step" of reasoning (e.g., write and select a plan for witing an argument).
        # This can also be a no-op, which simply returns the initial frontier.
        frontier: typing.List[Node] = self.first_step(tree, tot_parameters, response_parameters)
        # Perform subsequent reasoning steps (e.g., executing a step in a plan for writing an argument).
        for layer_index in range(tot_parameters.depth):
            self.logger.info(f"Performing reasoning step {layer_index + 1}/{tot_parameters.depth}.")
            new_frontier: typing.List[Node] = self.step(tree, frontier, tot_parameters, response_parameters)
            frontier = new_frontier
        if not self.do_pruning:
            # If the tree was created without pruning, then none of the nodes in the tree are scored. 
            # We therefore need to score the leaf nodes in the tree (i.e., the full sequence of thoughts/reasoning steps).
            self.evaluate_thoughts(new_nodes=frontier, tree=tree, tot_parameters=tot_parameters)
        response = self.final_step(tree, tot_parameters=tot_parameters, response_parameters=response_parameters)
        if verbose:
            tree.log_tree(logger=self.logger)
        if do_save_tree:
            tree.save_to_file(file_name="test_tree.json")
        if do_visualize_tree:
            visualize_tree.draw_graph(
                graph=visualize_tree.tree_to_graph(tree),
                name=f"{generate_name(tree.root.state.topic)}",
            )
        return response
        
    def step(
        self, 
        tree: Tree, 
        frontier: typing.List[int], 
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: ResponseParameters
    ) -> typing.List[int]:
        """
        Perform a step by generating thoughts, evaluating them, and selecting the best thoughts for the next layer of the tree.
        
        Perform a step in the Tree-Of-Thoughts framework to generate a collection of thoughts for the next layer of the tree.

        Given this collection of thoughts, evaluate them and select the top-k thoughts to continue to the next layer of the tree.
        This step expands multiple nodes in the the deepest layer of the tree (i.e., the frontier).

        Parameters:
            tree (Tree): The tree of thoughts.
            frontier (List[int]): The list of node indices in the frontier.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module. Defines the number of steps 
                (depth) to take in the tree, the number of samples to use when judging the quality of intermediate thoughts, 
                the number of samples to generate for each reasoning step, and the temperatures to use when judging and 
                generating thoughts.
            response_parameters (ResponseParameters): Parameters which define the nature of the response to generate (e.g., 
                length, tone, and language style).
        
        Returns:
            List[int]: The list of node indices in the next frontier.
        """
        # Generation:
        # Generate new thoughts for each node in the frontier (corresponds with creating a new layer in the tree).
        new_nodes: typing.List[Node] = self.generate_thoughts(tree, frontier, tot_parameters, response_parameters)   

        # Evaluation:
        # If we are pruning, we evaluate the nodes to determine which (top-k scoring nodes) to keep and which to prune.
        # Otherwise, we simply return the generated nodes.
        if self.do_pruning:
            judged_nodes: typing.List[Node] = self.evaluate_thoughts(new_nodes, tree, tot_parameters)
        else:
            return new_nodes
        
        # Selection:
        # If we are pruning, we select the top-k nodes to continue to the next layer of the tree, and prune the rest.
        frontier: typing.List[int] = self.select_thoughts(judged_nodes, tot_parameters)
        return frontier
    
    def generate_thoughts(
        self, 
        tree: Tree, 
        frontier: typing.List[Node], 
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: ResponseParameters,
    ) -> typing.List[Node]:
        """
        Expand each node in the frontier in order to obtain the nodes in the next layer of the tree.
        
        Parameters:
            tree (Tree): The tree of thoughts.
            frontier (List[Node]): The list of nodes in the frontier (i.e., the most recent/last layer of the tree, which consists
                of leaf nodes).
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module. Defines the number of steps 
                (depth) to take in the tree, the number of samples to use when judging the quality of intermediate thoughts, 
                the number of samples to generate for each reasoning step, and the temperatures to use when judging and 
                generating thoughts.
            response_length (str): The desired length of the response. For example, "a few sentences" or "a paragraph".
        
        Returns:
            List[TreeOfThoughtsNode]: The nodes in the next layer of the tree. Note that these nodes do not yet include
                the scores or reasoning behind the scores (which are added during the judging phase).
        """
        n_samples_generation = tot_parameters.n_samples_generation
        generation_temperature = tot_parameters.generation_temperature
        new_nodes = []
        generator: dspy.Module = self._get_generator(state=frontier[0].state)
        
        # Expand each node in the frontier
        for node in frontier:
            completions = generator(
                **node.state.state_to_generator_input(response_parameters=response_parameters),
                config=dict(n=n_samples_generation, temperature=generation_temperature),
            ).completions
            outputs: typing.List[str] = [response for response in completions.response]
            reasonings: typing.List[str] = completions.reasoning if self.use_chain_of_thoughts else ["N/A"] * len(outputs)
            if len(outputs) != n_samples_generation:
                self.logger.warning(f"The number of outputs generated must be equal to `n_samples_generation`. Expected: {n_samples_generation}, Actual: {len(outputs)}.")
            if len(reasonings) != n_samples_generation:
                self.logger.warning(f"The number of reasonings generated must be equal to `n_samples_generation`. Expected: {n_samples_generation}, Actual: {len(reasonings)}.")
            # Create a (child) node in the tree for each response from a parent node
            for output, reasoning in zip(outputs, reasonings):
                child_node = tree.create_child_node(index=len(tree.nodes), state=node.state, output=output)
                tree.add_child_node(parent_node=node, child_node=child_node, response=output, expansion_reasoning=reasoning)
                new_nodes.append(child_node)
        
        # Add a layer to the tree. We then select which nodes in this layer to keep, and which to prune
        tree.add_layer(new_nodes)    
        self.logger.info(f"\tGenerated new nodes: [{', '.join([str(node.index) for node in new_nodes])}]")
        return new_nodes
    
    def evaluate_thoughts(
        self,
        new_nodes: typing.List[Node],
        tree: Tree,
        tot_parameters: TreeOfThoughtsParameters,
    ) -> typing.List[Node]:
        if self.evaluation_strategy == constants.EvaluationStrategy.SCORE.value:    # Use "scoring" module for judges
            judged_nodes: typing.List[Node] = []
            for node in new_nodes:
                judged_nodes.append(self.score_thought(node=node, tree=tree, tot_parameters=tot_parameters))
        else:                                                                       # Use "voting" module for judges
            judged_nodes: typing.List[Node] = self.vote_on_thoughts(nodes=new_nodes, tree=tree, tot_parameters=tot_parameters)
        self.logger.info(f"\tNew node scores: {[node.score for node in judged_nodes]}")
        return judged_nodes

    def vote_on_thoughts(
        self,
        nodes: typing.List[Node],
        tree: Tree,
        tot_parameters: TreeOfThoughtsParameters
    ) -> typing.List[Node]:
        """
        Aggregate votes on which of the drafts is most persuasive, and assign scores to nodes based on the votes.
        
        Parameters:
            nodes (List[TreeOfThoughtsNode]): The nodes in the tree of thoughts.
            tree (Tree): The tree of thoughts.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module. Defines the number of steps 
                (depth) to take in the tree, the number of samples to use when judging the quality of intermediate thoughts, 
                the number of samples to generate for each reasoning step, and the temperatures to use when judging and 
                generating thoughts.
        
        Returns:
            List[Node]: The nodes in the tree of thoughts with the scores assigned (based on the votes).
                These nodes are also augmented to include the reasoning behind the votes (if applicable).
        """
        n_samples_judge = tot_parameters.n_samples_judge
        judge_temperature = tot_parameters.judge_temperature
        judge: dspy.Module = self._get_evaluator(state=nodes[0].state)
        completions = judge(
            **tree.create_voting_input_from_most_recent_layer(),
            config=dict(n=n_samples_judge, temperature=judge_temperature),
        ).completions
        one_hot_votes: typing.List[int] = completions.index   # A vector of length `n_samples_judge` with the index of the vote
        # Reasoning is a list of length `n_samples_judge`, which contains the reasoning for each vote (in an ensemble of voting 
        # judges). Therefore, the reasoning for the resulting scores of each node is the reasoning of the combined votes.
        reasoning = "\n\n".join(completions.reasoning) if self.use_chain_of_thoughts else "N/A"
        counts = [0 for _ in range(len(nodes))]
        for vote in one_hot_votes:
            counts[vote] += 1
        self.logger.info(f"\tNodes [{', '.join([str(node.index) for node in nodes])}] received vote counts: {counts}")
        # Scores are the fraction of votes for each node
        scores = [count / sum(counts) for count in counts]
        # Assign the scores to the nodes in the tree
        for node, score in zip(nodes, scores):
            tree.nodes[node.index].score = score
            tree.nodes[node.index].reasoning = reasoning
        return nodes

    def score_thought(
        self, 
        node: Node, 
        tree: Tree,
        tot_parameters: TreeOfThoughtsParameters,
    ) -> Node:
        """
        Judge the drafts generated for a node in the tree of thoughts.
        
        Parameters:
            node (Node): The node in the tree of thoughts.
            tree (Tree): The tree of thoughts.
            n_samples_judge (int): The number of samples to use when judging the drafts.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module.
                Includes the number of samples to use when judging the quality of intermediate thoughts, and the temperature
                to use when judging the thoughts.
        
        Returns:
            TreeOfThoughtsNode: The node in the tree of thoughts with the score assigned (and reasoning, if applicable).
        """
        n_samples_judge = tot_parameters.n_samples_judge
        judge_temperature = tot_parameters.judge_temperature
        judge: dspy.Module = self._get_evaluator(state=node.state)
        completions = judge(
            **node.state.state_to_evaluator_input(),
            config=dict(n=n_samples_judge, temperature=judge_temperature),
        ).completions
        score = np.mean(completions.score).round(3)  # Average the scores from the ensemble of (`n_samples_judge`) judges
        self.logger.info(f"\t\tScore for node {node.index} is {score}")      
        judge_reasoning = "\n\n".join(completions.reasoning) if self.use_chain_of_thoughts else "N/A"
        # Modify the node in the tree
        tree.nodes[node.index].score = score
        tree.nodes[node.index].reasoning = judge_reasoning
        return tree.nodes[node.index]
    

    def select_thoughts(
        self, 
        judged_nodes: typing.List[Node], 
        tot_parameters: TreeOfThoughtsParameters,
    ) -> typing.List[Node]:
        """
        Select the top-k nodes to continue to the next layer of the tree.

        Parameters:
            tree (Tree): The tree of thoughts.
            judged_nodes (List[Node]): The nodes in the tree of thoughts with the scores assigned.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the Tree-Of-Thoughts module. Defines the number of steps 
                (depth) to take in the tree, the number of samples to use when judging the quality of intermediate thoughts, 
                the number of samples to generate for each reasoning step, and the temperatures to use when judging and 
                generating thoughts.

        Returns:
            List[int]: The list of node indices in the next frontier.
        """
        if self.node_selection_strategy == constants.NodeSelectionStrategy.GREEDY.value:    # Select nodes greedily
            frontier: typing.List[Node] = sorted(
                judged_nodes, key=lambda child_node: child_node.score, reverse=True
            )[:tot_parameters.top_k]
        else:                                                                               # Select nodes with weighted sampling
            values: typing.List[float] = [child_node.score for child_node in judged_nodes]
            frontier: typing.List[Node] = np.random.choice(
                judged_nodes, size=tot_parameters.top_k, p=(np.array(values) / sum(values))
            ).tolist()
        # Sort by index once we've selected the top-k nodes. 
        # This is to ensure that the order of the nodes in the frontier is consistent.
        frontier: typing.List[Node] = sorted(frontier, key=lambda child_node: child_node.index)
        self.logger.info(f"\tSelected nodes: [{', '.join([str(node.index) for node in frontier])}]")

        # Prune the nodes that were not selected
        non_selected_nodes: typing.List[Node] = [node for node in judged_nodes if node not in frontier]
        self.logger.info(f"\tPruned nodes: [{', '.join([str(node.index) for node in non_selected_nodes])}]")
        for node in non_selected_nodes:
            node.is_pruned = True
        return frontier
