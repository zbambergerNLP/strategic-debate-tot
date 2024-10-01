
import typing
import dspy

# Tree of thoughts imports
from tree_of_thoughts import TreeOfThoughts, TreeOfThoughtsParameters
from monte_carlo_tree_of_thoughts import MonteCarloTreeOfThought, MonteCarloTreeOfThoughtParameters

from abstractions.tree.tree import State, Node, Tree, create_conversation_state
from abstractions.generator.generator import (
    ResponseParameters,
    SingleTurnResponseBranchingSignature, 
    MultiTurnResponseBranchingSignature,
)
from abstractions.evaluator.evaluator import (
    SingleTurnScoreSingature, 
    MultiTurnScoreSignature,
    SingleTurnVoteSignature, 
    MultiTurnVoteSignature,
)

# Iterative-drafting imports
from abstractions.tree.iterative_drafting_tree import (
    DraftState, 
    DraftTree,
    MCTSDraftNode,
    MCTSDraftTree,
)
from abstractions.generator.iterative_drafting_generator import (
    SingleTurnDraftBranchingSignature,
      MultiTurnDraftBranchingSignature,
)
from abstractions.evaluator.iterative_drafting_evaluator import (
    SingleTurnScoreWithDraftsSignature, 
    MultiTurnScoreWithDraftsSignature,
    SingleTurnVoteWithDraftsSignature, 
    MultiTurnVoteWithDraftsSignature,
)
from utils import constants
from utils.utils import set_up_dspy
from utils.flags import parser
from utils.search_parameters import TreeOfThoughtsParameters, MonteCarloTreeOfThoughtParameters


class IterativeDraftingTreeOfThoughts(TreeOfThoughts):

    def _initialize_reasoning_modules(self):
        # Initialize generator modules
        self.single_turn_first_step_generator = self._initialize_module(signature=SingleTurnResponseBranchingSignature)
        self.single_turn_subsequent_step_generator = self._initialize_module(signature=SingleTurnDraftBranchingSignature)
        self.multi_turn_first_step_generator = self._initialize_module(signature=MultiTurnResponseBranchingSignature)
        self.multi_turn_subsequent_step_generator = self._initialize_module(signature=MultiTurnDraftBranchingSignature)
        # Initialize evaluator modules
        if self.evaluation_strategy == constants.EvaluationStrategy.SCORE.value:
            self.single_turn_first_step_evaluator = self._initialize_module(signature=SingleTurnScoreSingature)
            self.single_turn_subsequent_step_evaluator = self._initialize_module(signature=SingleTurnScoreWithDraftsSignature)
            self.multi_turn_first_step_evaluator = self._initialize_module(signature=MultiTurnScoreSignature)
            self.multi_turn_subsequent_step_evaluator = self._initialize_module(signature=MultiTurnScoreWithDraftsSignature)
        else:       # Use voting
            self.single_turn_first_step_evaluator = self._initialize_module(signature=SingleTurnVoteSignature)
            self.single_turn_subsequent_step_evaluator = self._initialize_module(signature=SingleTurnVoteWithDraftsSignature)
            self.multi_turn_first_step_evaluator = self._initialize_module(signature=MultiTurnVoteSignature)
            self.multi_turn_subsequent_step_evaluator = self._initialize_module(signature=MultiTurnVoteWithDraftsSignature)
    
    def _initialize_tree(
        self,
        state: State,
        depth: int,
    ) -> Tree:
        """Initialize a draft tree with the given state.
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            depth (int): The maximum depth of the tree.
        
        Returns:
            Tree: The initialized tree
        """
        # Ignore the depth passed in, as that not relevant for this reasoning type.
        return DraftTree(state=state)

    def _get_evaluator(self, state: State) -> dspy.Module:
        """Return an evaluator module based on the number of previous drafts and whether there is a pre-existing conversation
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
        
        Returns:
            dspy.Module: The evaluator module to use for the given state.
        """
        assert isinstance(state, DraftState), f"Invalid state type: {type(state)}. Must be of type DraftState."
        if len(state.previous_drafts) < 2:
            return self.multi_turn_first_step_evaluator if state.conversation else self.single_turn_first_step_evaluator
        else:
            return self.multi_turn_subsequent_step_evaluator if state.conversation else self.single_turn_subsequent_step_evaluator
    
    def _get_generator(self, state: State) -> dspy.Module:
        """Return a generator module based on the number of previous drafts and whether there is a pre-existing conversation.
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
        
        Returns:
            dspy.Module: The generator module to use for the given state.
        """
        assert isinstance(state, DraftState), f"Invalid state type: {type(state)}. Must be of type DraftState."
        if len(state.previous_drafts) == 0:
            return self.multi_turn_first_step_generator if state.conversation else self.single_turn_first_step_generator
        else:
            return self.multi_turn_subsequent_step_generator if state.conversation else self.single_turn_subsequent_step_generator
    
    def first_step(
        self, 
        tree: Tree, 
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: ResponseParameters,
    ) -> typing.List[Node]:
        """No-op for the first step in the tree of thoughts. Simply return a frontier with the root node
        
        Parameters:
            tree (Tree): The tree of thoughts to reason with.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the tree of thoughts.
            response_parameters (ResponseParameters): The parameters for generating the response.
        
        Returns:
            typing.List[Node]: The frontier of the tree of thoughts after the first step. In this case, it is simply the root node.
        """
        return [tree.root]

    def final_step(self, tree: Tree, tot_parameters: TreeOfThoughtsParameters, response_parameters: ResponseParameters) -> str:
        """Return the best draft from the (leaf nodes of a) tree of thoughts
        
        Parameters:
            tree (Tree): The tree of thoughts to reason with.
            tot_parameters (TreeOfThoughtsParameters): The parameters for the tree of thoughts.
            response_parameters (ResponseParameters): The parameters for generating the response.
        
        Returns:
            str: The best draft from the tree of thoughts, which serves as the final response.
        """
        # Select the node in the frontier with the highest score
        frontier = sorted(tree.layers[-1], key=lambda child_node: child_node.score, reverse=True)
        self.logger.info(f"Using the draft from node #{frontier[0].index} as the final response.")
        best_node: Node = frontier[0]
        # We assume that the most recent draft is better than drafts from previous iterations. 
        # We therefore return the best draft among the leaf nodes.
        reasoning_steps: typing.List[str] = best_node.state.reasoning_steps
        return reasoning_steps[-1]


class IterativeDraftingMCTSTreeOfThoughts(IterativeDraftingTreeOfThoughts, MonteCarloTreeOfThought):

    def _initialize_tree(self, state: State, depth: int) -> Tree:
        """Initialize a draft tree with the given state
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            depth (int): The maximum depth of the tree.

        Returns:
            Tree: The initialized tree
        """
        return MCTSDraftTree(state=state)

    def __init__(
        self, 
        use_chain_of_thought: bool = False, 
        node_selection_strategy: str = constants.NodeSelectionStrategy.GREEDY.value, 
        evaluation_strategy: str = constants.EvaluationStrategy.SCORE.value, 
    ) -> None:
        """Initialize an iterative drafting tree of thoughts with Monte Carlo Tree Search
        
        Parameters:
            use_chain_of_thought (bool, optional): Whether to use chain of thoughts. Defaults to False.
            node_selection_strategy (str, optional): The strategy for selecting nodes to expand (or prune) in the tree. 
                Defaults to constants.NodeSelectionStrategy.GREEDY.value.
            evaluation_strategy (str, optional): The strategy for evaluating the quality of a response. 
                Defaults to constants.EvaluationStrategy.SCORE.value.
        """
        IterativeDraftingTreeOfThoughts.__init__(self, use_chain_of_thought, node_selection_strategy, evaluation_strategy)
    
    def forward(
        self, 
        state: State, 
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters, 
        do_visualize_tree: bool = False, 
        do_save_tree: bool = False, 
        verbose: bool = False,
    ) -> str:
        """Forward pass through the tree of thoughts with Monte Carlo Tree Search. Involves performing multiple MCTS iterations
        
        Parameters:
            state (State): The state of intermediate reasoning/thinking in a given task (towards generating a response).
            mcts_parameters (MonteCarloTreeOfThoughtParameters): The parameters for the Monte Carlo Tree of Thoughts.
            response_parameters (ResponseParameters): The parameters for generating the response.
            do_visualize_tree (bool, optional): Whether to visualize the tree. Defaults to False.
            do_save_tree (bool, optional): Whether to save the tree. Defaults to False.
            verbose (bool, optional): Whether to print verbose logs. Defaults to False.
        
        Returns:
            str: The final response generated by the tree of thoughts.
        """
        return MonteCarloTreeOfThought.forward(
            self, 
            state=state, 
            mcts_parameters=mcts_parameters,
            do_visualize_tree=do_visualize_tree, 
            do_save_tree=do_save_tree,
            verbose=verbose,
            response_parameters=response_parameters, 
        )
    
    def generate_response(
        self, 
        tree: Tree, 
        mcts_parameters: MonteCarloTreeOfThoughtParameters, 
        response_parameters: ResponseParameters,
    ) -> str:
        """Generate the final response from the tree of thoughts
        
        Parameters:
            tree (Tree): The tree of thoughts to reason with.
            mcts_parameters (MonteCarloTreeOfThoughtParameters): The parameters for the Monte Carlo Tree of Thoughts.
            response_parameters (ResponseParameters): The parameters for generating the response.
        
        Returns:
            str: The final response generated by the tree of thoughts.
        """
        del response_parameters
        root_index = tree.root.index
        root_node = tree.nodes[root_index]
        best_node: MCTSDraftNode = None
        max_score = 0
        for child_index in root_node.children_ids:
            child_node = tree.nodes[child_index]
            score = child_node.score / child_node.visits
            if score > max_score:
                max_score = score
                best_node = child_node
        return best_node.state.reasoning_steps[-1]


if __name__ == "__main__":

    # Parse the command-line arguments    
    args = parser.parse_args()
    set_up_dspy(
        openai_key_path=args.openai_key_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    # Initialize the conversation state
    conversation_state = create_conversation_state(
        topic=args.topic,
        stance=args.stance,
        conversation_path=args.conversation_path,
    )

    print('Arguments:')
    for arg in vars(args):
        print(f'\t{arg}: {getattr(args, arg)}')
    print(f'Initial conversation state:\n{conversation_state}')
    
    # Initialize the tree of thoughts
    if args.search_strategy == constants.SearchAlgorithm.MONTE_CARLO_TREE_SEARCH.value:
        print('Using Monte Carlo Tree Search.')
        tree_of_thoughts = IterativeDraftingMCTSTreeOfThoughts(
            use_chain_of_thought=args.use_chain_of_thought,
            node_selection_strategy=args.node_selection_strategy,
            evaluation_strategy=args.evaluation_strategy,
        )
        response = tree_of_thoughts(
            state=conversation_state,
            mcts_parameters=MonteCarloTreeOfThoughtParameters(
                n_samples_judge=args.n_samples_judge, 
                n_samples_generation=args.n_samples_generation,
                judge_temperature=args.judge_temperature,
                generation_temperature=args.generation_temperature,
                rollout_depth=args.depth,
                monte_carlo_iterations=args.mcts_iterations,
            ),
            do_visualize_tree=args.with_visualization,
            do_save_tree=args.save_tree,
            response_parameters=ResponseParameters(
                response_length=args.response_length,
                communication_tone=args.communication_tone,
                language_style=args.language_style,
            ),
        )
    else:   # Use beam search
        print('Using Beam Search.')
        tree_of_thoughts = IterativeDraftingTreeOfThoughts(
            use_chain_of_thought=args.use_chain_of_thought,
            node_selection_strategy=args.node_selection_strategy,
            evaluation_strategy=args.evaluation_strategy,
            do_pruning=args.do_pruning,
        )
        response = tree_of_thoughts(
            state=conversation_state,
            tot_parameters=TreeOfThoughtsParameters(
                top_k=args.top_k,
                generation_temperature=args.generation_temperature,
                judge_temperature=args.judge_temperature,
                n_samples_generation=args.n_samples_generation,
                n_samples_judge=args.n_samples_judge,
                depth=args.depth,
            ),
            do_visualize_tree=args.with_visualization,
            do_save_tree=args.save_tree,
            response_parameters=ResponseParameters(
                response_length=args.response_length,
                communication_tone=args.communication_tone,
                language_style=args.language_style,
            ),
        )

    print(f'Final response:\n{response}')
    