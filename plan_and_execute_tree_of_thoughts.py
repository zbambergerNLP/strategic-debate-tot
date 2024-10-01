import dspy
import typing

# Tree-of-thoughts imports
from tree_of_thoughts import TreeOfThoughts
from monte_carlo_tree_of_thoughts import MonteCarloTreeOfThought
from abstractions.tree.tree import (
    State,
    Node, 
    Tree, 
    create_conversation_state,
)
from utils import constants
from utils.utils import set_up_dspy
from utils.flags import parser
from utils.search_parameters import (
    TreeOfThoughtsParameters, 
    MonteCarloTreeOfThoughtParameters,
)

# Plan-and-execute imports
from abstractions.tree.plan_and_execute_tree import (
    PlanAndExecuteState, 
    PlanAndExecuteTree,
    MCTSPlanAndExecuteTree,
)
from abstractions.generator.generator import ResponseParameters
from abstractions.generator.plan_and_execute_generator import (
    SingleTurnPlanningBranchingSignature, 
    SingleTurnPlanExecutionBranchingSignature,
    MultiTurnPlanningBranchingSignature, 
    MultiTurnPlanExecutionBranchingSignature,
    SingleTurnPlanAndExecuteResponseSignature,
)
from abstractions.evaluator.plan_and_execute_evaluator import (
    SingleTurnScoreWithPlanSignature, 
    SingleTurnScoreWithPlanExecutionSignature,
    MultiTurnScoreWithPlanSignature, 
    MultiTurnScoreWithPlanExecutionSignature,
)

class PlanAndExecuteTreeOfThoughts(TreeOfThoughts):
    
    def _initialize_reasoning_modules(self):
        # Initialize generator modules
        self.single_turn_first_step_generator = self._initialize_module(signature=SingleTurnPlanningBranchingSignature)
        self.single_turn_subsequent_step_generator = self._initialize_module(signature=SingleTurnPlanExecutionBranchingSignature)
        self.multi_turn_first_step_generator = self._initialize_module(signature=MultiTurnPlanningBranchingSignature)
        self.multi_turn_subsequent_step_generator = self._initialize_module(signature=MultiTurnPlanExecutionBranchingSignature)
        # Initialize evaluator modules
        if self.evaluation_strategy == constants.EvaluationStrategy.SCORE.value:
            self.single_turn_first_step_evaluator = self._initialize_module(signature=SingleTurnScoreWithPlanSignature)
            self.single_turn_subsequent_step_evaluator = self._initialize_module(signature=SingleTurnScoreWithPlanExecutionSignature)
            self.multi_turn_first_step_evaluator = self._initialize_module(signature=MultiTurnScoreWithPlanSignature)
            self.multi_turn_subsequent_step_evaluator = self._initialize_module(signature=MultiTurnScoreWithPlanExecutionSignature)
        else:
            raise NotImplementedError("Voting is not yet implemented for the plan-and-execute reasoning type.")
        
        self.plan_and_execute_response_generator = self._initialize_module(signature=SingleTurnPlanAndExecuteResponseSignature)
    
    def _initialize_tree(
        self,
        state: State,
        depth: int
    ) -> Tree:
        """Initialize the tree of thoughts. 

        The children of the root node correspond with possible plans to execute.
        Descendents of these children represent executions of individual steps in the plan.
        """
        return PlanAndExecuteTree(state=state, max_depth=depth)
    

    def _get_evaluator(self, state: State) -> dspy.Module:
        """Return the appropriate evaluator module based on the state."""
        assert isinstance(state, PlanAndExecuteState), f"Invalid state type: {type(state)}. Must be of type PlanAndExecuteState."
        if state.claims_so_far:
            return self.multi_turn_subsequent_step_evaluator if state.conversation else self.single_turn_subsequent_step_evaluator
        else:
            return self.multi_turn_first_step_evaluator if state.conversation else self.single_turn_first_step_evaluator

    def _get_generator(self, state: State) -> dspy.Module:
        """Return the appropriate generator module based on the state."""
        assert isinstance(state, PlanAndExecuteState), f"Invalid state type: {type(state)}. Must be of type PlanAndExecuteState."
        if state.plan:
            return self.multi_turn_subsequent_step_generator if state.conversation else self.single_turn_subsequent_step_generator
        else:
            return self.multi_turn_first_step_generator if state.conversation else self.single_turn_first_step_generator
        
    def first_step(
        self, 
        tree: Tree, 
        tot_parameters: TreeOfThoughtsParameters,
        response_parameters: ResponseParameters,
    ) -> typing.List[Node]:
        """Generate a plan for the task."""
        self.logger.info(f"Generating a plan with {tot_parameters.depth} steps.")
        return TreeOfThoughts.step(
            self,
            tree=tree,
            frontier=[tree.root],
            tot_parameters=tot_parameters,
            response_parameters=response_parameters,
        )
    
    def final_step(self, tree: Tree, tot_parameters: TreeOfThoughtsParameters, response_parameters: ResponseParameters) -> str:
        """Return the final response from the tree of thoughts."""
        # We select the best leaf node, which is the node with the highest scoring plan + execution.
        frontier = sorted(tree.layers[-1], key=lambda child_node: child_node.score, reverse=True)
        self.logger.info(f"Using the draft from node #{frontier[0].index} as the final response.")
        best_node: Node = frontier[0]
        # We generate a final response that is based on the plan and it's execution, and which matches
        # the desired response length.
        response = self.plan_and_execute_response_generator(
            topic=best_node.state.topic,
            stance=best_node.state.stance,
            plan=best_node.state.plan,
            plan_execution=best_node.state.claims_so_far,
            response_parameters=response_parameters, 
        )
        return response.response


class PlanAndExecuteMCTSTreeOfThoughts(PlanAndExecuteTreeOfThoughts, MonteCarloTreeOfThought):

    def _initialize_tree(self, state: State, depth: int) -> Tree:
        return MCTSPlanAndExecuteTree(state=state, max_depth=depth)

    def __init__(
        self, 
        use_chain_of_thought: bool = False, 
        node_selection_strategy: str = constants.NodeSelectionStrategy.GREEDY.value, 
        evaluation_strategy: str = constants.EvaluationStrategy.SCORE.value,
    ):
        PlanAndExecuteTreeOfThoughts.__init__(self, use_chain_of_thought, node_selection_strategy, evaluation_strategy)
    
    def forward(
        self, 
        state: State, 
        mcts_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters, 
        do_visualize_tree: bool = False, 
        do_save_tree: bool = False, 
        verbose: bool = False,
    ) -> str:
        return MonteCarloTreeOfThought.forward(
            self, 
            state=state, 
            mcts_parameters=mcts_parameters,
            do_visualize_tree=do_visualize_tree, 
            do_save_tree=do_save_tree,
            verbose=verbose,
            response_parameters=response_parameters, 
        )
    
    def rollout(
        self, 
        node: Node, 
        tree: Tree, 
        mcst_parameters: MonteCarloTreeOfThoughtParameters,
        response_parameters: ResponseParameters,
    ) -> float:
        """
        Perform a rollout from the given node. The rollout consists of generating a sequence of reasoning steps
        (i.e, coming up with a plan and executing it) until we reach the maximum depth of the rollout or have finished
        executing all steps in the plan.
        """
        new_node = node
        # Rollout should not result in performing more execution steps than there are steps in the plan.
        rollout_depth: int = min(mcst_parameters.rollout_depth, len(node.state.plan) - len(node.state.claims_so_far))
        for layer in range(rollout_depth):
            # Since we are performing a rollout, we only want a **single** chain of responses. The nodes we create
            # here are **not** added to the tree.
            rollout_parameters = mcst_parameters.model_copy(update={"n_samples_generation": 1})
            response, _ = self.get_response(
                state=new_node.state, 
                mcts_parameters=rollout_parameters,
                response_parameters=response_parameters,
            )
            response: typing.List[str]  # `response` is a list of one element.
            new_node = tree.create_child_node(index=-layer, state=new_node.state, output=response[0])
        node.score, node.reasoning = self.get_score(
            state=new_node.state, mcts_parameters=mcst_parameters,
        )
        node.visits = 1
        return node.score

    def generate_response(self, tree: Tree, mcts_parameters: MonteCarloTreeOfThoughtParameters, response_parameters: ResponseParameters) -> str:
        """
        We select the best leaf node, which is the node with the highest scoring plan + execution.
        From there, we generate a final response that is based on the plan and it's execution, and which matches
        the desired response length.
        """
        tot_parameters = TreeOfThoughtsParameters(
            top_k=1,
            n_samples_judge=mcts_parameters.n_samples_judge,
            n_samples_generation=1,
            judge_temperature=mcts_parameters.judge_temperature,
            generation_temperature=mcts_parameters.generation_temperature,
            depth=1,
        )
        return PlanAndExecuteTreeOfThoughts.final_step(
            self, 
            tree=tree, 
            tot_parameters=tot_parameters,
            response_parameters=response_parameters,
        )


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
        tree_of_thoughts = PlanAndExecuteMCTSTreeOfThoughts(
            use_chain_of_thought=args.use_chain_of_thought,
            node_selection_strategy=args.node_selection_strategy,
            evaluation_strategy=args.evaluation_strategy,
        )
        response = tree_of_thoughts(
            state=conversation_state,
            mcts_parameters=MonteCarloTreeOfThoughtParameters(
                monte_carlo_iterations=args.mcts_iterations,
                rollout_depth=args.depth,
                generation_temperature=args.generation_temperature,
                judge_temperature=args.judge_temperature,
                n_samples_generation=args.n_samples_generation,
                n_samples_judge=args.n_samples_judge,
            ),
            do_visualize_tree=args.with_visualization,
            do_save_tree=args.save_tree,
            response_parameters=ResponseParameters(response_length=args.response_length, communication_tone=args.communication_tone, language_style=args.language_style),
        )
    else:
        tree_of_thoughts = PlanAndExecuteTreeOfThoughts(
            use_chain_of_thought=args.use_chain_of_thought,
            node_selection_strategy=args.node_selection_strategy,
            evaluation_strategy=args.evaluation_strategy,
            do_pruning=args.do_pruning,
        )
        response = tree_of_thoughts(
            state=conversation_state,
            tot_parameters=TreeOfThoughtsParameters(
                depth=args.depth,
                top_k=args.top_k,
                generation_temperature=args.generation_temperature,
                judge_temperature=args.judge_temperature,
                n_samples_generation=args.n_samples_generation,
                n_samples_judge=args.n_samples_judge,
            ),
            do_visualize_tree=args.with_visualization,
            do_save_tree=args.save_tree,
            response_parameters=ResponseParameters(response_length=args.response_length, communication_tone=args.communication_tone, language_style=args.language_style),
        )
    print(f'Final response:\n{response}')
