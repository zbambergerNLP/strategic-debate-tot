"""Implementation of the Tree-Of-Thoughts module, as presented in Yao et al. (2023).

See utils/flags.py for the command-line arguments that can be passed to this module.

Example usage:

```
python tree_of_thoughts.py \
    --openai_key_path "openai_key.txt" \
    --topic "The government should enforce regulation on AI technology." \
    --stance "PRO" \
    --conversation_path "data/conversations/ai_regulation/example_1.txt" \
    --evaluation_strategy score \
    --node_selection_strategy greedy \
    --generation_temperature 0.7 \
    --judge_temperature 0.7 \
    --n_samples_generation 3 \
    --n_samples_judge 5 \
    --top_k 2 \
    --depth 2 \
    --use_chain_of_thought \
    --max_tokens 1_000 \
    --response_length "a few sentences" \
    --model_name gpt-4o-mini \
    --openai_key_path "openai_key.txt" \
    --with_visualization \
    --save_tree
```

Explanation of the command-line arguments:
- `conversation_path`: The path to the conversation file. Each line in the file represents a message in the conversation.
- `evaluation_strategy`: The strategy to use for evaluating the drafts. Must be one of ["score", "vote"].
- `node_selection_strategy`: The strategy to use for selecting the nodes in the tree. Must be one of ["greedy", "sample"].
- `generation_temperature`: The temperature to use when generating the drafts.
- `judge_temperature`: The temperature to use when judging the drafts. A float between 0 and 1. 
- `n_samples_generation`: The number of samples to generate for each response.
- `n_samples_judge`: The number of samples to use when judging the drafts.
- `top_k`: The number of best nodes to select at each step (according to the node selection strategy).
- `depth`: The depth of the tree. The deeper the tree, the more reasoning is involved in producing candidate arguments.
- `use_chain_of_thought`: Whether to use the chain of thought.
- `max_tokens`: The maximum number of tokens to use when generating responses. Defaults to 1,000.
- `model_name`: The name of the language model to use. Defaults to "gpt-4o-mini".
- `openai_key_path`: The path to the OpenAI key file.
- `with_visualization`: Whether to visualize the tree of thought. Ommit this argument if you do not want to visualize the tree.
"""
import time
import typing
import dspy
import numpy as np

from tree import (
    DraftState,
    State,
    Tree,
    TreeOfThoughtsNode,
    initialize_tree,
    log_tree_of_thoughts,
    save_tree,
    draw_graph, 
    tree_to_graph,
)
from branching import (
    # Branching without drafts
    ToTSingleTurnResponseBranchingInput,
    ToTSingleTurnResponseBranchingSignature,
    ToTMultiTurnResponseBranchingInput,
    ToTMultiTurnResponseBranchingSignature,
    # Branching with drafts
    ToTSingleTurnDraftBranchingInput,
    ToTSingleTurnDraftBranchingSignature,
    ToTMultiTurnDraftBranchingInput,
    ToTMultiTurnDraftBranchingSignature,
)
from debate_judge import (
    # Scoring without drafts
    ToTMultiTurnJudgeInput, 
    ToTMultiTurnJudgeSignature, 
    ToTSingleTurnJudgeInput, 
    ToTSingleTurnJudgeSingature,
    # Scoring with drafts
    ToTSingleTurnJudgeWithDraftsInput,
    ToTSingleTurnJudgeWithDraftsSignature,
    ToTMultiTurnJudgeWithDraftsInput,
    ToTMultiTurnJudgeWithDraftsSignature,
    # Voting without drafts
    ToTJudgeVoteSingleTurnInput,
    ToTJudgeVoteSingleTurnSignature,
    ToTJudgeVoteMultiTurnInput,
    ToTJudgeVoteMultiTurnSignature,
    # Voting with drafts
    ToTJudgeVoteSingleTurnWithDraftsInput,
    ToTJudgeVoteSingleTurnWithDraftsSignature,
    ToTJudgeVoteMultiTurnWithDraftsInput,
    ToTJudgeVoteMultiTurnWithDraftsSignature,
)

# Utilities
import constants
from flags import parser
from utils import generate_name, set_up_dspy, setup_logger
from run_utils import create_conversation_state


class TreeOfThoughts(dspy.Module):
    """
    Utilize the Tree-Of-Thoughts framework to craft a persuasive argument on a given topic.
    """

    def _initialize_function(self, signature: dspy.Signature) -> dspy.Module:
        """
        Initialize a function based on the specified signature.

        Parameters:
            signature (dspy.Signature): The signature of the LLM utility/function to initialize (e.g., judging an argument).
        
        Returns:
            dspy.Module: The initialized function.
        """
        if self.use_chain_of_thoughts:
            return dspy.TypedChainOfThought(signature=signature)
        else:
            return dspy.TypedPredictor(signature=signature)

    def __init__(
        self,
        use_chain_of_thought: bool = False,
        node_selection_strategy: str = constants.NodeSelectionStrategy.GREEDY.value,
        evaluation_strategy: str = constants.EvaluationStrategy.SCORE.value,
    ):
        """
        Initialize the CreateCounterArguments module.
        
        Parameters:
            use_chain_of_thought (bool): Whether to use a chain of thought model for generating arguments and judging arguments.
            node_selection_strategy (str): The strategy to use for selecting the nodes in the tree. Must be one of ["greedy", "sample"].
            evaluation_strategy (str): The strategy to use for evaluating the quality of a response. Must be one of ["score", "vote"].
        """
        super().__init__()
        self.use_chain_of_thoughts = use_chain_of_thought
        self.node_selection_strategy = node_selection_strategy
        self.evaluation_strategy = evaluation_strategy
        self.logger = setup_logger(folder_path='./logs')
        
        # If the module is used with an empty conversation, the single-turn models are used. 
        # Otherwise, the multi-turn models are used.
        self.single_turn_first_draft_predictor = self._initialize_function(signature=ToTSingleTurnResponseBranchingSignature)
        self.single_turn_expansion_predictor = self._initialize_function(signature=ToTSingleTurnDraftBranchingSignature)
        self.single_turn_score_judge = self._initialize_function(signature=ToTSingleTurnJudgeSingature)
        self.single_turn_score_judge_with_drafts = self._initialize_function(signature=ToTSingleTurnJudgeWithDraftsSignature)
        self.single_turn_vote_judge = self._initialize_function(signature=ToTJudgeVoteSingleTurnSignature)
        self.single_turn_vote_judge_with_drafts = self._initialize_function(signature=ToTJudgeVoteSingleTurnWithDraftsSignature)

        self.multi_turn_first_draft_predictor = self._initialize_function(signature=ToTMultiTurnResponseBranchingSignature)
        self.multi_turn_expansion_predictor = self._initialize_function(signature=ToTMultiTurnDraftBranchingSignature)
        self.multi_turn_judge = self._initialize_function(signature=ToTMultiTurnJudgeSignature)
        self.multi_turn_score_judge_with_drafts = self._initialize_function(signature=ToTMultiTurnJudgeWithDraftsSignature)
        self.multi_turn_vote_judge = self._initialize_function(signature=ToTJudgeVoteMultiTurnSignature)
        self.multi_turn_vote_judge_with_drafts = self._initialize_function(signature=ToTJudgeVoteMultiTurnWithDraftsSignature)
    
    def forward(
        self,
        state: State,
        depth: int = 2,
        top_k: int = 2,
        generation_temperature: float = 0.7,
        judge_temperature: float = 0.7,
        n_samples_generation: int = 3,
        n_samples_judge: int = 5,
        do_visualize_tree: bool = False,
        do_save_tree: bool = False,
        response_length: str = "a few sentences",
    ) -> str:
        """
        Generates a tree structure representing the branching of arguments in a strategic debate.
        
        Args:
            state (State): The state of the debate. This includes the topic, and stance. A State can 
                also include a conversation, and previous drafts of the argument (if applicable).
            depth (int): The number of steps to take in the tree. Each step introduces another layer of arguments in the tree.
            top_k (int): The number of best nodes to select at each layer. The top k nodes are selected based on their scores 
                (obtained via a judge) and in accordance with the node selection strategy.
            generation_temperature (float): The temperature to use when generating responses. A float between 0 and 1.
            judge_temperature (float): The temperature to use when judging responses. A float between 0 and 1.
            n_samples_generation (int): The number of samples to generate for each argument.
            n_samples_judge (int): The number of samples to use when judging arguments. 
            do_visualize_tree (bool): Whether to visualize the tree of thoughts.
            do_save_tree (bool): Whether to save the tree of thoughts
            response_length (str): The desired length of the response. For example, "a few sentences" or "a paragraph".
        
        Returns:
            str: The best argumentative response generated by the Tree-Of-Thought module.
        """
        assert depth > 0, "The number of steps (depth) for Tree-Of-Thoughts must be greater than 0."
        start_time = time.time()
        tree: Tree = initialize_tree(
            debate_state=state, 
            tree_type=constants.TreeOfThoughtsType.BFS.value,
        )
        frontier: typing.List[TreeOfThoughtsNode] = [tree.root]
        for layer in range(depth):  # Iterate over one layer in the tree at each step (i.e., increase the depth of the tree)
            self.logger.info(f"Expanding layer #{layer + 1}. Frontier is: {[frontier_node.index for frontier_node in frontier]}")
            new_frontier = self.step(
                tree=tree,
                frontier=frontier,
                top_k=top_k,
                n_samples_judge=n_samples_judge,
                n_samples_generation=n_samples_generation,
                judge_temperature=judge_temperature,
                generation_temperature=generation_temperature,
                response_length=response_length,
            )
            frontier = new_frontier
        end_time = time.time()
        self.logger.info(f'Tree generation took {end_time - start_time} seconds.')
        log_tree_of_thoughts(tree=tree, logger=self.logger)
        if do_save_tree:
            save_tree(tree)
        if do_visualize_tree:
            draw_graph(
                graph=tree_to_graph(tree),
                name=f"{generate_name(tree.root.state.topic)}",
            )
        # Get the best node from the most recent frontier (leaf nodes)
        best_node: TreeOfThoughtsNode = max(frontier, key=lambda node: node.score)
        best_response = best_node.state.previous_drafts[-1]
        return best_response
        
    def step(
        self, 
        tree: Tree, 
        frontier: typing.List[TreeOfThoughtsNode], 
        top_k: int,
        n_samples_judge: int,
        n_samples_generation: int,
        judge_temperature: float,
        generation_temperature: float,
        response_length: str,
    ) -> typing.List[TreeOfThoughtsNode]:
        """
        Perform a step (generate, evaluate, select) arguments for the next layer of the tree.
        
        Perform a step in the Tree-Of-Thoughts framework to generate a collection of arguments for the next layer of the tree.
        Given this collection of arguments, evaluate them and select the top-k arguments to continue to the next layer of the tree.
        Unlike the first step, this step expands multiple nodes in the tree (i.e., the frontier). Each node in the frontier represents
        a different argumenative direction in the debate.

        Parameters:
            tree (Tree): The tree of thoughts.
            frontier (List[TreeOfThoughtsNode]): The list of nodes to expand in the tree. Expanded nodes are then judged and 
                selected in order to determine the next frontier.
            top_k (int): The number of best nodes to select at each layer of the tree.
            n_samples_judge (int): The number of samples to use when judging the drafts.
            n_samples_generation (int): The number of samples to generate for each response.
            judge_temperature (float): The temperature to use when judging the drafts.
            generation_temperature (float): The temperature to use when generating the drafts.
            response_length (str): The desired length of the response. For example, "a few sentences" or "a paragraph".
        
        Returns:
            List[TreeOfThoughtsNode]: The nodes in the next layer of the tree. These nodes are the top-k nodes selected for 
                further expansion.
        """
        # Generation
        new_nodes: typing.List[TreeOfThoughtsNode] = self.create_drafts(
            tree=tree,
            frontier=frontier,
            n_samples_generation=n_samples_generation,
            generation_temperature=generation_temperature,
            response_length=response_length,
        )
        self.logger.info(
            f"\tCreated nodes [{', '.join([str(node.index) for node in new_nodes])}]."
        )
        
        # Evaluation
        if self.evaluation_strategy == constants.EvaluationStrategy.SCORE.value:    # Use "scoring" module for judges
            judged_nodes: typing.List[TreeOfThoughtsNode] = []
            for node in new_nodes:
                judged_node: TreeOfThoughtsNode = self.score_node(
                    node=node,
                    tree=tree,
                    n_samples_judge=n_samples_judge,
                    judge_temperature=judge_temperature,
                )
                judged_nodes.append(judged_node)
        else:                                                                       # Use "voting" module for judges
            judged_nodes: typing.List[TreeOfThoughtsNode] = self.vote_on_nodes(
                nodes=new_nodes,
                tree=tree,
                n_samples_judge=n_samples_judge,
                judge_temperature=judge_temperature,
            )
        self.logger.info(
            f"\tNodes [{', '.join([str(node.index) for node in judged_nodes])}] "
            f"received scores: {[node.score for node in judged_nodes]}"
        )

        # Selection
        if self.node_selection_strategy == constants.NodeSelectionStrategy.GREEDY.value:
            frontier = sorted(
                judged_nodes,
                key=lambda node: node.score,
                reverse=True,
            )[:top_k]
        elif self.node_selection_strategy == constants.NodeSelectionStrategy.SAMPLE.value:
            values = [tree.nodes[node_id].score for node_id in new_nodes]
            ps = np.array(values) / sum(values)
            frontier = np.random.choice(judged_nodes, size=top_k, p=ps).tolist()
        else:
            raise ValueError(f'Invalid node selection strategy: {self.node_selection_strategy}. Must be one of ["greedy", "sample"].')

        # Sort by index once we've selected the top-k nodes. This is to ensure that the order of the nodes in the frontier is consistent.
        frontier = sorted(frontier, key=lambda node: node.index)
        return frontier
    
    def create_drafts(
        self, 
        tree: Tree, 
        frontier: typing.List[TreeOfThoughtsNode], 
        n_samples_generation: int,
        generation_temperature: float,
        response_length: str,
    ) -> typing.List[TreeOfThoughtsNode]:
        """
        Expand each node in the frontier in order to obtain the nodes in the next layer of the tree.
        
        Parameters:
            tree (Tree): The tree of thoughts.
            frontier (List[TreeOfThoughtsNode]): The nodes in the frontier.
            n_samples_generation (int): The number of samples to generate for each response.
            generation_temperature (float): The temperature to use when generating the drafts.
            response_length (str): The desired length of the response. For example, "a few sentences" or "a paragraph".
        
        Returns:
            List[TreeOfThoughtsNode]: The nodes in the next layer of the tree. Note that these nodes do not yet include
                the scores or reasoning behind the scores (which are added during the judging phase).
        """
        new_nodes = []
        # Expand each node in the frontier
        for node in frontier:
            if len(node.state.previous_drafts) == 0:
                if len(node.state.conversation) == 0:
                    completions = self.single_turn_first_draft_predictor(
                        branching_input=ToTSingleTurnResponseBranchingInput(
                            topic=node.state.topic,
                            stance=node.state.stance,
                            length=response_length,
                        ),
                        config=dict(n=n_samples_generation, temperature=generation_temperature),
                    ).completions
                else:
                    completions = self.multi_turn_first_draft_predictor(
                        branching_input=ToTMultiTurnResponseBranchingInput(
                            topic=node.state.topic,
                            stance=node.state.stance,
                            length=response_length,
                            conversation=node.state.conversation,
                        ),
                        config=dict(n=n_samples_generation, temperature=generation_temperature),
                    ).completions
            else:
                if len(node.state.conversation) == 0:
                    completions = self.single_turn_expansion_predictor(
                        branching_input=ToTSingleTurnDraftBranchingInput(
                            topic=node.state.topic,
                            stance=node.state.stance,
                            length=response_length,
                            previous_drafts=node.state.previous_drafts,
                        ),
                        config=dict(n=n_samples_generation, temperature=generation_temperature),
                    ).completions
                else:
                    completions = self.multi_turn_expansion_predictor(
                        branching_input=ToTMultiTurnDraftBranchingInput(
                            topic=node.state.topic,
                            stance=node.state.stance,
                            length=response_length,
                            conversation=node.state.conversation,
                            previous_drafts=node.state.previous_drafts,
                        ),
                        config=dict(n=n_samples_generation, temperature=generation_temperature),
                    ).completions
            # 'completions.branching_output' is a list of length `n_samples_generation`, which contains the outputs for each 
            # sample.
            outputs = [output.response for output in completions.branching_output]
            # 'completions.reasoning' is a list of length `n_samples_generation`, which contains the reasoning for each output.
            reasonings = completions.reasoning if self.use_chain_of_thoughts else ["N/A"] * len(outputs)
            if len(outputs) != n_samples_generation:
                self.logger.warning(f"The number of outputs generated must be equal to `n_samples_generation`. Expected: {n_samples_generation}, Actual: {len(outputs)}.")
            if len(reasonings) != n_samples_generation:
                self.logger.warning(f"The number of reasonings generated must be equal to `n_samples_generation`. Expected: {n_samples_generation}, Actual: {len(reasonings)}.")
            # Create a (child) node in the tree for each response from a parent node
            for output, reasoning in zip(outputs, reasonings):
                child_node = TreeOfThoughtsNode(
                    index=len(tree.nodes),
                    state=DraftState(
                        topic=node.state.topic,
                        stance=node.state.stance,
                        conversation=node.state.conversation,
                        previous_drafts=node.state.previous_drafts + [output],
                    ),
                    parent_id=node.index,
                )
                tree.add_child_node(
                    parent_node=node, 
                    child_node=child_node, 
                    response=output, 
                    expansion_reasoning=reasoning,
                )
                new_nodes.append(child_node)
        return new_nodes

    @staticmethod
    def _wrap_previous_drafts(previous_drafts: typing.List[str]) -> str:
        """
        Wrap the previous drafts an informative string.
        
        Parameters:
            previous_drafts (List[str]): The previous drafts to wrap.
        
        Returns:
            str: The wrapped previous drafts. The previous drafts have a header and are separated by newlines.
        """
        result = ""
        for i, draft in enumerate(previous_drafts):
            result += f"Debator #{i + 1} draft:\n{draft}\n\n"
        return result.strip()
    
    @staticmethod
    def _wrap_judge_reasoning(
        reasoning: typing.List[str],
    ) -> str:
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
    
    def vote_on_nodes(
        self,
        nodes: typing.List[TreeOfThoughtsNode],
        tree: Tree,
        n_samples_judge: int,
        judge_temperature: float,
    ) -> typing.List[TreeOfThoughtsNode]:
        """
        Aggregate votes on which of the drafts is most persuasive, and assign scores to nodes based on the votes.
        
        Parameters:
            nodes (List[TreeOfThoughtsNode]): The nodes in the tree of thoughts.
            tree (Tree): The tree of thoughts.
            n_samples_judge (int): The number of samples to use when judging the drafts.
            judge_temperature (float): The temperature to use when judging the drafts.
        
        Returns:
            List[TreeOfThoughtsNode]: The nodes in the tree of thoughts with the scores assigned (based on the votes).
                These nodes are also augmented to include the reasoning behind the votes (if applicable).
        """
        if len(nodes[0].state.conversation) == 0:
            if len(nodes[0].state.previous_drafts) < 2:
                completions = self.single_turn_vote_judge(
                    judge_input=ToTJudgeVoteSingleTurnInput(
                        topic=nodes[0].state.topic,
                        arguments={node_index: node.state.previous_drafts[-1] for node_index, node in enumerate(nodes)},
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
            else:
                completions = self.single_turn_vote_judge_with_drafts(
                    judge_input=ToTJudgeVoteSingleTurnWithDraftsInput(
                        topic=nodes[0].state.topic,
                        arguments={node_index: node.state.previous_drafts[-1] for node_index, node in enumerate(nodes)},
                        previous_drafts={
                            node_index: self._wrap_previous_drafts(node.state.previous_drafts[:-1])
                            for node_index, node in enumerate(nodes)
                        },
                    )
                ).completions
        else:
            # All nodes in the frontier are in the same layer of the tree. They all share the same number of previous drafts.
            if len(nodes[0].state.previous_drafts) < 2:  # The current argument is the last element of the previous drafts.
                completions = self.multi_turn_vote_judge(
                    judge_input=ToTJudgeVoteMultiTurnInput(
                        topic=nodes[0].state.topic,
                        arguments={node_index: node.state.previous_drafts[-1] for node_index, node in enumerate(nodes)},
                        # Conversation is static. All nodes share the same conversation.
                        conversation=nodes[0].state.conversation,   
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
            else:
                completions = self.multi_turn_vote_judge_with_drafts(
                    judge_input=ToTJudgeVoteMultiTurnWithDraftsInput(
                        topic=nodes[0].state.topic,
                        arguments={node_index: node.state.previous_drafts[-1] for node_index, node in enumerate(nodes)},
                        # Conversation is static. All nodes share the same conversation.
                        conversation=nodes[0].state.conversation,
                        # Convert list of previous drafts to unified string
                        previous_drafts={
                            node_index: self._wrap_previous_drafts(node.state.previous_drafts[:-1])
                            for node_index, node in enumerate(nodes)
                        },
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
        one_hot_votes: typing.List[int] = [completion.index for completion in completions.judge_output]
        # Reasoning is a list of length `n_samples_judge`, which contains the reasoning for each vote (in an ensemble of voting 
        # judges). Therefore, the reasoning for the resulting scores of each node is the reasoning of the combined votes.
        reasoning = self._wrap_judge_reasoning(completions.reasoning) if self.use_chain_of_thoughts else "N/A"
        counts = [0 for _ in range(len(nodes))]
        for vote in one_hot_votes:
            counts[vote] += 1
        # Scores are the fraction of votes for each node
        scores = [count / sum(counts) for count in counts]
        # Assign the scores to the nodes in the tree
        for node, score in zip(nodes, scores):
            tree.nodes[node.index].score = score
            tree.nodes[node.index].reasoning = reasoning
        return nodes

    def score_node(
        self, 
        node: TreeOfThoughtsNode, 
        tree: Tree,
        n_samples_judge: int,
        judge_temperature: float,
        num_decimals_in_score: int = 3,
    ) -> TreeOfThoughtsNode:
        """
        Judge the drafts generated for a node in the tree of thoughts.
        
        Parameters:
            node (TreeOfThoughtsNode): The node in the tree of thoughts.
            tree (Tree): The tree of thoughts.
            n_samples_judge (int): The number of samples to use when judging the drafts.
            judge_temperature (float): The temperature to use when judging the drafts.
            num_decimals_in_score (int): The number of decimals to round the score to.
        
        Returns:
            TreeOfThoughtsNode: The node in the tree of thoughts with the score assigned (and reasoning, if applicable).
        """
        if len(node.state.conversation) == 0:
            if len(node.state.previous_drafts) < 2:
                completions = self.single_turn_score_judge(
                    judge_input=ToTSingleTurnJudgeInput(
                        topic=node.state.topic,
                        argument=node.state.previous_drafts[-1],
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
            else:
                completions = self.single_turn_score_judge_with_drafts(
                    judge_input=ToTSingleTurnJudgeWithDraftsInput(
                        topic=node.state.topic,
                        argument=node.state.previous_drafts[-1],
                        previous_drafts=self._wrap_previous_drafts(node.state.previous_drafts[:-1]),
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
            score = np.mean([completion.judge_output.score for completion in completions]).round(num_decimals_in_score).item()
        else:
            if len(node.state.previous_drafts) < 2:
                completions = self.multi_turn_judge(
                    judge_input=ToTMultiTurnJudgeInput(
                        topic=node.state.topic,
                        conversation=node.state.conversation + [node.state.previous_drafts[-1]],  # Use the most recent draft
                        argument=node.state.previous_drafts[-1],
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
            else:
                completions = self.multi_turn_score_judge_with_drafts(
                    judge_input=ToTMultiTurnJudgeWithDraftsInput(
                        topic=node.state.topic,
                        conversation=node.state.conversation,
                        argument=node.state.previous_drafts[-1],
                        previous_drafts=self._wrap_previous_drafts(node.state.previous_drafts[:-1]),
                    ),
                    config=dict(
                        n=n_samples_judge,                
                        temperature=judge_temperature,
                    ),
                ).completions
        score = np.mean([completion.judge_output.score for completion in completions]).round(num_decimals_in_score).item()
        judge_reasoning = self._wrap_judge_reasoning(completions.reasoning) if self.use_chain_of_thoughts else "N/A"
        # Modify the node in the tree
        tree.nodes[node.index].score = score
        tree.nodes[node.index].reasoning = judge_reasoning
        return tree.nodes[node.index]


if __name__ == '__main__':
    # Parse the command-line arguments    
    args = parser.parse_args()
    set_up_dspy(
        openai_key=args.openai_key,
        openai_key_path=args.openai_key_path,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    # Initialize the conversation state
    # TODO: Modify 'create_conversation_state' to be more user-friendly.
    conversation_state = create_conversation_state(
        topic=args.topic,
        stance=args.stance,
        conversation_path=args.conversation_path,
    )

    print('Arguments:')
    for arg in vars(args):
        print(f'\t{arg}: {getattr(args, arg)}')
    print(f'Initial conversation state:\n{conversation_state}')

    # Initialize the Tree-Of-Thoughts module
    tree_of_thoughts = TreeOfThoughts(
        use_chain_of_thought=args.use_chain_of_thought,
        node_selection_strategy=args.node_selection_strategy,
        evaluation_strategy=args.evaluation_strategy,
    )

    # Generate the tree of thoughts
    response = tree_of_thoughts(
        state=conversation_state,
        depth=args.depth,
        top_k=args.top_k,
        generation_temperature=args.generation_temperature,
        judge_temperature=args.judge_temperature,
        n_samples_generation=args.n_samples_generation,
        n_samples_judge=args.n_samples_judge,
        do_visualize_tree=args.with_visualization,
        do_save_tree=args.save_tree,
        response_length=args.response_length,
    )
    print(f'Final response:\n{response}')
