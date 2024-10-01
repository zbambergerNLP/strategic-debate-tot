from logging import Logger
import typing
from abstractions.tree.tree import (
    State, 
    Node,
    MonteCarloNode,
    Tree,
)
from abstractions.generator.generator import ResponseParameters
from pydantic import Field


class DraftState(State):
    """Represents a planning step (draft) by a debator when writing a persuasive argument for a debate (on the given topic)."""
    previous_drafts: typing.List[str] = Field(
        default_factory=list,
        description="A list of previous drafts of the debator's argument.",
    )

    @property
    def reasoning_steps(self) -> typing.List[str]:
        return self.previous_drafts

    def state_to_generator_input(self, response_parameters: ResponseParameters) -> typing.Dict[str, str | typing.List[str]]:
        """
        Converts the state to an input for the generator.
        """
        if len(self.conversation) == 0:
            if len(self.previous_drafts) == 0:
                return {"topic": self.topic, "stance": self.stance, "response_parameters": response_parameters}
            else:
                return {
                    "topic": self.topic, "stance": self.stance, "response_parameters": response_parameters, "previous_drafts": self.previous_drafts
                }
        else:
            if len(self.previous_drafts) == 0:
                return {"topic": self.topic, "stance": self.stance,  "response_parameters": response_parameters, "conversation": self.conversation}
            else:
                return {
                    "topic": self.topic, 
                    "stance": self.stance, 
                    "response_parameters": response_parameters, 
                    "previous_drafts": self.previous_drafts,
                    "conversation": self.conversation,
                }
            
    def reasoning_steps_to_string(self, exclude_most_recent: bool) -> str:
        """
        Converts the reasoning steps to an informative string.
        
        Parameters:
            exclude_most_recent (bool): Whether to exclude the most recent draft.
        
        Returns:
            str: The reasoning steps as a string. The reasoning steps have a header and are separated by newlines.
        """
        result = ""
        previous_drafts_to_consider = self.previous_drafts[:-1] if exclude_most_recent else self.previous_drafts
        for i, draft in enumerate(previous_drafts_to_consider):
            result += f"Draft {i + 1}:\n{draft}\n\n"
        return result.strip()
    
    def state_to_evaluator_input(self) -> typing.Dict[str, str | typing.List[str]]:
        """
        Converts the state to an input for the evaluator.
        The most recent draft is the argument to be evaluated.
        """
        if len(self.conversation) == 0:
            if len(self.previous_drafts) < 2:       # Only a single draft has been made so far
                return {"topic": self.topic, "stance": self.stance, "argument": self.previous_drafts[-1]}
            else:
                return {
                    "previous_drafts": self.reasoning_steps_to_string(exclude_most_recent=True),
                    "argument": self.previous_drafts[-1],
                    "topic": self.topic,
                }
        else:
            if len(self.previous_drafts) < 2:
                return {
                    "argument": self.previous_drafts[-1],
                    "conversation": self.conversation,
                    "topic": self.topic,
                }
            else:
                return {
                    "previous_drafts": self.reasoning_steps_to_string(exclude_most_recent=True),
                    "argument": self.previous_drafts[-1],
                    "conversation": self.conversation,
                    "topic": self.topic,
                }
    

class DraftNode(Node):
    """A node in a search tree that stores drafts of persuasive arguments for a debate."""
    score: float = Field(
        default=0.0,
        description="The quality score of a node, which entails a perspective argumentative response.",
    )
    state: DraftState = Field(
        ...,
        description="""
A representation of a node's state in a Tree-of-Thought framework.
A state contains the topic of the debate, the stance of the debator towards the topic, and conversation so far.
Notably, this state includes a list of previous drafts of the debator's argument, where each draft is designed to improve
upon the previous draft iterations.
""".strip(),
    )

class MCTSDraftNode(DraftNode, MonteCarloNode):
    """A node in a search tree that stores drafts of persuasive arguments for a debate."""


class DraftTree(Tree):
    """
    A search tree for a debate, where nodes are conversation states in the debate,
    and edges are responses between rival debators.
    
    In this type of tree, nodes also contain "drafts" of persuasive arguments for the debate.
    That is, a child node within the tree crafts a new draft while taking into account the previous drafts.
    """

    def _add_node_to_tree_from_existing_conversation(
        self, 
        message_index: int, 
        topic: str, 
        stance: str, 
        conversation: typing.List[str],
    ) -> Node:
        return DraftNode(
            index=message_index,
            state=DraftState(
                topic=topic,
                stance=stance,
                conversation=conversation[:message_index],
                previous_drafts=[],
            ),
            parent_id=message_index - 1 if message_index > 0 else None,
            children_ids=[message_index + 1],
        )
    
    def _initialize_root(self, state: State) -> Node:
        return DraftNode(
            index=len(state.conversation),
            state=DraftState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                previous_drafts=[],
            ),
            parent_id=(
                len(state.conversation) - 1  if len(state.conversation) > 0  else None
            ),
        )

    def create_child_node(self, index: int, state: DraftState, output: str) -> Node:
        return DraftNode(
            index=index,
            state=DraftState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                previous_drafts=state.previous_drafts + [output],
            ),
        )

    def create_voting_input_from_most_recent_layer(self) -> typing.Dict[str, typing.Any]:
        most_recent_layer = self.layers[-1]
        node = most_recent_layer[0]     # Select an arbitrary node from the most recent layer
        arguments = {node_index: node.state.reasoning_steps[-1] for node_index, node in enumerate(most_recent_layer)}
        if len(node.state.conversation) == 0:           # Single-turn
            if len(node.state.reasoning_steps) < 2:     # Only first reasoning step
                return {
                    'arguments': arguments,
                    'topic': node.state.topic,
                }
            else:
                return {
                    'previous_drafts': {
                        node_index: node.state.reasoning_steps_to_string(exclude_most_recent=True)
                        for node_index, node in enumerate(most_recent_layer)
                    },
                    'arguments': arguments,
                    'topic': node.state.topic,
                }
        else:  # Multi-turn
            if len(node.state.reasoning_steps) < 2:
                return {
                    'conversation': node.state.conversation,
                    'arguments': arguments,
                    'topic': node.state.topic,
                }
            else:
                return {
                    'previous_drafts': {
                        node_index: node.state.reasoning_steps_to_string(exclude_most_recent=True)
                        for node_index, node in enumerate(most_recent_layer)
                    },
                    'conversation': node.state.conversation,
                    'arguments': arguments,
                    'topic': node.state.topic,
                }

    def log_tree(self, logger: Logger):
        for node in self.nodes:
            logger.info(f"\n\nNode {(node.index)}, Stance: {node.state.stance}")
            if(node.parent_id is not None):
                logger.info(f'\tPrevious draft: {self.edges[node.parent_id][node.index].response}')
            logger.info(f'\tTree-of-Thoughts score: {node.score}')
            logger.info('\tDrafts:')
            for child_index in node.children_ids:
                logger.info(
                    f'\n\t- Draft #{child_index} [score = {self.nodes[child_index].score}]: '
                    f'{self.edges[node.index][child_index].response}'
                )
                # Print the reasoning associated with the response if it exists
                if self.edges[node.index][child_index].reasoning:
                    logger.info(f'\tReasoning: {self.edges[node.index][child_index].reasoning}')

class MCTSDraftTree(DraftTree):
    """
    A search tree for a debate, where nodes are conversation states in the debate,
    and edges are responses between rival debators.
    
    In this type of tree, nodes also contain "drafts" of persuasive arguments for the debate.
    That is, a child node within the tree crafts a new draft while taking into account the previous drafts.
    
    Furthermore, since this is a Monte Carlo Tree Search (MCTS) tree, a node's score is accumulated over multiple simulations, 
    and the node contains an additional field for the number of visits.
    """

    def _add_node_to_tree_from_existing_conversation(
        self, 
        message_index: int, 
        topic: str, 
        stance: str, 
        conversation: typing.List[str],
    ) -> Node:
        return MCTSDraftNode(
            index=message_index,
            state=DraftState(
                topic=topic,
                stance=stance,
                conversation=conversation[:message_index],
                previous_drafts=[],
            ),
            parent_id=message_index - 1 if message_index > 0 else None,
            children_ids=[message_index + 1],
            score=0,
            visits=0,
        )
    
    def _initialize_root(self, state: State) -> Node:
        return MCTSDraftNode(
            index=len(state.conversation),
            state=DraftState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                previous_drafts=[],
            ),
            parent_id=(
                len(state.conversation) - 1  if len(state.conversation) > 0  else None
            ),
            score=0,
            visits=0,
        )

    def create_child_node(
        self,
        index: int,
        state: DraftState,
        output: str,
    ):
        return MCTSDraftNode(
            index=index,
            state=DraftState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                previous_drafts=state.previous_drafts + [output],
            ),
            score=0,
            visits=0,
        )
    
    def log_tree(self, logger: Logger):
        for node in self.nodes:
            logger.info(f"\n\nNode {(node.index)}, Stance: {node.state.stance}")
            if(node.parent_id is not None):
                logger.info(f'\tPrevious draft: {self.edges[node.parent_id][node.index].response}')
            logger.info(f'\tMonte Carlo score: {node.score}. Number of visits: {node.visits}')
            logger.info('\tDrafts:')
            for child_index in node.children_ids:
                logger.info(
                    f'\t- Draft #{child_index} [score = {self.nodes[child_index].score}]: '
                    f'{self.edges[node.index][child_index].response}'
                )
                # Print the reasoning associated with the response if it exists
                if self.edges[node.index][child_index].reasoning:
                    logger.info(f'\t\tReasoning: {self.edges[node.index][child_index].reasoning}')
