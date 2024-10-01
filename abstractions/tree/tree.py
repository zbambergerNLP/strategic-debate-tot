import abc
import json
import logging
import os
from pydantic import (
    BaseModel, 
    Field, 
)
from abstractions.generator.generator import ResponseParameters
import typing

from utils.utils import generate_name


###########################################################
### Abstractions for Strategic Debate + Tree-of-Thought ###
###########################################################

class State(BaseModel):
    """Represents a point in time in a debate between two rival debators (with opposing stances towards the given topic)."""

    topic: str = Field(
        description="The topic of the debate (e.g., 'Collaboration is better than competition').",
    )
    stance: str = Field(
        description="The stance of the debate. Either 'PRO' or 'ANTI'.",
    )
    conversation: typing.List[str] = Field(
        default=[],
        description="""
A list of messages in the conversation so far. The last message is the most recent message. 
Each message is preceded by the message of the rival debator.
""".strip(),
    )

    # TODO: Convert 'length' to 'generation_config' (which includes parameters such as 'length', 'formality', and 'tone')
    def state_to_generator_input(self, response_parameters: ResponseParameters) -> typing.Dict[str, str | typing.List[str]]:
        """Converts the state to an input for the generator."""
        if len(self.conversation) == 0:
            return {'topic': self.topic, 'stance': self.stance, 'response_parameters': response_parameters}
        else:
            return {'topic': self.topic, 'stance': self.stance, 'response_parameters': response_parameters, 'conversation': self.conversation}
    
    @property
    def reasoning_steps(self) -> typing.List[str]:
        """Retrieves the reasoning steps from the conversation."""
        return self.conversation
    
    def reasoning_steps_to_string(self, exclude_most_recent: bool = False) -> str:
        """
        Wraps the reasoning steps in a single string.
        """
        if exclude_most_recent:
            return "\n".join(self.reasoning_steps[:-1])
        return "\n".join(self.reasoning_steps)
    
    def state_to_evaluator_input(self) -> typing.Dict[str, str | typing.List[str]]:
        """
        Converts the state to an input for the evaluator.
        The most recent message in the conversation is the argument to be evaluated.
        """
        if len(self.conversation) == 0:
            return {'topic': self.topic, 'stance': self.stance, 'argument': self.conversation[-1]}
        else:
            return {
                'topic': self.topic, 
                'stance': self.stance, 
                'conversation': self.conversation[:-1], 
                'argument': self.conversation[-1]
            }
    

############
### Node ###
############

# Minimax Node

class Node(BaseModel):
    """A node in a search tree."""

    index: int = Field(
        ...,
        description="The index of the node in the tree.",
    )
    state: State = Field(
        ...,
        description="""
A representation of a node's state in a Tree-of-Thought framework. 
A state contains the topic of the debate, the stance of the debator towards the topic, and conversation so far.
""".strip(),
    )
    parent_id: typing.Optional[int] = Field(
        default=None, 
        description="""
The index of the parent node of this node. The response from the rival debator is an edge from the parent to this node.
""".strip(),
    )
    score: typing.Optional[float] = Field(
        default=0, 
        description="""
A floating point number between 0 and 1, where 1 represents that 'stance_to_maximize' is most likely to win the debate,
and 0 represents that 'stance_to_maximize' is most likely to lose the debate.
""".strip(),
    )
    children_ids: typing.Optional[typing.List[int]] = Field(
        default_factory=list,
        description="""
The indices of the children of this node. The edges to each child correspond with possible responses to the most recent message in the debate.
""".strip(),
    )
    reasoning: typing.Optional[str] = Field(
        default=None,
        description="The reasoning behind the score of this node (as computed from the leaf nodes via minimax).",
    )
    is_pruned: bool = Field(
        default=False,
        description="Whether the node has been pruned from the tree.",
    )

    def __lt__(self, other: 'Node'):
        return self.score < other.score
    

class MonteCarloNode(Node):
    """A node in a search tree that stores the results of Monte Carlo simulations."""
    score: float = Field(
        default=0.0,
        description="The average score of this node from the Monte Carlo simulations.",
    )
    visits: int = Field(
        default=0,
        description="The number of times this node has been visited in the Monte Carlo simulations.",
    )


############
### Edge ###
############

class Edge(BaseModel):
    """An edge in search tree representing a response from one debator to their rival"""

    response: str = Field(
        ...,
        description="""
The text response produced by the debator in the source node (who's stance is opposite the stance of the 
debator in the target node).
""".strip,
    )
    reasoning: typing.Optional[str] = Field(
        default=None,
        description="The reasoning behind the response.",
    )

############
### Tree ###
############

class Tree(abc.ABC):

    @abc.abstractmethod
    def _add_node_to_tree_from_existing_conversation(
        self,
        message_index: int,
        topic: str,
        stance: str,
        conversation: typing.List[str],
    ) -> Node:
        """
        Adds a node to the tree based on an existing conversation.
        
        Parameters:
            message_index (int): The index of the message in the conversation.
            topic (str): The topic of the debate.
            stance (str): The stance of the debator (towards the topic) at the specified message index.
            conversation (List[str]): The full conversation so far.
        
        Returns:
            Node: The node to add to the tree (corresponding to the specified message).
        """
        pass

    def _initialize_conversation(
        self,
        stance: str,
        topic: str,
        conversation: typing.List[str],
    ) -> typing.List[str]:
        """
        Initializes the conversation with the given messages.
        
        Parameters:
            conversation (List[str]): The messages to initialize the conversation with.
        
        Returns:
            List[str]: The initialized conversation.
        """
        # Add nodes and edges for each message in the conversation
        current_stance = stance
        # Determine the stance of the first message in the conversation by reversing the order of the conversation,
        # and iteratively flipping the stance for each message (given only the status of the "root" message)
        for message in reversed(conversation):
            current_stance = "PRO" if current_stance == "ANTI" else "ANTI"
        for message_index, message in enumerate(conversation):
            # Add a node for the message
            self.nodes.append(
                self._add_node_to_tree_from_existing_conversation(
                    message_index=message_index,
                    topic=topic,
                    stance=current_stance,
                    conversation=conversation,
                )
            )
            # Set the stance for the next message
            current_stance = "PRO" if current_stance == "ANTI" else "ANTI"
            # Add an edge from the parent to the child
            if message_index > 0:
                self.add_edge(
                    source_index=message_index - 1,
                    target_index=message_index,
                    edge=Edge(response=message),
                )
    
    @abc.abstractmethod
    def _initialize_root(
        self,
        state: State,
    ) -> Node:
        """
        Initializes the root node of the tree.
        
        Parameters:
            state (State): A representation of the context that a language model uses to generate a response.

        Returns:
            Node: The root node of the tree.
        """
        return Node(
            index=len(state.conversation),
            state=State(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
            ),
            parent_id=(
                len(state.conversation) - 1  if len(state.conversation) > 0  else None
            ),
        )
        
    def __init__(
        self, 
        state: State,
    ):
        """
        Initializes the search tree from the given state.
        """
        self.nodes, self.edges = [], {}
        self.layers: typing.List[typing.List[Node]] = []
        if len(state.conversation):  # Non-empty conversation
            # Initialize nodes corresponding to the existing conversation
            self._initialize_conversation(
                stance=state.stance,
                topic=state.topic,
                conversation=state.conversation,
            )
            # Add the root node
            self.root = self._initialize_root(state)
            # Add the root node to the list of nodes
            self.nodes.append(self.root)
            # Add an edge to the root node from the last message in the conversation
            self.add_edge(
                source_index=len(state.conversation) - 1, 
                target_index=len(state.conversation), 
                edge=Edge(response=state.conversation[-1])
            )
        else:
            self.root = self._initialize_root(state)
            self.nodes.append(self.root)
    
    def add_layer(self, layer: typing.List[Node]) -> None:
        """
        Adds a layer to the tree.
        
        Parameters:
            layer (List[Node]): The layer to add.
        """
        self.layers.append(layer)
    
    @abc.abstractmethod
    def create_child_node(self, index: int, state: State, output: str) -> Node:
        """
        Create a child node, and include the output in the child state.

        Parameters:
            index (int): The index of the child node.
            state (State): The state of the child node that we are creating.
            output (str): The output to include in the child state.
        """
        pass

    def add_edge(self, source_index: int, target_index: int, edge: Edge) -> None:
        """
        Add an edge to the tree from the source node to the target node.
        
        Parameters:
            source_index (int): The index of the source node.
            target_index (int): The index of the target node.
            edge (Edge): The edge to add.    
        """
        if source_index not in self.edges:
            self.edges[source_index] = {}
        self.edges[source_index][target_index] = edge

    def add_child_node(
        self,
        parent_node: Node,
        child_node: Node,
        response: str,
        expansion_reasoning: str,
    ) -> None:
        """
        Adds a single child node to the tree based on the given claim.

        Adds an edge from the parent node to the child node, updates the parent node's children_ids list, 
        includes the parent ID in the child node, and adds the child node to the tree.

        Parameters:
            parent_node (Node): The parent node to which the child node will be added.
            child_node (Node): The child node to add.
            response (str): The response produced by the parent node (included in the state of the child node, and in the 
                edge from the parent node to the child node).
            expansion_reasoning (str): The reasoning behind the expansion of the tree. This is produced by the LLM when
                using chain-of-thought.
        """
        edge = Edge(source=parent_node.index, target=child_node.index, response=response, reasoning=expansion_reasoning)
        parent_node.children_ids.append(child_node.index)
        child_node.parent_id = parent_node.index
        self.nodes.append(child_node)
        self.add_edge(source_index=parent_node.index, target_index=child_node.index, edge=edge)
    
    def create_voting_input_from_most_recent_layer(self) -> BaseModel:
        """
        Creates an input for the voting judged based on the most recent layer of the tree.

        Extracts the most recent layer from the tree, and creates an input for the voting judge. The judge consists of an
        ensemble of LLMs, each of which votes for the best chain of reasoning so far (i.e., the paths from the root to each
        node in the most recent layer).

        Returns:
            BaseModel: The input for the voting judge. The specific input structure depends on the reasoning type of the tree.
        """
        raise NotImplementedError(f"Voting is not supported in module {self.__class__.__name__}.")
    
    def save_to_file(self, file_name: str = None) -> None:
        """
        Save the search tree to a JSON file.
        
        Parameters:
            file_name (str): The name of the file to save the tree to.
        """
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        if not file_name:
            file_name = generate_name(self.root.state.topic)
            json_path = os.path.join('outputs',f'{file_name}.json')
        else:
            json_path = file_name
        
        nodes = [json.loads(node.json()) for node in self.nodes]
        edges = {
            source: {
                target: json.loads(edge.model_dump_json()) for target, edge in edges.items()
            } for source, edges in self.edges.items()
        }
        root = json.loads(self.root.model_dump_json())
        print(f'Saving tree to {json_path}')
        with open(json_path, 'w') as file:
            json.dump({"root": root, "nodes": nodes, "edges": edges}, file, indent=4)
    
    def log_tree(self, logger: logging.Logger):
        """
        Logs the nodes and edges of the tree.
        
        Parameters:
            logger (logging.Logger): The logger to use for logging the tree.
        """
        for node in self.nodes:
            logger.info(f"\n\nNode {(node.index)}")
            if(node.parent_id is not None):
                logger.info(f'\tParent ID: {node.parent_id}')
                logger.info(f'\tParent Stance: {self.nodes[node.parent_id].state.stance}')
                logger.info(f'\tParent Response: {self.edges[node.parent_id][node.index].response}')
                logger.info(f'\tParent reasoning: {self.edges[node.parent_id][node.index].reasoning}')
            logger.info(f'\tStance: {node.state.stance}')
            if node.parent_id in self.nodes:
                logger.info(f'\tResponse: {self.edges[node.parent_id][node.index].response}')
            logger.info(f'\tScore: {node.score}')
            logger.info(f'\tChild nodes:')
            for child_index in node.children_ids:
                logger.info(
                    f'\t- Response #{child_index} [score = {self.nodes[child_index].score}]: '
                    f'{self.edges[node.index][child_index].response}'
                )
                # Print the reasoning associated with the response if it exists
                if self.edges[node.index][child_index].reasoning:
                    logger.info(f'\t\tReasoning: {self.edges[node.index][child_index].reasoning}')


#################
### Functions ###
#################

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
        get_random_topic (bool): Whether to use a random topic. Defaults to False.
    
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
