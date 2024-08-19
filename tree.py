import json
import logging
import os
from pydantic import (
    BaseModel, 
    Field, 
)
import typing
from utils import generate_name
from constants import TreeOfThoughtsType, Colors
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt
import textwrap


###########################################################
### Abstractions for Strategic Debate + Tree-of-Thought ###
###########################################################

class State(BaseModel):
    """Represents the context in which a debator chooses to produce a persuasive argument."""

    topic: str = Field(
        description="The topic of the argument (e.g., 'Collaboration is better than competition').",
    )
    stance: str = Field(
        description="The stance that the debator is taking towards the topic. Either 'PRO' or 'ANTI'.",
    )
    conversation: typing.List[str] = Field(
        description="A list of messages in the conversation so far. Each message is preceded by the message of the rival debator.",
    )

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
        conversation_path (str): The path to the conversation file. Each line in the file represents a message in the conversation.
    
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

class DraftState(State):
    """Represents a planning step (draft) by a debator when writing a persuasive argument (on the given topic).
    In addition to the topic, stance, and conversation, this state includes a list of previous drafts of the debator's argument.
    Each draft is designed to improve upon the previous draft iterations.
    """
    previous_drafts: typing.List[str] = Field(
        default_factory=list,
        description="A list of previous drafts of the debator's argument.",
    )

############
### Node ###
############

class Node(BaseModel):
    """A node in a search tree."""
    index: int = Field(
        default_factory=int,
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
The index of the parent node of this node. The argument by the debator in the parent node is the content of the edge 
whose target is this node.
""".strip(),
    )
    score: typing.Optional[float] = Field(
        default=None, 
        description="""
A floating point number between 0 and 1, where 1 represents that the argument is highly persuasive and 0 represents that the 
argument is not persuasive.
""".strip(),
    )
    children_ids: typing.Optional[typing.List[int]] = Field(
        default_factory=list,
        description="""
The indices of the children of this node. The edges from this node to each child correspond with possible argumentative responses.
""".strip(),
    )
    reasoning: typing.Optional[str] = Field(
        default=None,
        description="The reasoning which a LLM judge provided for the score of this node.",
    )

    def __lt__(self, other: 'Node'):
        return self.score < other.score
    

class TreeOfThoughtsNode(Node):
    """A node in a search tree that stores drafts of persuasive arguments."""
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

############
### Edge ###
############

class Edge(BaseModel):
    """An edge in search tree representing a persuasive argumentative response (text)."""

    response: str = Field(
        ...,
        description="A persuasive argumentative response.",
    )
    reasoning: typing.Optional[str] = Field(
        default=None,
        description="The reasoning behind the response.",
    )

############
### Tree ###
############

class Tree(BaseModel):
    """
    A search tree for argumentative responses, where nodes represent the state of an argument (i.e., the conversational context), 
    and edges are candidate responses.
    """

    root: Node = Field(
        ...,
        description="The root node of the tree. The root node represents the initial state of the argument.",
    )
    nodes: typing.List[Node] = Field(
        default_factory=list,
        description="""
The nodes in the tree represent states of the argument.
Each node contains the topic of the argument, the stance of the debator towards the topic, and the conversation so far.
""",
    )
    edges: typing.Dict[int, typing.Dict[int, Edge]] = Field(
        default_factory=dict,
        description="""
Persuasive arguments that are iteratively refined in the tree (e.g., via draft-refinement or draft-expansion).
The outer key is the index of the source node, and the inner key is the index of the target node. 
The value is the edge between the source and target nodes.
""".strip(),
    )

    def add_edge(
        self, 
        source_index: int,
        target_index: int,
        edge: Edge,
    ) -> None:
        """
        Adds an edge to the tree.
        
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

        Parameters:
            parent_node (Node): The parent node to which the child node will be added.
            child_node (Node): The child node to be added to the tree.
            response (str): The argumentative response to be added as a child node.
            expansion_reasoning (str): The reasoning which the LLM debator provided for generating the argumentative response.
        """
        
        edge = Edge(
            source=parent_node.index,
            target=child_node.index,
            response=response,
            reasoning=expansion_reasoning,
        )
        parent_node.children_ids.append(child_node.index)
        self.nodes.append(child_node)
        
        if parent_node.index not in self.edges:
            self.edges[parent_node.index] = {}
        
        self.edges[parent_node.index][child_node.index] = edge
        
#################
### Functions ###
#################


def initialize_tree(
    debate_state: State,
    tree_type: str,
) -> Tree:
    """
    Initializes the search tree for the Tree-of-Thoughts module.
    
    Parameters:
        debate_state (State): The state of the debate.
    
    Returns:
        Tree: The initialized search tree.
    """
    assert (
        tree_type in TreeOfThoughtsType.__members__, 
        f"Invalid tree type: {tree_type}. Must be one of {TreeOfThoughtsType.__members__}"
    )
    edges = {}
    nodes = []

    # Add nodes and edges for each message in the conversation
    current_stance = debate_state.stance

    # Determine the stance of the first message in the conversation by reversing the order of the conversation,
    # and iteratively flipping the stance for each message (given only the status of the "root" message)
    for message in reversed(debate_state.conversation):
        current_stance = "PRO" if current_stance == "ANTI" else "ANTI"

    for message_index, message in enumerate(debate_state.conversation):
        nodes.append(
            TreeOfThoughtsNode(
                index=message_index,
                state=DraftState(
                    topic=debate_state.topic,
                    stance=current_stance,
                    conversation=debate_state.conversation[:message_index],
                    previous_drafts=[],
                ),
                parent_id=message_index - 1 if message_index > 0 else None,
                children_ids=[message_index + 1],
            )
        )

        # Set the stance for the next message
        current_stance = "PRO" if current_stance == "ANTI" else "ANTI" 

        # Add an edge from the parent to the child
        if message_index > 0:
            if message_index - 1 not in edges:
                edges[message_index - 1] = {
                    message_index: Edge(
                        response=debate_state.conversation[message_index - 1],
                    )
                }
            else:
                edges[message_index - 1][message_index] = Edge(
                    response=message,
                )

    # Add the root node
    nodes.append(
        TreeOfThoughtsNode(
            index=len(debate_state.conversation),
            state=DraftState(
                topic=debate_state.topic,
                stance=debate_state.stance,
                conversation=debate_state.conversation,
                previous_drafts=[],
            ),
            parent_id=(
                len(debate_state.conversation) - 1  if len(debate_state.conversation) > 0  else None
            ),
        )
    )

    # Add an edge from the parent to the root node if the conversation is not empty
    if debate_state.conversation:
        if len(debate_state.conversation) - 1 not in edges:
            edges[len(debate_state.conversation) - 1] = {
                len(debate_state.conversation): Edge(
                    response=debate_state.conversation[-1],
                )
            }

        edges[len(debate_state.conversation) - 1][len(debate_state.conversation)] = Edge(
            response=debate_state.conversation[-1],
        )
    
    tree = Tree(
        root=nodes[-1], 
        nodes=nodes, 
        edges=edges,
    )
    return tree

def truncate_text(text: str, max_words: int = 50) -> str:
    """
    Truncates the text to a maximum number of words.
    
    Parameters:
        text (str): The text to truncate.
        max_words (int): The maximum number of words to truncate the text to.
    
    Returns:
        str: The truncated text.
    """
    return " ".join(text.split()[:max_words]) + "..." if len(text.split()) > max_words else text

def log_tree_of_thoughts(
    tree: Tree,
    logger: logging.Logger,
    max_words: int = 50,
):
    """
    Logs the nodes and edges of the tree.
    
    Parameters:
        tree (Tree): The tree to log.
        logger (logging.Logger): The logger to use for logging the tree.
    """
    for node in tree.nodes:
        logger.info(f"\n\n\nNode {(node.index)}, Stance: {node.state.stance}")
        logger.info(f"\tParent node: {node.parent_id}")
        if node.parent_id is not None:
            logger.info(
                f'\tPrevious draft: {truncate_text(tree.edges[node.parent_id][node.index].response, max_words=max_words)}'
            )
        if node.state.conversation:
            logger.info('\tConversation:')
            for message in node.state.conversation:
                logger.info(f'\t- {truncate_text(message)}')
        logger.info(f'\tTree-of-Thoughts score: {node.score}')
        logger.info('\tChildren (new drafts):')
        for child_index in node.children_ids:
            logger.info(
                f'\n\n\t\t- Node #{child_index} [score = {tree.nodes[child_index].score}]: '
                f'{truncate_text(tree.edges[node.index][child_index].response)}'
            )
            # Print the reasoning associated with the response if it exists
            if tree.edges[node.index][child_index].reasoning:
                logger.info(
                    f'\t\t\tDebator reasoning: {truncate_text(tree.edges[node.index][child_index].reasoning, max_words=max_words)}'
                )
            if tree.nodes[child_index].reasoning:
            # Print the reasoning associated with the judge if it exists
                logger.info(
                    f'\t\t\tJudge reasoning: {truncate_text(tree.nodes[child_index].reasoning, max_words=max_words)}'
                )


#####################
### Visualization ###
#####################

def save_tree(
        tree: Tree,
        file_name: str = None
    ):
        """
        Save the search tree to a JSON file.
        
        Parameters:
            tree (Tree): The search tree to save.
            file_name (str): The name of the file to save the tree to.
        """
        
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        
        if not file_name:
            file_name = generate_name(tree.root.state.topic)
            json_path = os.path.join(
                'outputs',
                f'{file_name}.json'
            )
        else:
            json_path = file_name
        
        with open(json_path, 'w') as file:
            json.dump(tree.model_dump(), file, indent=4)

def wrap_text(
        text: str, 
        width: int = 30, 
        max_words:int = 8
    ) -> str:
    """
    Wraps text to a specified width and maximum number of words.

    Parameters:
        text (str): The text to wrap.
        width (int): The width to wrap to.
        max_words (int): The maximum number of words to wrap to.

    Returns:
        str: The wrapped text.
    """
    # Replace colons with semi-colons. This is necessary to avoid an issue with graphviz.
    text = text.replace(":", ";")

    words = text.replace("\n", " ").split()[:max_words]
    return "\n".join(textwrap.wrap(" ".join(words), width=width)) + "..."

def tree_to_graph(
        tree: Tree,
    ) -> nx.DiGraph:
    """
    Converts a tree to a directed graph.

    Parameters:
        tree (Tree): The tree to convert.

    Returns:
        nx.DiGraph: The directed graph representation of the tree.
    """
    graph = nx.DiGraph()

    def add_node(
        node: Node,
        root_score: float,
        node_size: int = 300,
    ) -> None:
        """
        Adds a node to the directed graph.

        Parameters:
            node (Node): The node to add.
            root_score (float): The score of the root node.
            node_size (int): The base size of the node. The size will be scaled by the node's score.
        """
        node_color = Colors.BLUE.value if node.state.stance == 'PRO' else Colors.RED.value

        # If the node's score is None, it preceded the root node and should be assigned the root
        # node's score.
        if node.score is None:
            node.score = root_score

        node_size += 1000 * node.score  # Scale node size by score

        graph.add_node(
            node_for_adding=node.index, 
            color=node_color,
            size=node_size,
            label=node.score,
        )
    
    def add_edge(
        source: int,
        target: int,
        edge: Edge,
    ) -> None:
        """
        Adds an edge to the directed graph.

        Parameters:
            edge (Edge): The edge to add.
        """

        # Determine the color of the edge based on the stance of the source node
        color = Colors.BLUE.value if tree.nodes[source].state.stance == 'PRO' else Colors.RED.value

        graph.add_edge(
            u_of_edge=source,
            v_of_edge=target,
            label=wrap_text(edge.response, width=30),
            color=color,
        )

    for node in tree.nodes:
        add_node(node, root_score=tree.root.score)
    
    for source, target_edges in tree.edges.items():
        for target, edge in target_edges.items():
            add_edge(source, target, edge)

    return graph
    

def draw_graph(
    graph: nx.DiGraph,
    name: str = None,
) -> None:
    """
    Draws a directed graph.

    Parameters:
        graph (nx.DiGraph): The directed graph to draw.
        name: The name of the file to save the graph to.
    """
    pos = graphviz_layout(graph, prog='dot')
    plt.figure(
        figsize=(30, 20)
    )
    nx.draw(
        graph, 
        pos, 
        with_labels=True, 
        labels=nx.get_node_attributes(graph, 'label') , 
        node_size=[data['size'] for _, data in graph.nodes(data=True)], 
        node_color=[data['color'] for _, data in graph.nodes(data=True)], 
        edge_color=[data[2]['color'] for data in graph.edges(data=True)],
        font_color='white',
        font_size=12, 
        arrows=True,
        arrowsize=10,
    )
    nx.draw_networkx_edge_labels(
        graph, 
        pos, 
        edge_labels=nx.get_edge_attributes(graph, 'label'),
        font_size=6,
        label_pos=0.5,
    )    
    if name is not None:
        if not os.path.exists('outputs/graphs'):
            os.mkdir('outputs/graphs')
        plt.savefig(f'outputs/graphs/{name}.png', format='png', dpi=300)
    
    plt.title("Tree-of-Thought Visualization")
    plt.show()

