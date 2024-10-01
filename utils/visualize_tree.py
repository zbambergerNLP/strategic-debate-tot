import os
import typing
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from matplotlib import pyplot as plt
import textwrap

from abstractions.tree.tree import Edge, Node, Tree

BLUE = 'blue'
RED = 'red'

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
        ):
        """
        Adds a node to the directed graph.

        Parameters:
            node (Node): The node to add.
            root_score (float): The score of the root node.
            node_size (int): The base size of the node. The size will be scaled by the node's score.
        """
        node_color = BLUE if node.state.stance == 'PRO' else RED

        # If the node's score is None, it preceded the root node and should be assigned the root
        # node's score.
        if node.score is None or node.score == float('-inf'):  
            node.score = root_score

        node_size += 1000 * node.score  if node.score else 300 # Scale node size by score. Default to size 300 if score is None.

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
        ):
        """
        Adds an edge to the directed graph.

        Parameters:
            edge (Edge): The edge to add.
        """

        # Determine the color of the edge based on the stance of the source node
        color = BLUE if tree.nodes[source].state.stance == 'PRO' else RED

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
    ):
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
