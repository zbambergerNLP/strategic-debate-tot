from logging import Logger
import typing
from abstractions.generator.generator import ResponseParameters
from abstractions.tree.tree import State, Node, Edge, Tree, MonteCarloNode
from pydantic import Field

class PlanAndExecuteState(State):
    """Represents a planning step (draft) by a debator when writing a persuasive argument for a debate (on the given topic)."""
    claims_to_generate: int = Field(
        ...,
        description="The number claims to generate. Each claim is a single step in the plan.",
    )
    plan: typing.List[str] = Field(
        default_factory=list,
        description="A list of brief argumentative claims that the debator plans to make in the debate.",
    )
    claims_so_far: typing.List[str] = Field(
        default_factory=list,
        description="A list of argumentative claims that the debator has made so far in the debate (in accordance with the plan).",
    )

    @property
    def reasoning_steps(self) -> typing.List[str]:
        """Retrieves reasoning steps in the form of claims generated so far. Each claim is an execution of a step in the plan."""
        return self.claims_so_far

    def state_to_generator_input(self, response_parameters: ResponseParameters) -> typing.Dict[str, str | typing.List[str]]:
        """Converts the state to an input for the generator."""
        if len(self.conversation) == 0:
            if self.plan:
                return {
                    "topic": self.topic,
                    "stance": self.stance,
                    "existing_claims": self.claims_so_far[:-1] if self.claims_so_far else [],
                    "claim_to_make": self.plan[len(self.claims_so_far) - 1],
                }
            else:
                return {
                    "topic": self.topic, "stance": self.stance, "number_of_claims": self.claims_to_generate}
        else:       # If the conversation is not empty (i.e., there are previous exchanges we need to consider)
            if self.plan:
                return {
                    "topic": self.topic,
                    "stance": self.stance,
                    "existing_claims": self.claims_so_far[:-1] if self.claims_so_far else [],
                    "claim_to_make": self.plan[len(self.claims_so_far) - 1],
                    "conversation": self.conversation,
                }
            else:
                return {
                    "topic": self.topic,
                    "stance": self.stance,
                    "number_of_claims": self.claims_to_generate,
                    "conversation": self.conversation,
                }
    
    def state_to_evaluator_input(self) -> typing.Dict[str, str | typing.List[str]]:
        """Converts the state to an input for the evaluator."""
        if len(self.conversation) == 0:
            if self.claims_so_far:
                plan_index = len(self.claims_so_far) - 1
                return {
                    "claim_plan": self.plan[plan_index],
                    "new_claim": self.claims_so_far[-1],
                    "claims_so_far": self.claims_so_far[:-1],
                    "topic": self.topic,
                }
            else:
                return {"plan": self.plan, "topic": self.topic}
        else:
            if self.claims_so_far:
                plan_index = len(self.claims_so_far) - 1
                return {
                    "claim_plan": self.plan[plan_index],
                    "new_claim": self.claims_so_far[-1],
                    "claims_so_far": self.claims_so_far[:-1],
                    "topic": self.topic,
                    "conversation": self.conversation,
                }
            else:
                return {"plan": self.plan, "topic": self.topic, "conversation": self.conversation}
            
class PlanAndExecuteNode(Node):
    """Represents a node in a tree that is used for planning and executing an argument in a debate."""
    state: PlanAndExecuteState = Field(
        ...,
        description="""
A representation of a node's state in a Tree-of-Thought framework.
A state contains the topic of the debate, the stance of the debator towards the topic, and conversation so far.
Notably, this state includes a plan (consisting of a list of claims) that the debator is aiming to make in the debate, 
and the claims the debator has made so far.
""",
    )
    score: float = Field(
        default=0,
        description="""
The score of the node between 0 and 1. The higher the score, the higher the quality of the reasoning until that node.
This applies to both the planning and execution phases.
""".strip(),
    )

class MCTSPlanAndExecuteNode(PlanAndExecuteNode, MonteCarloNode):
    """Represents a node in a Monte Carlo Tree Search tree that is used for planning and executing an argument in a debate."""


class PlanAndExecuteTree(Tree):

    def _plan_to_string(self, plan: typing.List[str]) -> str:
        """
        Converts a plan to a string.
        """
        return str("\n".join(plan))

    def __init__(self, state: State, max_depth: int):
        self.claims_to_generate = max_depth
        super().__init__(state)

    def _add_node_to_tree_from_existing_conversation(
        self, 
        message_index: int, 
        topic: str, 
        stance: str, 
        conversation: typing.List[str],
    ) -> Node:
        return Node(
            index=message_index,
            state=PlanAndExecuteState(
                topic=topic,
                stance=stance,
                conversation=conversation[:message_index],
                claims_to_generate=self.claims_to_generate,
                plan=[],
                claims_so_far=[],
            ),
            parent_id=message_index - 1 if message_index > 0 else None,
            children_ids=[message_index + 1],
            score=0,
        )
    
    def _initialize_root(
        self,
        state: State,
    ) -> Node:
        return Node(
            index=len(state.conversation),
            state=PlanAndExecuteState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                claims_to_generate=self.claims_to_generate,
                plan=[],
                claims_so_far=[],
            ),
            parent_id=len(state.conversation) - 1 if len(state.conversation) > 0 else None,
            score=0,
        )
    
    def create_child_node(
        self,
        index: int,
        state: PlanAndExecuteState,
        output: typing.Union[str, typing.List[str]],
    ):
        if isinstance(output, str):  # Execution phase
            return Node(
                index=index,
                state=PlanAndExecuteState(
                    topic=state.topic,
                    stance=state.stance,
                    conversation=state.conversation,
                    claims_to_generate=state.claims_to_generate,
                    plan=state.plan,                                    # Plan remains static
                    claims_so_far=state.claims_so_far + [output],       # Output (claim) is the execution of a plan step
                ),
            )
        else:  # Planning phase
            return Node(
                index=index,
                state=PlanAndExecuteState(
                    topic=state.topic,
                    stance=state.stance,
                    conversation=state.conversation,
                    claims_to_generate=state.claims_to_generate,
                    plan=output,                                        # Output (list of claims) is the plan
                    claims_so_far=state.claims_so_far,                  # Claims are not yet produced
                ),
            )
    
    def add_child_node(
        self, 
        parent_node: Node, 
        child_node: Node, 
        response: str, 
        expansion_reasoning: str,
    ):
        edge = Edge(
            source=parent_node.index,
            target=child_node.index,
            response=self._plan_to_string(response) if isinstance(response, list) else response,
            reasoning=expansion_reasoning,
        )
        parent_node.children_ids.append(child_node.index)
        child_node.parent_id = parent_node.index
        self.nodes.append(child_node)
        self.add_edge(
            source_index=parent_node.index,
            target_index=child_node.index,
            edge=edge,
        )

    def log_tree(self, logger: Logger):
        for node in self.nodes:
            logger.info(f"\n\nNode {(node.index)}")
            if(node.parent_id is not None):
                if node.state.claims_so_far:
                    logger.info(f'\tClaims so far: {node.state.claims_so_far[:-1]}')
                    logger.info(f'\tNewest claim: {node.state.claims_so_far[-1]}')
                else:
                    logger.info(f'\tPlan: {node.state.plan}')
            logger.info(f'\tScore: {node.score}')
            logger.info('\tChildren:')
            for child_index in node.children_ids:
                logger.info(
                    f'\n\t- Node #{child_index} [score = {self.nodes[child_index].score}]: '
                    f'{self.edges[node.index][child_index].response}'
                )
                # Print the reasoning associated with the response if it exists
                if self.edges[node.index][child_index].reasoning:
                    logger.info(f'\tReasoning: {self.edges[node.index][child_index].reasoning}')
    

class MCTSPlanAndExecuteTree(PlanAndExecuteTree):

    def _add_node_to_tree_from_existing_conversation(
        self,
        message_index: int, 
        topic: str, 
        stance: str, 
        conversation: typing.List[str],
    ) -> Node:
        return MCTSPlanAndExecuteNode(
            index=message_index,
            state=PlanAndExecuteState(
                topic=topic,
                stance=stance,
                conversation=conversation[:message_index],
                claims_to_generate=self.claims_to_generate,
                plan=[],
                claims_so_far=[],
            ),
            parent_id=message_index - 1 if message_index > 0 else None,
            children_ids=[message_index + 1],
            score=0,
            visits=0,
        )
    
    def _initialize_root(
        self,
        state: State,
    ) -> Node:
        return MCTSPlanAndExecuteNode(
            index=len(state.conversation),
            state=PlanAndExecuteState(
                topic=state.topic,
                stance=state.stance,
                conversation=state.conversation,
                claims_to_generate=self.claims_to_generate,
                plan=[],
                claims_so_far=[],
            ),
            parent_id=len(state.conversation) - 1 if len(state.conversation) > 0 else None,
            score=0,
            visits=0,
        )
    
    def create_child_node(
        self,
        index: int,
        state: PlanAndExecuteState,
        output: typing.Union[str, typing.List[str]],
    ):
        if isinstance(output, str):
            return MCTSPlanAndExecuteNode(
                index=index,
                state=PlanAndExecuteState(
                    topic=state.topic,
                    stance=state.stance,
                    conversation=state.conversation,
                    claims_to_generate=state.claims_to_generate,
                    plan=state.plan,
                    claims_so_far=state.claims_so_far + [output],
                ),
                visits=0,
            )
        else:
            return MCTSPlanAndExecuteNode(
                index=index,
                state=PlanAndExecuteState(
                    topic=state.topic,
                    stance=state.stance,
                    conversation=state.conversation,
                    claims_to_generate=state.claims_to_generate,
                    plan=output,
                    claims_so_far=state.claims_so_far,
                ),
                visits=0,
            )
