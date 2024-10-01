from pydantic import BaseModel, Field


class TreeOfThoughtsParameters(BaseModel):
    """
    Parameters for the Tree of Thoughts module, which define the behavior of the search over reasoning steps.
    """
    top_k: int = Field(
        ...,
        description="""
The number of nodes to select at each reasoning step (layer) in the tree. 
Prioritizes nodes with the highest scores from intermediate reasoning evaluations.
""".strip(),
    )
    n_samples_judge: int = Field(
        ...,
        description="""
The number of samples to use when judging the quality of intermediate thoughts/reasoning steps.
The final score of a node is the average of the scores obtained from the ensemble of judges.
""".strip(),
    )
    n_samples_generation: int = Field(
        ...,
        description="""
The number of samples to generate for each reasoning step. The number of child nodes for each parent node.
""".strip(),
    )
    judge_temperature: float = Field(
        ...,
        description="""
The temperature to use when judging the quality of intermediate thoughts/reasoning steps.
A float between 0 and 1.
""".strip(),
    )
    generation_temperature: float = Field(
        ...,
        description="""
The temperature to use when generating intermediate thoughts/reasoning steps and responses.
A float between 0 and 1.
""".strip(),
    )
    depth: int = Field(
        ...,
        description="""
The number of intermediate thoughts/reasoning steps to take in the tree. 
Each step (layer) increases the depth of the tree by one.
""".strip(),
    )

class MonteCarloTreeOfThoughtParameters(BaseModel):
    """
    Parameters for the Monte Carlo Tree of Thoughts module, which define the behavior of the search over reasoning steps.
    """

    n_samples_judge: int = Field(
        ...,
        description="""
The number of samples to use when judging the quality of intermediate thoughts/reasoning steps.
The final score of a node is the average of the scores obtained from the ensemble of judges.
""".strip(),
    )
    n_samples_generation: int = Field(
        ...,
        description="""
The number of samples to generate for each reasoning step. The number of child nodes for each parent node.
""".strip(),
    )
    judge_temperature: float = Field(
        ...,
        description="""
The temperature to use when judging the quality of intermediate thoughts/reasoning steps. A float between 0 and 1.
""".strip(),
    )
    generation_temperature: float = Field(
        ...,
        description="""
The temperature to use when generating intermediate thoughts/reasoning steps and responses. A float between 0 and 1.
""".strip(),
    )
    rollout_depth: int = Field(
        ...,
        description="""
The depth of the rollout.This is the maxiumum number of intermediate thoughts/reasoning steps to generate as part of 
the rollout before scoring. 
""".strip(),
    )
    monte_carlo_iterations: int = Field(
        ...,
        description="""
The number of iterations to perform the monte carlo tree search. Iterations consist of node selection, 
expansion, simulation, and backpropagation.
""".strip(),
    )