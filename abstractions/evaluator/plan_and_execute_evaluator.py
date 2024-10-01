import dspy
import typing

from abstractions.evaluator.evaluator import (
    TOPIC_DESC,
    CONVERSATION_DESC,
)

# Inputs
PLAN_DESC = "A step-by-step plan for how to persuade the audience to take the debator's stance on the topic."
EXECUTION_DESC = "The execution of the plan so far."
PLAN_STEP_DESC = "A single step in the plan for how to persuade the audience to take the debator's stance on the topic."
CLAIM_TO_MAKE_DESC = "The claim that the debator should make as part of the current step of the plan."

# Outputs
PLAN_SCORE_DESC = "The floating-point score (between 0 and 1) which corresponds with the quality of the plan."
EXECUTION_SCORE_DESC = "The floating-point score (between 0 and 1) which corresponds with the quality of the execution of the plan." 

##################
### Signatures ###
##################

class SingleTurnScoreWithPlanSignature(dspy.Signature):
    """
    Evaluate the quality and effectiveness of a step-by-step plan for constructing a persuasive argument on a given topic.

    As an objective and fair judge, assess the plan based on its clarity, logical structure, comprehensiveness, and strategic 
    alignment with the goal of persuading an audience member. 

    A strong plan should outline clear, actionable steps that build upon each other to form a cohesive strategy for argumentation.
    A weak plan is vague, disorganized, lacks essential components, or includes redundant steps. 
    Assess whether each step is purposeful and contributes to the overall objective of persuading the audience.

    Provide a floating-point score between 0 and 1, where 1 indicates a strong plan and 0 indicates a weak plan.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    plan: typing.List[str] = dspy.InputField(desc=PLAN_DESC)
    score: float = dspy.OutputField(desc=PLAN_SCORE_DESC)

class SingleTurnScoreWithPlanExecutionSignature(dspy.Signature):
    """
    Evaluate the quality and persuasiveness of an argumentative claim within the context of a predefined plan.

    As an objective and fair judge, assess the execution step based on its clarity, relevance, logical coherence, and alignment 
    with the current step of the plan. 

    A strong execution step should effectively advance the plan by making a clear and relevant claim that supports the overall 
    argument. 
    A weak execution step makes a claim that is unclear, irrelevant, lacks logical structure, or deviates from the planned strategy.
    Assess whether the new claim integrates smoothly with previous claims and contributes meaningfully to persuading the audience.

    Provide a floating-point score between 0 and 1, where 1 indicates a strong execution and 0 indicates a weak execution.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    claim_plan: str = dspy.InputField(desc=PLAN_STEP_DESC)
    new_claim: str = dspy.InputField(desc=CLAIM_TO_MAKE_DESC)
    claims_so_far: typing.List[str] = dspy.InputField(desc=EXECUTION_DESC)
    score: float = dspy.OutputField(desc=EXECUTION_SCORE_DESC)


class MultiTurnScoreWithPlanSignature(dspy.Signature):
    """
    Evaluate the quality of a step-by-step plan for constructing a persuasive argument within a multi-turn debate.
    
    As an objective and fair judge, assess the plan based on its clarity, logical structure, comprehensiveness, and strategic 
    alignment with the goal of persuading an audience member over multiple turns of debate.
    
    A strong plan should account for the dynamic nature of debates, including anticipating counter-arguments and adapting 
    strategies as the conversation progresses. 
    A weak plan is rigid, fails to address potential opposition, lacks essential components, or includes redundant steps. 
    Assess whether each step is purposeful and contributes to the overall objective of persuading the audience.

    Provide a floating-point score between 0 and 1, where 1 indicates a strong plan and 0 indicates a weak plan.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    plan: typing.List[str] = dspy.InputField(desc=PLAN_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    score: float = dspy.OutputField(desc=PLAN_SCORE_DESC)


class MultiTurnScoreWithPlanExecutionSignature(dspy.Signature):
    """
    Assess the quality and persuasiveness of an argumentative claim within the context of a predefined plan in a multi-turn debate.

    As an objective and fair judge, evaluate the claim's clarity, relevance, logical coherence, and alignment with the 
    current step of the plan. This new claim should ideally build on the steps of the plan executed so far, as well as the 
    ongoing conversation. 
    
    A strong execution step in a debate context should consider previous arguments by the opponent (as well as arguments they 
    are likely to make in the future), and directly accomplishes the current step of the plan while building upon previously 
    executed steps.
    A weak execution step makes a claim that is unclear, irrelevant, lacks logical structure, deviates from the plan, or fails to 
    engage with the ongoing conversation. 
    Assess whether the new claim integrates smoothly with previous claims and contributes meaningfully to persuading the audience.

    Provide a floating-point score between 0 and 1, where 1 indicates a strong execution and 0 indicates a weak execution.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    claim_plan: str = dspy.InputField(desc=PLAN_STEP_DESC)
    new_claim: str = dspy.InputField(desc=CLAIM_TO_MAKE_DESC)
    claims_so_far: typing.List[str] = dspy.InputField(desc=EXECUTION_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    score: float = dspy.OutputField(desc=EXECUTION_SCORE_DESC)
