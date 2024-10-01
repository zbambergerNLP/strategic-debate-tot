import dspy
import typing
from abstractions.generator.generator import (
    TOPIC_DESC,
    STANCE_DESC,
    CONVERSATION_DESC,
    ResponseParameters
)


###################
# Plan Generation #
###################

# Input
NUMBER_OF_CLAIMS_DESC = "The desired number of steps in the plan for writing the argument."

# Output
PLAN_RESPONSE = "The plan for how to persuade the audience to take the debator's stance on the topic."

class SingleTurnPlanningBranchingSignature(dspy.Signature):
    """
    Generate a structured, step-by-step plan for persuading the audience to adopt your stance on the provided topic.
    
    The plan should consist of the specified number of steps, each representing a distinct claim, statement, or
    argumentative strategy.

    If the stance is 'PRO', each step should advocate for the topic's claim; if 'ANTI', each step should argue against it. 
    
    Each claim within the plan should aim to strengthen your argument or boost your credibility. 
    The steps should be logically ordered to build a cohesive and persuasive strategy, ensuring that each claim effectively 
    contributes to the overall objective of persuading the audience. 
    Avoid vague or redundant steps, and ensure that each step is concise, clear, and actionable to facilitate detailed expansion 
    in subsequent execution phases.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=TOPIC_DESC)
    number_of_claims: int = dspy.InputField(desc=NUMBER_OF_CLAIMS_DESC)
    response: typing.List[str] = dspy.OutputField(desc=PLAN_RESPONSE)

class MultiTurnPlanningBranchingSignature(dspy.Signature):
    """
    Generate a structured, step-by-step plan for persuading the audience to adopt your stance on the provided topic.

    The plan should consist of the specified number of steps, each representing a distinct claim, statement, or 
    argumentative strategy.
    
    If the stance is 'PRO', each step should advocate for the topic's claim; if 'ANTI', each step should argue against it. 
    
    Each claim within the plan should aim to strengthen your argument, weaken your opponent's argument, or boost your 
    credibility. The steps should be logically ordered to build a cohesive and persuasive strategy, ensuring that each 
    claim effectively contributes to the overall objective of persuading the audience. 
    Additionally, consider the ongoing conversation with the opponent to ensure that the plan is adaptive and responsive to 
    counter-arguments. 
    Avoid vague or redundant steps, and ensure that each step is concise, clear, and actionable to facilitate detailed expansion 
    in subsequent execution phases.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    number_of_claims: int = dspy.InputField(desc=NUMBER_OF_CLAIMS_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    response: typing.List[str] = dspy.OutputField(desc=PLAN_RESPONSE)


#############################
# Plan Execution Generation # 
#############################

# Input
PLAN_DESC = "A step-by-step plan for how to persuade the audience to take the debator's stance on the topic."
EXISTING_CLAIMS_DESC = "The claims that the debator has made so far (executions of previous steps in the plan)."
CLAIM_TO_MAKE_DESC = "The claim that the debator should make as part of the current step of the plan."

# Output
PLAN_EXECUTION_RESPONSE = "The persuasive argument that the debator generated to argue for the provided claim."


class SingleTurnPlanExecutionBranchingSignature(dspy.Signature):
    """
    Execute a step in your plan to persuade the audience to adopt your stance on the provided topic.

    You are provided with a specific claim to make as part of your argument, along with the claims you have made so far. 
    Generate a persuasive argument that supports, expands, or implements the claim, and which aligns with your overall stance. 
    
    Utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. Support your claims with 
    numerical, factual, and easily verifiable evidence to ensure credibility. Avoid making statements that you cannot
    support with evidence or logical reasoning.

    Clarify complex concepts without relying on jargon or repetitive statements. While incorporating humor, metaphors, or other 
    rhetorical devices can enhance engagement, ensure they are executed appropriately to avoid undermining the argument's 
    effectiveness. Adhere to the specified response length, language style, and communication tone. Avoid vague or unsupported 
    statements, inappropriate language that contradicts the specified style, overly complex sentences, and redundant or 
    repetitive points.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    existing_claims: typing.List[str] = dspy.InputField(desc=EXISTING_CLAIMS_DESC)
    claim_to_make: str = dspy.InputField(desc=CLAIM_TO_MAKE_DESC)
    response: str = dspy.OutputField(desc=PLAN_EXECUTION_RESPONSE)


class MultiTurnPlanExecutionBranchingSignature(dspy.Signature):
    """
    Execute a step in your plan to persuade the audience of your stance in an ongoing debate about the provided topic.

    You are provided with a specific claim to make as part of your argument, the claims you have made so far, and the ongoing 
    conversation with a rival debater. Generate a persuasive argument that supports, expands, or implements the claim, and which 
    aligns with your overall stance. 

    Utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. Support your claims with numerical, 
    factual, and easily verifiable evidence to ensure credibility. Do not make any claims or statements which you cannot support 
    with evidence or logic. Clarify complex concepts without relying on jargon or repetitive statements. 
    While incorporating humor, metaphors, or other rhetorical devices can enhance engagement, ensure they are executed 
    appropriately to avoid undermining the argument's and effectiveness.

    The generated argument should build upon previous claims, addressing any identified weaknesses and reinforcing 
    existing strengths. Additionally, it should respond to or anticipate counter-arguments from the opponent based on 
    the ongoing conversation. Adhere to the specified response length, language style, and communication tone to maintain 
    consistency and effectiveness in persuasion. Avoid vague or unsupported statements, inappropriate language that 
    contradicts the specified style, overly complex sentences, redundant or repetitive points, and deviating from 
    addressing the opponent's claims within the conversation.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    existing_claims: typing.List[str] = dspy.InputField(desc=EXISTING_CLAIMS_DESC)
    claim_to_make: str = dspy.InputField(desc=CLAIM_TO_MAKE_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    response: str = dspy.OutputField(desc=PLAN_EXECUTION_RESPONSE)


#############################
# Final Response Generation #
#############################

FINAL_RESPONSE_DESC = """The final persuasive argument that the debator generated to argue for their stance on the topic.
This argument should be coherent and persuasive, and build on the claims and arguments from the plan and plan execution steps.
The argument must adhere to the specified response length.
""".strip()

class SingleTurnPlanAndExecuteResponseSignature(dspy.Signature):
    """
    Generate a persuasive argument that synthesizes your plan and its execution to support your stance on the topic.

    You have access to a structured plan outlining your key claims and the corresponding executed arguments that elaborate on 
    each claim. Consolidate these elements into a coherent and persuasive argument that effectively supports your stance on the 
    topic. 
    
    Utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. 
    Ensure that each claim is supported by numerical, factual, and easily verifiable evidence to maintain credibility. 
    Clarify any complex concepts without relying on jargon or repetitive statements to ensure accessibility to the audience. 
    While incorporating humor, metaphors, or other rhetorical devices can enhance engagement, they should be executed 
    appropriately to avoid undermining the argument's seriousness and effectiveness.

    The final argument should seamlessly integrate the strengths of your plan and executions, addressing any previously 
    identified weaknesses and reinforcing existing strengths. 
    Adhere to the specified response length, language style, and communication tone to maintain consistency and effectiveness in 
    persuasion. Avoid vague or unsupported statements, inappropriate language that contradicts the specified style, overly 
    complex sentences, and redundant or repetitive points.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    plan: typing.List[str] = dspy.InputField(desc=PLAN_DESC)
    plan_execution: typing.List[str] = dspy.InputField(desc=EXISTING_CLAIMS_DESC)
    response_parameters: ResponseParameters = dspy.InputField(desc="Parameters for generating a persuasive argument.")
    response: str = dspy.OutputField(desc=FINAL_RESPONSE_DESC)
