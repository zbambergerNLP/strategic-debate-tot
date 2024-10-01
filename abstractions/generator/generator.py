import typing
import dspy
from pydantic import BaseModel, Field

#######################
# Argument Generation #
####################### 


# Inputs
TOPIC_DESC = "The topic of the argument."
STANCE_DESC = "The stance to take when writing an argument about the topic (either PRO or ANTI)."
CONVERSATION_DESC = "A conversation between two rival debators, each holding an opposing stance (PRO vs. ANTI) about the topic."
RESPONSE_LENGTH_DESC = "The length of the argument to generate (e.g., 'one sentence', 'a few sentences', 'one paragraph', etc...)."
LANGUAGE_STYLE_DESC = "The degree of formality for the argument. One of 'slang', 'casual', or 'formal'."
COMMUNICATION_TONE_DESC = "The tone of the argument to generate. One of 'logical', 'sarcastic', or 'aggressive'."
RESPONSE_PARAMETERS_DESC = "Parameters for generating a persuasive argument."

# Outputs
SINGLE_TURN_RESPONSE_DESC = """
A persuasive argument that takes the specified stance towards the specified topic while adhering to the specified response length.
""".strip()
MULTI_TURN_RESPONSE_DESC = """
A persuasive argument that takes the specified stance towards the specified topic while adhering to the specified response length, 
and effectively addresses the main (and/or anticipated) claims of the opponent in the conversation.
""".strip()

class ResponseParameters(BaseModel):
    response_length: str = Field(desc=RESPONSE_LENGTH_DESC, default="a few sentences")
    communication_tone: str = Field(desc=COMMUNICATION_TONE_DESC, default="logical")
    language_style: str = Field(desc=LANGUAGE_STYLE_DESC, default="casual")
    
class SingleTurnResponseBranchingSignature(dspy.Signature):
    """
    Generate a persuasive argument according to the provided parameters.
    The argument should take the provided stance towards the given topic.
    Utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. 
    Support claims with numerical, factual, and easily verifiable evidence. 
    Clarify complex concepts without relying on jargon or repetitive statements. 
    While using humor, metaphors, or other rhetorical methods can be beneficial, ensure they are executed appropriately to avoid 
    undermining the argument. 
    Adhere to the specified response length, language style, and communication tone when generating the argument.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    response_parameters: ResponseParameters = dspy.InputField(desc=RESPONSE_PARAMETERS_DESC)
    response: str = dspy.OutputField(desc=SINGLE_TURN_RESPONSE_DESC)


class MultiTurnResponseBranchingSignature(dspy.Signature):
    """
    Generate a persuasive argument within an ongoing multi-turn debate according to the provided parameters.
    Your argument should take the provided stance towards the given topic, and effectively address or anticipate 
    counter-arguments from an opponent with an opposing stance.
    Utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. 
    Support claims with numerical, factual, and easily verifiable evidence. 
    Clarify complex concepts without relying on jargon or repetitive statements. 
    While using humor, metaphors, or other rhetorical methods can be beneficial, ensure they are executed appropriately to avoid 
    undermining the argument. 
    Additionally, maintain coherence with previous turns and strategically counter opposing points to enhance persuasiveness. 
    Adhere to the specified response length, language style, and communication tone when generating the argument.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    response_parameters: ResponseParameters = dspy.InputField(desc=RESPONSE_PARAMETERS_DESC)
    response: str = dspy.OutputField(response=MULTI_TURN_RESPONSE_DESC)
