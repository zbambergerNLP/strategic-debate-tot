import dspy
import typing
from abstractions.generator.generator import (
    TOPIC_DESC, 
    STANCE_DESC, 
    CONVERSATION_DESC, 
)
from abstractions.generator.generator import ResponseParameters

####################
# Draft Generation #
# ##################  

# Inputs
PREVIOUS_DRAFTS_DESC = """
A list of argumentative drafts that the debator is aiming to improving upon in the current draft.
Each draft is an attempted improvement in quality and persuasiveness upon the previous draft.
""".strip()

# Outputs
SINGLE_TURN_DRAFT_RESPONSE_DESC = """
A persuasive argument that takes the specified stance towards the specified topic while adhering to the specified response length.
This improved argument is meant to improve on the weaknesses of the previous drafts, while maintaining their relative strengths.
""".strip()
MULTI_TURN_DRAFT_RESPONSE_DESC = """
A persuasive argument that takes the specified stance towards the specified topic while adhering to the specified response length,
and effectively addresses the main claims of the opponent in the conversation.
This improved argument is meant to improve on the weaknesses of the previous drafts, while maintaining their relative strengths.
""".strip()


class SingleTurnDraftBranchingSignature(dspy.Signature):
    """
    Generate an improved persuasive argument draft based on previous drafts, and according to the provided parameters.
    The generated argument should take the provided topic stance towards the topic. 
    
    Your argument should utilize rhetorical strategies such as ethos, logos, and pathos to enhance persuasiveness. 
    Support your claims with numerical, factual, and easily verifiable evidence to ensure credibility.
    You should not make any claims or statements that you cannot support with evidence or logical reasoning.
    Your argument should clarify complex concepts without relying on jargon or repetitive statements, making it accessible 
    to the audience. 
    While incorporating humor, metaphors, or other rhetorical devices can enhance engagement, they should be executed 
    appropriately to avoid undermining the argument's seriousness and effectiveness.

    Your new draft should demonstrate a clear improvement over previous versions by addressing identified weaknesses 
    and reinforcing existing strengths. It should adhere to the specified response length, language style, and communication 
    tone to maintain consistency and effectiveness in persuasion.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    response_parameters: ResponseParameters = dspy.InputField(desc="Parameters for generating the response.")
    previous_drafts: typing.List[str] = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    response: str = dspy.OutputField(desc=SINGLE_TURN_DRAFT_RESPONSE_DESC)


class MultiTurnDraftBranchingSignature(dspy.Signature):
    """
    Generate an improved persuasive argument draft within a multi-turn debate according to the provided parameters.

    The generated argument should support or oppose the given topic stance, effectively addressing or anticipating 
    counter-arguments from the opponent within the conversation context. It must employ rhetorical strategies such as ethos, 
    logos, and pathos to enhance persuasiveness. Claims should be backed by numerical, factual, and easily verifiable evidence 
    to ensure reliability and credibility. Do not make claims which you cannot support with evidence or logic accessible to the 
    average person. The argument should clarify complex concepts without the use of jargon or repetitive statements, making it 
    comprehensible to the audience.

    Incorporating humor, metaphors, or other rhetorical devices can enhance the argument's engagement, provided they are 
    executed appropriately to avoid detracting from the argument's seriousness and effectiveness. Additionally, the new draft 
    should maintain coherence with previous turns and strategically counter opposing points to strengthen persuasiveness.

    This improved argument aims to enhance the persuasiveness and coherence of the previous drafts by addressing their 
    weaknesses and reinforcing their strengths. It should adhere to the specified response length, language style, and 
    communication tone to ensure consistency and effectiveness throughout the debate.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    stance: str = dspy.InputField(desc=STANCE_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    response_parameters: ResponseParameters = dspy.InputField(desc="Parameters for generating the response.")
    previous_drafts: typing.List[str] = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    response: str = dspy.OutputField(desc=MULTI_TURN_DRAFT_RESPONSE_DESC)
