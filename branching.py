import typing
from pydantic import BaseModel, Field
import dspy


##############
### Mixins ###
##############

class ConversationMixin(BaseModel):
    conversation: typing.List[str] = Field(
        description="""
A conversation between two rival debators -- each holding an opposing stance (PRO vs. ANTI) about the provided topic.
The conversation is a list of messages, where each message is preceded by the message of the rival debator.
That is, the most recent message in the conversation was written by the rival debator (i.e., the debator with the opposing stance).
""".strip(),
    )

class PreviousDraftsMixin(BaseModel):
    previous_drafts: typing.List[str] = Field(
        description="""
A list of argumentative drafts that the debator is aiming to improving upon in the current draft.
Each draft is an attempted improvement in quality and persuasiveness upon the previous draft.
""".strip(),
    )

####################
# Branching Inputs #
####################

class ToTSingleTurnResponseBranchingInput(BaseModel):
    """
    The input to an LLM debator tasked with generating a persuasive argument.
    """
    topic: str = Field(
        description="The topic of the argument.",
    )
    stance: str = Field(
        description="""
The stance of the LLM debator's argument towards the topic. 'PRO', if the argument is in favor of the topic's claim,
or 'ANTI', if the argument is against the topic's claim.
""".strip(),
    )
    length: str = Field(
        default="a few sentences",
        description="The length of the argument to generate (e.g., 'one sentence', 'a few sentences', 'one paragraph', etc...).",
    )

class ToTMultiTurnResponseBranchingInput(ToTSingleTurnResponseBranchingInput, ConversationMixin):
    """
    The input to an LLM debator tasked with generating a persuasive argument in an ongoing debate with a rival debator.

    The debate is expressed as a 'conversation' (list of string messages) between two debators with opposing stances towards
    a given topic. Each message in the conversation is opposed by the preceding message in the conversation (i.e., the
    debators take turns responding).
    """


class ToTSingleTurnDraftBranchingInput(ToTSingleTurnResponseBranchingInput, PreviousDraftsMixin):
    """
    The input to an LLM debator tasked with generating a persuasive argument which improves upon previous drafts.
    """

class ToTMultiTurnDraftBranchingInput(ToTSingleTurnDraftBranchingInput, ConversationMixin):
    """
    The input to an LLM assistant tasked with generating a persuasive argument in an ongoing debate.
    The debate is expressed as a 'conversation' (list of string messages) between two debators with opposing stances towards
    a given topic. Each message in the conversation is opposed by the preceding message in the conversation (i.e., the
    debators take turns responding).
    Each draft is meant to improve upon the previous drafts the debator has considered (in the context of the conversation).
    """

#####################
# Branching Outputs #
#####################

class ToTResponseBranchingOutput(BaseModel):
    """
    A persuasive argument that the LLM debator produced.
    """
    response: str = Field(
        description="The persuasive argument that the LLM debator generated.",
    )

########################
# Branching Signatures #
########################


class ToTSingleTurnResponseBranchingSignature(dspy.Signature):
    """
    Generate persuasive arguments that takes the specified stance towards the specified topic.
    """
    branching_input: ToTSingleTurnResponseBranchingInput = dspy.InputField()
    branching_output: ToTResponseBranchingOutput = dspy.OutputField()

class ToTSingleTurnDraftBranchingSignature(dspy.Signature):
    """
    Generate persuasive arguments that takes the specified stance towards the specified topic.
    Your argument should improve upon the previous drafts you have considered.
    """
    branching_input: ToTSingleTurnDraftBranchingInput = dspy.InputField()
    branching_output: ToTResponseBranchingOutput = dspy.OutputField()

class ToTMultiTurnDraftBranchingSignature(dspy.Signature):
    """
    Generate persuasive arguments in an ongoing debate.
    Your argument should take the specified stance towards the specified topic.
    You should effectively address the main claims of the opponent in the conversation.
    Your argument should improve upon the previous drafts you have considered.
    """
    branching_input: ToTMultiTurnDraftBranchingInput = dspy.InputField()
    branching_output: ToTResponseBranchingOutput = dspy.OutputField()

class ToTMultiTurnResponseBranchingSignature(dspy.Signature):
    """
    Generate a persuasive argument in an ongoing debate.
    Your argument should take the specified stance towards the specified topic.
    You should effectively address the main claims of the opponent in the conversation.
    """
    branching_input: ToTMultiTurnResponseBranchingInput = dspy.InputField()
    branching_output: ToTResponseBranchingOutput = dspy.OutputField()
