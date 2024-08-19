import typing
import dspy
from pydantic import (
    BaseModel,
    Field,
)


#########################
### Mixins for inputs ###
#########################

class TopicMixin(BaseModel):
    topic: str = Field(
        description="""
The topic of the argument. For example:
1. "The government should increase the number of street surveillance cameras"
2. "Climate change is a serious threat to humanity"
3. "Artificial intelligence research and development should be regulated"
""",
    )

class ArgumentMixin(BaseModel):
    argument: str = Field(
        description="""
The persuasive argument that the debator is making about the provided topic.
""".strip(),
    )

class MultiArgumentMixin(BaseModel):
    arguments: typing.Dict[int, str] = Field(
        description="""
A dictionary mapping argument indices to the persuasive arguments that the debators are making about the provided topic.
Each one of the arguments takes the provided stance towards the topic.
""".strip(),
    )

class PreviousDraftsMixin(BaseModel):
    previous_drafts: str = Field(
        description="""
The previous drafts of the argument. Each draft is a persuasive argument that the debator made about the provided topic.
previous drafts are formatted into a single string, where each draft has a header, and is followed by two newlines.
""".strip(),
    )

class MultiArgumentPreivousDraftsMixin(BaseModel):
    previous_drafts: typing.Dict[int, str] = Field(
        description="""
A dictionary mapping between an argument's index to the previous drafts of that argument. That is, the previous drafts
for `arguments[i]` are stored in `previous_drafts[i]`.
Each one of the arguments takes the provided stance towards the topic.
""".strip(),
    )

class ConversationMixin(BaseModel):
    conversation: typing.List[str] = Field(
        description="""
A conversation between two rival debators -- each holding an opposing stance (""PRO" vs. "ANTI"") about the provided topic.
The conversation is a list of messages, where each message is preceded by the message of the rival debator.
""".strip(),
    )

##########################
### Mixins for outputs ###
##########################

class ScoreMixin(BaseModel):
    score: float = Field(
        description="""
The floating-point score (between 0 and 1) of the argument. If the argument is very strong, the score should be close to 1.
If the argument is very weak, the score should be close to 0.
""".strip(),
    )

class VoteIndexMixin(BaseModel):
    index: int = Field(
        description="""
The index of the most persuasive argument among the provided list of arguments.
""".strip(),
    )

################
# Judge Inputs #
################

# Scoring
class ToTSingleTurnJudgeInput(TopicMixin, ArgumentMixin):
    """The input to a LLM Judge, who is tasked with evaluating the quality of a persuasive argument"""

class ToTSingleTurnJudgeWithDraftsInput(TopicMixin, ArgumentMixin, PreviousDraftsMixin):
    """The input to a LLM Judge, who is tasked with evaluating the quality of an argument 
    (in the context of previous drafts of that argument)"""

class ToTMultiTurnJudgeInput(TopicMixin, ConversationMixin, ArgumentMixin):
    """The input to a LLM Judge, who is taked with evaluating the quality of an argument in a multi-turn debate."""

class ToTMultiTurnJudgeWithDraftsInput(TopicMixin, ConversationMixin, ArgumentMixin, PreviousDraftsMixin):
    """The input to a LLM Judge, who is taked with evaluating the quality of an argument in a multi-turn debate.
    This input includes the previous drafts of the arguments in the 'previous_drafts' field.
    """

# Voting
class ToTJudgeVoteSingleTurnInput(TopicMixin, MultiArgumentMixin):
    """The input to a LLM Judge, who is tasked with voting for the most persuasive arguments among multiple candidates."""

class ToTJudgeVoteSingleTurnWithDraftsInput(TopicMixin, MultiArgumentMixin, MultiArgumentPreivousDraftsMixin):
    """The input to a LLM Judge, who is tasked with voting for the most persuasive arguments among multiple candidates.
    Each of the candidate arguments is accompanied by the previous drafts of that argument.
    """

class ToTJudgeVoteMultiTurnInput(TopicMixin, MultiArgumentMixin, ConversationMixin):
    """
    The input to a LLM Judge, who is tasked with voting for the most persuasive arguments among multiple candidates.
    Each of these arguments is a possible response in a multi-turn debate versus a rival debator with an opposing stance
    """

class ToTJudgeVoteMultiTurnWithDraftsInput(TopicMixin, MultiArgumentMixin, ConversationMixin, MultiArgumentPreivousDraftsMixin):
    """
    The input to a LLM Judge, who is tasked with voting for the most persuasive arguments among multiple candidates.
    Each of these arguments is a possible response in a multi-turn debate versus a rival debator with an opposing stance.
    Each of the candidate arguments is accompanied by the previous drafts of that argument.
    """

#################
# Judge Outputs #
#################

class ToTSingleTurnJudgeOutput(ScoreMixin):
    """
    A floating-point score (between 0 and 1) which quantifies the quality of a written argument.
    If the argument is very strong, the score should be close to 1. If the argument is very weak, the score should be
    close to 0.
    """

class ToTJudgeVoteOutput(VoteIndexMixin):
    """
    The index of the most persuasive argument among the provided list of arguments.
    The indices are the keys of the 'arguments' dictionary in the input.
    """

class ToTMultiTurnJudgeOutput(ToTSingleTurnJudgeOutput):
    """
    A floating-point score (between 0 and 1) which quantifies the quality of a written argument in the context of the debate.
    If the argument that the debator made was very strong, the score should be close to 1. If the argument was very weak,
    the score should be close to 0.
    """

##################################
# DSPy Signatures for LLM Judges #
##################################

class ToTSingleTurnJudgeSingature(dspy.Signature):
    """
    Judge the quality of a persuasive argument with the provided stance towards the provided topic. 

    You are an objective and fair judge, and must evaluate strictly the quality and persuasiveness of the argument.
    Your task is to determine a floating-point score for the provided argument, where "1" indicates that the argument
    was very well-written, and was very persuasive, and a "0" indicates that the argument was very poorly written, and
    was not persuasive.
    """
    judge_input: ToTSingleTurnJudgeInput = dspy.InputField()
    judge_output: ToTSingleTurnJudgeOutput = dspy.OutputField()


class ToTSingleTurnJudgeWithDraftsSignature(dspy.Signature):
    """
    Judge the quality of a persuasive argument with the provided stance towards the provided topic. 

    You are an objective and fair judge, and must evaluate strictly the quality and persuasiveness of the argument.
    Your task is to determine a floating-point score for the provided argument, where "1" indicates that the argument
    was very well-written, and was very persuasive, and a "0" indicates that the argument was very poorly written, and
    was not persuasive.
    """
    judge_input: ToTSingleTurnJudgeWithDraftsInput = dspy.InputField()
    judge_output: ToTSingleTurnJudgeOutput = dspy.OutputField()

class ToTMultiTurnJudgeSignature(dspy.Signature):
    """
    Judge the quality and persuasiveness of an argument in the context of a multi-turn debate.
    If the argument is very strong, the score should be close to 1. If the argument is very weak, the score should be close to 0.
    """
    judge_input: ToTMultiTurnJudgeInput = dspy.InputField()
    judge_output: ToTMultiTurnJudgeOutput = dspy.OutputField()

class ToTMultiTurnJudgeWithDraftsSignature(dspy.Signature):
    """
    Judge the quality and persuasiveness of the provided argument in the context of a multi-turn debate.
    If the argument is very strong, the score should be close to 1. If the argument is very weak, the score should be close to 0.
    """
    judge_input: ToTMultiTurnJudgeWithDraftsInput = dspy.InputField()
    judge_output: ToTMultiTurnJudgeOutput = dspy.OutputField()

class ToTJudgeVoteSingleTurnSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates.
    Each candidate argument takes the provided stance towards the provided topic.
    Your vote is represented as an integer index, which specifies the most persuasive argument among the provided list of arguments.
    """
    judge_input: ToTJudgeVoteSingleTurnInput = dspy.InputField()
    judge_output: ToTJudgeVoteOutput = dspy.OutputField()


class ToTJudgeVoteMultiTurnSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates.
    Each candidate argument takes the provided stance towards the provided topic.
    Your vote is represented as an integer index, which specifies the most persuasive argument among the provided list of arguments.
    """
    judge_input: ToTJudgeVoteMultiTurnInput = dspy.InputField()
    judge_output: ToTJudgeVoteOutput = dspy.OutputField()


class ToTJudgeVoteSingleTurnWithDraftsSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates.
    Each candidate argument takes the provided stance towards the provided topic, and builds on previous argumentative drafts.
    Your vote is represented as an integer index, which specifies the most persuasive argument among the provided list of arguments.
    """
    judge_input: ToTJudgeVoteSingleTurnWithDraftsInput = dspy.InputField()
    judge_output: ToTJudgeVoteOutput = dspy.OutputField()


class ToTJudgeVoteMultiTurnWithDraftsSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates.
    Each candidate argument takes the provided stance towards the provided topic.
    Your vote is represented as an integer index, which specifies the most persuasive argument among the provided list of arguments.
    """
    judge_input: ToTJudgeVoteMultiTurnWithDraftsInput = dspy.InputField()
    judge_output: ToTJudgeVoteOutput = dspy.OutputField()
