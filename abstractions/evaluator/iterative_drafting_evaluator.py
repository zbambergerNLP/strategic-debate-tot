import typing
import dspy
from abstractions.evaluator.evaluator import (
    TOPIC_DESC,
    ARGUMENT_DESC,
    VOTE_CANDIDATES_DESC,
    SCORE_DESC,
    VOTE_INDEX_DESC,
    CONVERSATION_DESC,
)


PREVIOUS_DRAFTS_DESC = """
A list of argumentative drafts that the debator is aiming to improving upon in the new draft.
Each draft is an attempted improvement in quality and persuasiveness upon previous drafts.
""".strip()


class SingleTurnScoreWithDraftsSignature(dspy.Signature):
    """
    Evaluate the quality and improvement of a new persuasive argument draft with the provided stance towards a given topic.

    As an objective and fair judge, assess the new draft based on its clarity, logical structure, comprehensiveness, and 
    strategic alignment with the goal of persuading an audience member with reasonable general knowledge. 
    Additionally, determine whether the new draft demonstrates a clear improvement over the previous drafts in terms of quality 
    and persuasiveness.

    A strong draft presents clear, actionable enhancements that significantly improve the argument's persuasiveness and 
    coherence compared to earlier versions. A weak draft is vague, disorganized, lacks essential components, includes redundant 
    elements, or fails to improve upon previous drafts.

    Provide a floating-point score between 0 and 1, where 1 indicates a strong and improved draft and 0 indicates a weak 
    or regressive draft.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    previous_drafts: str = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    argument: str = dspy.InputField(desc=ARGUMENT_DESC)
    score: float = dspy.OutputField(desc=SCORE_DESC)

class MultiTurnScoreWithDraftsSignature(dspy.Signature):
    """
    Evaluate the quality and improvement of a new persuasive argument draft within the context of a multi-turn debate.

    As an objective and fair judge, assess the new draft based on its clarity, logical structure, comprehensiveness, and 
    strategic alignment with the goal of persuading an audience member over multiple turns of debate. Additionally, 
    determine whether the new draft demonstrates progressive enhancements over previous drafts to increase persuasiveness.

    A strong draft accounts for the dynamic nature of debates, including addressing previous arguments and anticipating future 
    counter-arguments, while showing continuous improvement in persuasiveness and coherence over previously attempted drafts. 
    A weak draft is rigid, fails to address potential opposition, lacks essential components, includes redundant elements, or 
    does not show improvement over previous iterations.
    
    Provide a floating-point score between 0 and 1, where 1 indicates a strong and improved draft and 0 indicates a weak 
    or regressive draft.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    previous_drafts: str = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    argument: str = dspy.InputField(desc=ARGUMENT_DESC)
    score: float = dspy.OutputField(desc=SCORE_DESC)

class SingleTurnVoteWithDraftsSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates, considering their iterative improvements over previous drafts.

    Each candidate argument takes the provided stance towards the topic and builds upon previous argumentative drafts to enhance 
    quality and persuasiveness.

    A strong argument presents clear, actionable enhancements that significantly improve the argument's persuasiveness and 
    coherence compared to earlier versions. 
    A weak argument is vague, disorganized, lacks essential components, includes redundant elements, or fails to improve upon 
    previous drafts.

    Your vote is represented as an integer index, specifying the most persuasive argument among the provided list of arguments.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    arguments: typing.Dict[int, str] = dspy.InputField(desc=VOTE_CANDIDATES_DESC)
    previous_drafts: typing.Dict[int, str] = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    index: int = dspy.OutputField(desc=VOTE_INDEX_DESC)

class MultiTurnVoteWithDraftsSignature(dspy.Signature):
    """
    Vote for the most persuasive argument among multiple candidates within the context of a multi-turn debate.

    Each candidate argument takes the provided stance towards the topic and builds upon previous argumentative drafts to enhance 
    quality and persuasiveness.

    A argument draft accounts for the dynamic nature of debates, including addressing previous arguments and anticipating future 
    counter-arguments, while showing continuous improvement in persuasiveness and coherence over previously attempted drafts. 
    A weak argument is rigid, fails to address potential opposition, lacks essential components, includes redundant elements, or 
    does not show improvement over previous iterations.

    Your vote is represented as an integer index, specifying the most persuasive argument among the provided list of arguments.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    arguments: typing.Dict[int, str] = dspy.InputField(desc=VOTE_CANDIDATES_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    previous_drafts: typing.Dict[int, str] = dspy.InputField(desc=PREVIOUS_DRAFTS_DESC)
    index: int = dspy.OutputField(desc=VOTE_INDEX_DESC)
