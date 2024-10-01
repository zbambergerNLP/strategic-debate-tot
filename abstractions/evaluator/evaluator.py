import typing
import dspy


##################################
# DSPy Signatures for LLM Judges #
##################################

# Inputs
TOPIC_DESC = "The topic of the argument."
CONVERSATION_DESC = "A conversation between two rival debators, each holding an opposing stance (PRO vs. ANTI) about the topic."
ARGUMENT_DESC = "The persuasive argument that the debator is making about the provided topic."
VOTE_CANDIDATES_DESC = "A dictionary mapping indices to candidate arguments about the provided topic."

# Outputs
SCORE_DESC = "The floating-point score (between 0 and 1) which corresponds with the quality of the argument."
VOTE_INDEX_DESC = "The index of the most persuasive argument among the provided list of arguments."

class SingleTurnScoreSingature(dspy.Signature):
    """
    Assess the effectiveness and persuasiveness of an argument with the provided stance towards a given topic.
    
    As an impartial and objective judge, you must rigorously evaluate the argument's clarity, logical coherence, use of 
    evidence, and rhetorical strategies. Assign a floating-point score between 0 and 1, where 1 signifies an exceptionally 
    well-crafted and highly persuasive argument, and 0 denotes a poorly constructed and unconvincing one. 
    
    Ensure your evaluation accounts for potential edge cases, such as ambiguous language, lack of supporting evidence, 
    logical fallacies, or biased reasoning, and maintain consistency and fairness in your scoring process.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    argument: str = dspy.InputField(desc=ARGUMENT_DESC)
    score: float = dspy.OutputField(desc=SCORE_DESC)


class MultiTurnScoreSignature(dspy.Signature):
    """
    Evaluate the quality and persuasiveness of an argument within the context of a multi-turn debate on a given topic.

    As an objective and impartial judge, evaluate the argument's ability to persuade an audience member with reasonable
    general knowledge. Consider the argument's clarity, logical coherence, use of evidence, and rhetorical strategies.
    Additionally, assess how well the argument addresses and counters the opponent's previous points, anticipates potential 
    rebuttals, and maintains relevance throughout the conversation. 
    
    Assign a floating-point score between 0 and 1, where 1 indicates an exceptionally strong argument that convincingly 
    persuades the audience and effectively engages with the debate, while 0 signifies a weak, unconvincing argument that 
    fails to address the opponent's stance. 
    
    Ensure your evaluation accounts for edge cases such as fragmented dialogue, irrelevant or repetitive points, logical 
    fallacies, biased reasoning, and insufficient evidence, maintaining consistency and fairness in your scoring process.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    argument: str = dspy.InputField(desc=ARGUMENT_DESC)
    score: float = dspy.OutputField(desc=SCORE_DESC)


class SingleTurnVoteSignature(dspy.Signature):
    """
    Select the most persuasive argument from a set of candidate arguments with the same stance towards a given topic.

    As an objective and impartial judge, evaluate each candidate argument based on clarity, logical coherence, use of evidence, 
    and rhetorical effectiveness. You should favor the argument that is most likely to persuade an audience member with reasonable 
    general knowledge. Consider factors such as the strength of the argument's claims, the relevance and credibility of 
    supporting evidence, and the overall persuasiveness of the rhetoric. 
    
    Assign your vote by selecting the integer index corresponding to the most compelling argument from the provided dictionary
    of candidates.
    
    Ensure your selection process accounts for potential edge cases, including closely matched arguments, redundant or 
    repetitive points, logical fallacies, biased reasoning, and insufficient evidence, while maintaining consistency and 
    fairness in your voting criteria.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    arguments: typing.Dict[int, str] = dspy.InputField(desc=ARGUMENT_DESC)
    index: int = dspy.OutputField(desc=VOTE_INDEX_DESC)


class MultiTurnVoteSignature(dspy.Signature):
    """
    Select the most persuasive argument from a set of candidate arguments within a multi-turn debate context.
    
    As an objective and impartial judge, evaluate each candidate argument based on its clarity, logical coherence, use of 
    evidence, and rhetorical effectiveness. Additionally, consider how well each argument engages with the ongoing conversation 
    by addressing and countering the opponent's previous points, anticipating potential rebuttals, and maintaining relevance to 
    the topic. 
    
    Your goal is to determine which argument is most likely to persuade an audience member with general knowledge, taking into 
    account the strength of the claims, the credibility of supporting evidence, and the effectiveness of counter-arguments. 
    
    Assign your vote by selecting the integer index corresponding to the most compelling argument from the 
    provided dictionary of candidates. 
    
    Ensure your selection process accounts for potential edge cases, such as closely matched arguments, redundant or 
    repetitive points, logical fallacies, biased reasoning, and insufficient evidence, while maintaining consistency and 
    fairness in your voting criteria.
    """
    topic: str = dspy.InputField(desc=TOPIC_DESC)
    arguments: typing.Dict[int, str] = dspy.InputField(desc=ARGUMENT_DESC)
    conversation: typing.List[str] = dspy.InputField(desc=CONVERSATION_DESC)
    index: int = dspy.OutputField(desc=VOTE_INDEX_DESC)
