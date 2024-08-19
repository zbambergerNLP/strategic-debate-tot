import argparse
import constants

parser = argparse.ArgumentParser(
    description='Debate using Minimax with Tree-Of-Thought',
)

parser.add_argument(
    '--depth',
    type=int,
    default=2,
    help='The depth of the tree (i.e., the maximum number of turns in the debate).',
)

parser.add_argument(
    '--topic',
    type=str,
    default="The United States Government should adopt widespread use of street cameras to monitor public spaces.",
    help='The topic of the debate.',
)

parser.add_argument(
    '--stance',
    type=str,
    default="PRO",
    choices=["PRO", "ANTI"],
    help='The stance of the debator.',
)

parser.add_argument(
    '--conversation_path',
    type=str,
    default="data/conversations/street_cameras/example_0.txt",
    help="""
The path to the conversation file. Each line in the file represents a message in the conversation.
NOTE: The conversation path should be relevant to the topic you provide.
""",
)

parser.add_argument(
    '--model_name',
    type=str,
    default="gpt-4o-mini",
    help='The name of the language model to use.',
)

parser.add_argument(
    '--max_tokens',
    type=int,
    default=1_000,
    help='The maximum number of tokens to use when generating responses.',
)

parser.add_argument(
    '--openai_key_path',
    type=str,
    default='openai_key.txt',
    help='The path to the OpenAI key file.',
)

parser.add_argument(
    '--openai_key',
    type=str,
    default=None,
    help='The OpenAI key.',
)

parser.add_argument(
    '--with_visualization',
    action='store_true',
    help='Whether to visualize the tree of thought.',
    default=False,
)

parser.add_argument(
    '--save_tree',
    action='store_true',
    help='Whether to save the tree of thought.',
    default=False,
)

parser.add_argument(
    '--use_chain_of_thought',
    default=True,
    action='store_true',
    help='Whether to use the chain of thought.',
)

parser.add_argument(
    '--get_random_topic',
    action='store_true',
    help='Whether to get a random topic.',
    default=False,
)

parser.add_argument(
    '--top_k',
    type=int,
    default=2,
    help='The number of top-k arguments to consider in each expansion step of Tree-of-Thoughts.',
)

parser.add_argument(
    '--n_samples_generation',
    type=int,
    default=3,
    help='The number of samples to generate for each response.',
)

parser.add_argument(
    '--n_samples_judge',
    type=int,
    default=5,
    help='The number of samples to generate for each judgment of a persuasive argument.',
)

parser.add_argument(
    '--generation_temperature',
    type=float,
    default=0.7,
    help='The temperature to use when generating responses.',
)

parser.add_argument(
    '--response_length',
    type=str,
    default="a few sentences",
    help='The length of the generated argument.',
)

parser.add_argument(
    '--judge_temperature',
    type=float,
    default=0.7,
    help='The temperature to use when judging responses.',
)

# Strategies
parser.add_argument(
    '--node_selection_strategy',
    type=str,
    default=constants.NodeSelectionStrategy.GREEDY.value,
    help='The manner in which we select nodes to expand (or prune) in Tree-of-Thoughts.',
)

parser.add_argument(
    '--evaluation_strategy',
    type=str,
    default=constants.EvaluationStrategy.SCORE.value,
    help='The strategy for evaluating the quality of a response.',
)

