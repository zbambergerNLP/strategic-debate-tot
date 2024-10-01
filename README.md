# Strategic Debate:

Strategic Debate is a Python library that implements various prompt engineering techniques for creating and evaluating persuasive arguments.

### Tree of Thoughts

Our first release is a [DSPy](https://dspy-docs.vercel.app/)-based implementation of the [Tree of Thoughts](https://arxiv.org/abs/2305.10601) framework, meant for crafting persuasive arguments in debates. 

This library extends the original [Tree-of-Thoughts repository](https://github.com/princeton-nlp/tree-of-thought-llm) by explicitly catering to argument generation and evaluation. We support both single-turn arguments and multi-turn debates, while allowing for the same flexibility in evaluation strategies as the original framework.


![Tree-of-Thoughts](figures/tot_diagram.png)

### Features

- Tree of Thoughts implementation for strategic argument generation and evaluation
- Support for both single-turn and multi-turn debates
- Diverse reasoning types for iterative argument generation.
- Flexible evaluation strategies: scoring and voting.
- Customizable node selection strategies: greedy and weighted random
- Leverages either Beam Search or Monte-Carlo Tree Search for argument generation on top of the Tree of Thoughts framework.
- Visualization of the argument tree with [Graphviz](https://graphviz.org/).
- Integration with OpenAI's language models

## How Tree-of-Thoughts Works

### Beam-Search

Tree-of-Thoughts is a sequential prompting framework that is meant to evoke complex reasoning and planning capabilities in language models. We apply the Tree-of-Thoughts framework to the task of generating persuasive arguments (rather than tasks like `game-of-24`, `crosswords` or `creative writing`, which were used in the original paper).

A core idea in Tree-of-Thoughts is iterative expansion of a tree structure. In this tree structure, nodes represent "thoughts" ($z_t$), which are textual responses that incorporate intermediate/reasoning steps towards a final response. As part of the tree structure, a "thought" in layer `t` creates a set of "child thoughts" in layer `t+1`. This is known as the "generating" step (by a component known in the original paper as "Thought Generator", $G()$). 

Since unmediated branching can lead to a combinatorial explosion, we use an "evaluator" to assess the quality of each child thought after each expansion. The "evaluator" is a model that scores the quality of a thought based on its quality (a component known in the original paper as "States Evaluator", $V()$). Only the top scoring nodes in layer `t+1` are retained for further expansion. This is known as the "selection" step. As a result, we obtain an iterative-tree expansion algorithm that resembles beam search.

Our psuedocode for this is based on the "BFS" algoriothm proposed in [Tree-of-Thoughts](https://arxiv.org/pdf/2305.10601):

![Tree-of-Thoughts Pseudocode](figures/tot_bfs_pseudocode.png)

### Monte-Carlo Tree Search

In a MCTS step, the LLM can explores new states with a *rollout* (simulation) operation. Over the course of a rollout, the LLM generate a sequence of thoughts until reaching a terminal state, which is then scored. The score is then backpropagated to the root node, and the LLM can then select the next node to expand based on the scores of the nodes in the tree. MCTS balances between exploration and exploitation via the UCT algorithm, which guides recursive selection of child nodes.

![Monte Carlo Tree Search](figures/mcts_diagram.png)

We also include our pseudocode for the MCTS-based tree expansion algorithm: 

![Tree-of-Thoughts Pseudocode](figures/tot_mcts_pseudocode.png)


### Our Interpretation

We interpret the Tree-of-Thoughts framework as a way to generate persuasive arguments. Our library currently supports closed-source language models (e.g., `gpt-4o`, `gpt-4o-mini`) to serve as both the "Thought Generator" and the "States Evaluator".

Rather than viewing nodes in the tree as "thoughts", we view them as "argument states" (which can include context about the existing conversation, previous drafts, a plan, etc...). The "thoughts" in the original paper are meant to be intermediate steps towards a final response. Similarly, in our case, intermediate thoughts serve to develop a persuasive argument in a structured and iterative fashion.

Unlike the original paper, we think that thought generators and state evaluators come in three flavors: 

1. **First Step**: Optionally, perform some initial plan or action to prepare for iterative reasoning steps to come (e.g., generate a plan, which you will then execute over the course of the next steps).
2. **Subsequent Step**: This can occur multiple times, and involve expanding the existing line of reasoning towards a final response.
3. **Final Step**: Optionally, perform some final action to prepare for the final response (e.g., aggregate different ideas, or reflections on drafts in order to produce a stronger response).


### Reasoning Types

Tree-of-Thoughts suggests several ways to customize an instantiation of their method. The first, is to derive a way to break down a task into reasoning step. Next, one must decide how to generate and evaluate candidate intermediate steps. Finally, one must select a search algorithm to traverse various lines of reasoning, and identify the most promising one.

We came up with several reasoning types, each with their own thought generators and state evaluators. Each of our reasoning types is applicable with both Beam Search (referred to in the original ToT paper as BFS) and Monte-Carlo Tree Search (which was not explored in the ToT paper). We are releasing two reasoning types for now:

1. **Iterative Drafting**: This reasoning type involves iterating over drafts at each layer of the tree. The thought generator generates drafts based on the conversation history and the current draft. The state evaluator scores drafts based on their relevance to the conversation, their persuasiveness, and the extent to which the new draft improved on previous drafts. 

2. **Plan-and-Execute**: This reasoning type involves three distinct steps. First, the thought generator proposes several plans for the argument, which are in turn evaluated by the state evaluator. Next, the thought generator executes one step of the plan in each layer of the tree (while the state evaluator identifies which plan executions appear most promising). Finally, the thought generator generates a final draft based on the plan executions. 

We are considering additional reasoning types such as "Additive Reasoning" (adding one new relevant/helpful claim at each reasoning step) and "Repetitive Questioning" (identifying an ambiguity and resolving it in each reasoning step). We are looking for collaborators to develop these with us :D

## Installation

To install the Strategic Debate library, you must first clone or download the repository from GitHub:

```bash
git clone https://github.com/zbambergerNLP/tree-of-thoughts.git
```

Next, navigate to the local repository directory:

```bash
cd tree-of-thoughts
```

Create and activate a new Python virtual environment (we use Python 3.12):

```bash
python3.12 -m venv .env
source .env/bin/activate
```

Then, navigate to the repository directory and install the required dependencies:

```bash
pip install -r requirements.txt
```

### Optional Dependencies for Visualization
For visualization, you must install graphviz using the following command:

```bash
sudo apt-get install graphviz
```

Alternatively, on Mac, you can use the following command:

```bash
brew install graphviz
```

## Usage

Here's a basic example of how to use the Strategic Debate library:

```python
from strategic_debate import TreeOfThoughts, create_conversation_state
from utils import set_up_dspy

# Set up the environment
set_up_dspy(openai_key="your_openai_key_here")

# Initialize the conversation state
state = create_conversation_state(
    topic="The government should enforce regulation on AI technology.",
    stance="PRO",
    conversation_path="path/to/conversation.txt"
)

# Initialize the Tree of Thoughts module
tot = TreeOfThoughts(
    use_chain_of_thought=True,
    node_selection_strategy="greedy",
    evaluation_strategy="score"
)

# Generate the best argument
response = tot(
    state=state,
    depth=2,
    top_k=2,
    generation_temperature=0.7,
    judge_temperature=0.7,
    n_samples_generation=3,
    n_samples_judge=5,
    do_visualize_tree=True,
    do_save_tree=True,
    response_length="a few sentences"
)

print(f"Generated argument: {response}")
```

## Command-line Interface

The library also provides a command-line interface for easy experimentation:

```bash
python tree_of_thoughts.py --topic "AI regulation" --stance "PRO" --depth 2 --top_k 2
```

For a full list of command-line options and their full documentation, run:

```bash
python tree_of_thoughts.py --help
```

For a brief overview of the command-line options, see [Configuration](#configuration) section below.

## Configuration

The library offers various configuration options:

| Parameter | Description | Default Value |
| --- | --- | --- |
| `topic` | The topic of the debate. For example, "Collaboration is better than competition." | None |
| `stance` | The stance of the debator (a string). Must be one of [`"PRO"`, `"ANTI"`]. | None |
| `conversation_path` | The path to the conversation file. Each line in the file represents a message in the conversation. If no file is provided, conversation defaults to empty (`[]`). | None |
| `reasoning_type` | The type of reasoning to use. One of `iterative_drafting` or `plan_and_execute`, | `iterative_drafting` |
| `search_strategy` | The search strategy to use. Must be one of [`beam_search`, `monte_carlo`]. | `beam_search` |
| `do_pruning` | Whether to prune nodes (with the lowest scores) in intermediate layers of the tree. | `True` |
| `evaluation_strategy` | The strategy to use for evaluating the drafts. Must be one of [`"score"`, `"vote"`]. | `"score"` |
| `node_selection_strategy` | The strategy to use for selecting the nodes in the tree. Must be one of [`"greedy"`, `"sample"`]. | `"greedy"` |
| `generation_temperature` | The temperature to use when generating the drafts. A float between `0` and `1` | `0.7` |
| `judge_temperature` | The temperature to use when judging the drafts. A float between `0` and `1`. | `0.7` |
| `n_samples_generation` | The number of samples to generate for each response. | `3` |
| `n_samples_judge` | The number of samples to use when judging the drafts. | `5` |
| `top_k` | The number of best nodes to select at each step (according to the node selection strategy). | `2` |
| `depth` | The depth of the tree. The deeper the tree, the more reasoning is involved in producing candidate arguments. | `2` |
| `mcts_iterations` | The number of iterations to use in Monte-Carlo Tree Search. | `10` |
| `use_chain_of_thought` | Whether to use the chain of thought. | `True` |
| `max_tokens` | The maximum number of tokens to use when generating responses. | `1,000` |
| `model_name` | The name of the language model to use. One of [`gpt-4o`, `gpt-4o-mini`]. | `"gpt-4o-mini"` |
| `openai_key_path` | The path to the OpenAI key file. | None |
| `with_visualization` | Whether to visualize the tree of thought. Include this flag to include a visualization, and ommit it if you do not want to produce a visualization. | `False` |
| `with_saving` | Whether to save the tree of thought to a file. | `False` |
| `response_length` | The desired length of the response. Must be one of [`"a few sentences"`, `"a paragraph"`, `"a few paragraphs"`]. | `"a few sentences"` |
| `communication_tone` | The tone that the argument should take. One of [`logical`, `sarcastic`, `aggressive`]. | `logical` |
| `language_style` | The style of language to use. One of [`slang`, `casual`, `formal`]. | `casual` |



## Contributing

Contributions to the Strategic Debate library are welcome! 

Some of the ways you can contribute include:

1. Optimizing components of the Tree of Thoughts framework using [DSPy's optimizers](https://dspy-docs.vercel.app/docs/building-blocks/optimizers). Namely, we'd be interested in optimizing the efficacy of our debate judges using existing datasets for argument quality (e.g., [IBM-ArgQ-Rank-30kArgs](https://research.ibm.com/haifa/dept/vst/debating_data.shtml), [Change My View](https://convokit.cornell.edu/documentation/winning.html), [Persuasion For Good](https://convokit.cornell.edu/documentation/persuasionforgood.html), [Anthropic/persuasion](https://huggingface.co/datasets/Anthropic/persuasion), etc...).

2. Experimenting with the alternative forms of reasoning types (as discussed in the [Reasoning Types](#reasoning-types) section).

For additional information on how to contribute, please reach out to Zachary Bamberger at `zacharybamberger1@gmail.com`.

## License

This is an open-source project licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments

This library implements the Tree of Thoughts framework as presented in Yao et al. (2023).

The Co-Creators of this library:

- Zachary Bamberger (Lead)
- Ze'ev Sheleg
- Yosef Ben Yehuda

The Supervisors of this project:
- Dr. Amir Feder (Google, Columbia University)
- Dr. Ofra Amir (Technion -- Israel Institute of Technology)

If you have any questions or concerns, please feel free to reach out to Zachary Bamberger at `zacharybamberger1@gmail.com`

If you wish to build on this work, please reference this repository:

```bibtex
@software{strategic-debate,
  author = {Zachary Bamberger, Ze'ev Sheleg, Yosef Ben Yehuda, Amir Feder, Ofra Amir},
  title = {Strategic Debate: Tree of Thoughts},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/zbambergerNLP/tree-of-thoughts}
}