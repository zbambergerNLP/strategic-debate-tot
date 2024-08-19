# Strategic Debate:

Strategic Debate is a Python library that implements various prompt engineering techniques for creating and evaluating persuasive arguments.

## Tree-of-Thoughts

### Overview

Our first release is a [DSPy](https://dspy-docs.vercel.app/)-based implementation of [Tree of Thoughts](https://arxiv.org/abs/2305.10601) framework for crafting persuasive arguments in debates. This library extends the original [Tree-of-Thoughts repository](https://github.com/princeton-nlp/tree-of-thought-llm) by explicitly catering to argument generation and evaluation. We support both single-turn arguments and multi-turn debates, while allowing for the same flexibility in evaluation strategies as the original framework.

### Features

- Tree of Thoughts implementation for strategic argument generation and evaluation
- Support for both single-turn and multi-turn debates
- Flexible evaluation strategies: scoring and voting
- Customizable node selection strategies: greedy and weighted random
- Visualization of the argument tree with [Graphviz](https://graphviz.org/).
- Integration with OpenAI's language models

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

## Configuration

The library offers various configuration options:

- `depth`: The number of steps to take in the tree
- `top_k`: The number of best nodes to select at each layer
- `generation_temperature`: Temperature for argument generation
- `judge_temperature`: Temperature for argument evaluation
- `n_samples_generation`: Number of samples to generate for each argument
- `n_samples_judge`: Number of samples to use when judging arguments
- `do_visualize_tree`: Whether to visualize the tree of thoughts
- `do_save_tree`: Whether to save the tree of thoughts
- `response_length`: Desired length of the response

## Command-line Interface

The library also provides a command-line interface for easy experimentation:

```bash
python tree_of_thoughts.py --topic "AI regulation" --stance "PRO" --depth 2 --top_k 2
```

For a full list of command-line options, run:

```bash
python tree_of_thoughts.py --help
```

## Contributing

Contributions to the Strategic Debate library are welcome! Please refer to our contributing guidelines for more information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

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