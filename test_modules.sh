# Run each of the modules in each possible setting in order to validate the
# robustness of the modules.

######################
# Iterative drafting #
######################

# No pruning
python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score" 

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "vote"

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score"

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "vote"

# Pruning
python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score" \
    --do_pruning

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "vote" \
    --do_pruning

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score" \
    --do_pruning

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "beam_search" \
    --evaluation_strategy "vote" \
    --do_pruning

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score" \
    --do_pruning

python iterative_drafting_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "iterative_drafting" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score" \
    --do_pruning

####################
# Plan and execute #
####################

# No pruning
python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score"

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score"

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score"

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score"


# Pruning
python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score" \
    --do_pruning

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "beam_search" \
    --evaluation_strategy "score" \
    --do_pruning

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_0.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score"

python plan_and_execute_tree_of_thoughts.py \
    --max_tokens 1000 \
    --conversation_path "data/conversations/street_cameras/example_1.txt" \
    --reasoning_type "plan_and_execute" \
    --search_strategy "monte_carlo" \
    --evaluation_strategy "score"
