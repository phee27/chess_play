# Fine-tuning Llama 3.1 8B Instruct for Chess with GRPO

This project fine-tunes the Llama 3.1 8B Instruct model to play chess using a 
Stockfish dataset and Group Relative Policy Optimization (GRPO).

## 🚀 Setup Instructions

### 1. Environment Setup

#### Install Miniconda
# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
# Create and activate conda environment
conda create -n chess-rl python=3.10
conda activate chess-rl

### 2. Install Dependencies

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
or
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129 
(Please use this if you use new generation GPU like RTX 4090)
# Install from requirements file
pip install -r requirements.txt

###  3. Authentication setup

# Login to Hugging Face (required for Llama model access)
huggingface-cli login (then enter your token)
#### Weights & Biases 
wandb login (then enter your token)

### 4. Install Stockfish Engine

# Update package manager and install Stockfish
apt-get update && apt-get install -y stockfish
# Find Stockfish installation path
find /usr -name "stockfish" 2>/dev/null
# Add Stockfish to PATH (adjust path based on find results)
export PATH="/usr/games:$PATH"
# Test Stockfish installation
echo -e "position startpos\neval\nquit" | stockfish

### 5. Download base model for fine tuning
python test_model.py

### 6. Download data
mkdir -p data
wget -O data/stockfish_evaluations.jsonl "https://huggingface.co/datasets/bingbangboom/stockfish-evaluations/resolve/main/stockfish_evaluations.jsonl"

### 7. Download my fine tunned weights
mkdir -p models
hf download phee27/chess-grpo-llama-8b --include "checkpoint-500/*" --local-dir ./models


## 🏃‍♂️ Running the Training
# Start training with logging
python src/train.py 2>&1 | tee out.logs


chess_play/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── data/                        # Dataset files (stockfish_evaluations.jsonl)
├── models/                      # Saved model checkpoints and outputs
├── src/                         # Source code directory
│   ├── data_processing.py      # Data preprocessing script for preprocess stockfish_evaluations.jsonl 
│   │                            (split the dataset into train/val/test )
│   ├── train.py                # Main training script with GRPO implementation
│   └── utils.py                # Helper functions and utilities (reward function and helper functions)
├── test_load_trained.py        # Script to load and test trained/fine-tunned models
├── test_model.py               # Script to download llama 3.1 8B Instruct base model
├── test_model-Qwen.py          # Script to download Qwen base model
├── wandb/                      # Weights & Biases experiment tracking logs
├── out.logs                    # General training output logs


#############################################################################
### Architecture Design

## Model Selection
When selecting the base model for fine-tuning, I evaluated candidates against the assignment requirements:
Requirements:

- Fewer than 10 billion parameters
- Instruction-tuned for general tasks
- Not specifically fine-tuned for chess

Evaluation Criteria:
Given that chess is a complex reasoning task requiring the model to follow instructions precisely, 
I established three key criteria for selection:

1. Format Adherence: Model can consistently respond in the specified format, enabling reliable extraction of moves and reasoning
2. Chess Understanding: Model demonstrates sound reasoning about chess positions, rules, and concepts when 
provided with appropriate context 
(note: excellence is not expected from general models, but basic comprehension of chess definitions and rules is essential)
3. Model Capacity: Largest possible model under the 10B parameter constraint to maximize reasoning capability

Final Selection:
After testing multiple candidates, I compared Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct across all criteria. 
Based on initial evaluations, 
Llama-3.1-8B-Instruct consistently outperformed Qwen2.5 in:

Instruction following and format consistency
Chess reasoning quality and rule comprehension
Overall task performance

Therefore, Llama-3.1-8B-Instruct was selected as the base model for fine-tuning.


## Algorithm Selection
GRPO (Group Relative Policy Optimization) was chosen over alternatives like PPO, DPO, or supervised 
fine-tuning for several key reasons. Unlike supervised learning which only teaches the model to mimic 
Stockfish's moves without understanding position quality, GRPO or other RLHF algorithm can teach model to learn 
the best move and the reasoning associated with those moves. Since chessis a complex reasoning task, 
this trait makes RLHF superior than supervised learning in this task

As per this task, GRPO is better than PPO for several reason. First GRPO is simpler since we do not need 
to train the value function. Compared to PPO and DPO, GRPO enables more exploration of the move policy 
space while maintaining stability, which suits chess well since there are often multiple reasonable 
moves in a position that the model should consider rather than converging too quickly to a single 
"correct" move. Furthermore, the group-wise comparison of GRPO also enables our policy to learn 
which move is the best among several ones which is the important decision in chess.

Therefore, GRPO is the most suitable for this task.


## Reward Function Design

The reward function is designed with a multi-tier structure that balances response completeness, 
move quality assessment, and optimal move promotion. The base reward structure starts with a foundation 
of 1.0 points for valid data processing, followed by an additional 1.0 points for predicting a legal move. 
This ensures the model is rewarded for satisfying basic requirements - providing parseable responses with 
valid chess moves - without allowing these completeness rewards to dominate the learning signal. 

The legal move bonus encourages the model to understand chess rules while maintaining a relatively modest 
contribution to the total reward.

The core positional reward uses a ranking-based approach rather than raw Stockfish evaluations to provide 
fair and normalized scoring across all positions. Since Stockfish evaluation scales can vary a lot between 
positions/engines and have wide range, using raw evaluation differences would create inconsistent learning signals. 
Instead, the ranking reward assigns 10 points for the best move in single-move positions, and scales linearly 
from 10 points (best move) down to 1 point (worst evaluated move) based on the move's rank among all 
legal alternatives. The mathematical formula for ranking reward is:

ranking_reward = {
    10.0                                                    if total_moves = 1
    10 - ((predicted_rank - 1) × 9.0) / (total_moves - 1)   if total_moves > 1
}
(Please not that the board evaluation is calculated from White perspective so large positive value means white is at advantage)

Finally, a substantial 10-point bonus is awarded when the model predicts the exact best move. 
This will create a strong incentive for optimal play to make sure that reward for deciding on the best move 
(main goal of this assignment) is much larger than a good move. 

The complete reward function can be expressed as:

total_reward = base_reward + legal_move_bonus + ranking_reward + ground_truth_bonus

Where:
base_reward = 1.0 (for valid data)
legal_move_bonus = 1.0 (if predicted move is legal, 0 otherwise)  
ranking_reward = calculated using formula above
ground_truth_bonus = 10.0 (if predicted_move = ground_truth_move (best_move), 0 otherwise)


## Data Processing

The data processing pipeline implements a train/validation/test split strategy. Following the assignment requirements, 
the last 1,000 rows from the raw Stockfish evaluations dataset are reserved as the test set with minimal filtering 
(only valid JSON parsing). For the training and validation data, the pipeline randomly samples 15,000 positions from 
all remaining rows (everything before the test set) to ensure diverse position coverage while maintaining 
computational feasibility. In the ideal situation where more computational resources are provided, more positions 
will be used in training. Quality filters are applied to the training data including a minimum depth threshold (default 10 moves) 
and position validation to ensure legal chess positions and moves. The random sampling approach prevents bias 
that could occur from simply taking the first N rows and promote variety of chess position model can learn. 
The training data is further split with 5% reserved for validation (approximately 750 positions), creating 
a final distribution of ~14,250 training samples, 750 validation samples, and 1,000 test samples. Each position 
is converted into a structured prompt that includes the FEN notation, current turn, legal moves list, and visual board 
representation to provide the model with comprehensive context for move prediction.


## Key decision

1. Prompt:
# Prompt Template:
Example prompt: 

Please Analyze the following position and provide your best move.

Position (FEN): 1k1r4/p2P1p2/1pQ3q1/8/6P1/P4P1p/1P6/1K1R4 w - -
Turn: White
Legal moves to chosse from: [list of legal moves]

CURRENT BOARD:
{board}

Your response MUST follow this EXACT format, without any extra text, PGN, or game history. 

Example Format of your response:

Best move: [The single best move in Standard Algebraic Notation, e.g., Ra7]
<reasoning>
[Explain the strategic and tactical reasons for your move. Address the opponent's threats and your own opportunities.]
</reasoning>
END OF RESPONSE


Your response for the given position:

Example prompt: 
Please Analyze the following position and provide your best move.

Position (FEN): 1k1r4/p2P1p2/1pQ3q1/8/6P1/P4P1p/1P6/1K1R4 w - -
Turn: White
Legal moves to chosse from: Ka1, Ka2, Kc1, Qc2, Qe4, Qxg6, Rd3

CURRENT BOARD:
. k . r . . . .
p . . P . p . .
. p Q . . . q .
. . . . . . . .
. . . . . . P .
P . . . . P . p
. P . . . . . .
. K . R . . . .

Your response MUST follow this EXACT format, without any extra text, PGN, or game history. 

# Example Format of your response:

Best move: [The single best move in Standard Algebraic Notation, e.g., Ra7]
<reasoning>
[Explain the strategic and tactical reasons for your move. Address the opponent's threats and your own opportunities.]
</reasoning>
END OF RESPONSE


Your response for the given position:


2. Reasoning:
Since Chess is a complex strategic reasoning task, I write a prompt to model to request explanations of 
tactical considerations, threat assessment, and strategic opportunities in addition to best move prediction. 
While the reward function directly evaluates only move quality through Stockfish-based scoring, this reasoning 
component serves as an implicit learning mechanism - the model learns to associate good strategic analysis 
with high-reward moves through the reinforcement learning process. Observation across training epochs demonstrates 
improvement in reasoning quality, with later iterations producing tactical analyses that align more with sound
chess principles, suggesting the model successfully internalizes the connection between strategic understanding 
and optimal play.

3. Providing legal moves:
I tested the chess task with foundational models like Claude, ChatGPT, and Gemini. Despite their massive size, 
these models struggle to produce legal moves, let alone best moves. Therefore, I added the legal moves to each prompt 
during fine-tuning. This tactic simplifies the problem and allows the LLM to focus on selecting the best move 
from the available legal moves. As discussed in the results section, the performance of fine-tuned models in 
selecting legal and best moves is significantly better.


4. Providing textual boards
Instead of sending the raw FEN notation in the prompt, I used functions from the python-chess 
library to generate a visual 2D chess board representation and included it in the prompt. 
This approach enables the model to better understand the position by providing a clear visual 
layout of piece placement, making it easier to identify available moves and assess the relative positions 
of all pieces on the board.


### Benchmark Performance.

We will evaluate the performance of our fine-tuned chess model against the base Llama-3.1-8B-Instruct 
model using three metrics on a test set consisting of the final 1,000 samples from stockfish_evaluations.jsonl:

# 1. Move Quality (MSE)
Objective: Measure how close the predicted moves are to optimal play in terms of positional evaluation.
Method:

Generate a move prediction for each test position (FEN)
Use Stockfish to evaluate the resulting position after applying the predicted move
Compare this value to the value after applying the ground truth best move
Calculate the mean squared error between these values

Formula:
MSE = (1/N) × Σ(eval_predicted - eval_best)²
Where:

eval_predicted = Stockfish evaluation of position after predicted move
eval_best = Stockfish evaluation of position after ground truth best move
N = number of test samples

Interpretation: Lower MSE indicates the model's moves lead to positions with evaluations closer to the optimal moves.

2. Best Move Accuracy
Objective: Measure exact match rate with Stockfish's top choice.
Method: Count cases where the model's predicted move exactly matches the ground truth best move from the dataset.
Formula:
Accuracy = (Number of exact matches / Total predictions) × 100%
Interpretation: Higher accuracy indicates better alignment with engine recommendations.

3. Legal Move Rate
Objective: Ensure the model generates valid chess moves.
Method: Parse each predicted move and verify it's legal in the given position using chess rules validation.
Formula:
Legal Move Rate = (Number of legal moves / Total predictions) × 100%
Interpretation: Values below 100% suggest the model sometimes generates invalid moves despite provided legal move list.

Note: All evaluations will be calculated at a corresponding depth from the dataset to ensure fair comparison 
between predicted and ground truth moves.


## Model Performance Comparison

| Metric | Base Model (Llama-3.1-8B-Instruct) | Fine-tuned Model (GRPO Checkpoint-500) | Improvement |
|--------|-------------------------------------|----------------------------------------|-------------|
| **Move Accuracy** | 5.6% | 7.7% | +2.1% (+37.5%) |
| **Legal Move Rate** | 98.7% | 99.9% | +1.1% (+1.2%) |
| **Stockfish MSE** | 44.64 | 31.67 | -12.97 (-29.0%) |
| **Test Samples** | 1,000 | 1,000 | - |

### Key Findings

- **Move Quality**: The fine-tuned model shows significant improvement in move quality, 
with a 29% reduction in Stockfish MSE, indicating moves that lead to positions much closer to optimal evaluations.

- **Accuracy**: Best move accuracy improved by 37.5% relative to the base model

- **Legal Moves**: The illegal moves from fine-tuned is 1/1000 compared to 13/1000 in base model

### Analysis

The GRPO fine-tuning shows clear improvements across all metrics. The most substantial gain is in move quality 
(MSE reduction), suggesting the model learned to evaluate positions more effectively through the Stockfish-based 
reward function. It also shows improvements in exact move accuracy and legal move generation, indicating that
 the training successfully enhanced the model's chess playing capabilities.

### Future Improvement
- Fine-tune the learning rate and hyperparameters: Due to time and resource limitations, 
I did not have sufficient time to experiment with many hyperparameters, including the learning rate, 
which could significantly affect training stability. In the future, I would like to explore this 
area to optimize training and improve the quality of the final model.
- Transition to numerical reward function: Currently, I am using a rank-based reward system to place 
strong emphasis on finding the best move. I would like to explore numerical rewards such as 
MSE between predicted moves and best moves' Stockfish evaluations, as well as clipping techniques. 
Furthermore, I want to add more terms to the reward function, such as entropy, to promote diversity 
in the model's responses. Observing the current model reveals that the base model usually prefers 
moving pawns over other pieces.
- Annotate data with more sophisticated reasoning from foundation models: Models under 10B parameters 
do not have reasoning capabilities on the same level as foundation models. In the future, 
I want to annotate the given dataset with sound reasoning from GPT-4 or Claude to explain 
why the best move is superior to alternatives. This annotated reasoning can be incorporated into 
the reward function and serve as a valuable signal for model learning.


### Scaling Up
- Scaling with computational resources: With additional computational resources, I would expand the training 
dataset beyond 15,000 samples to include more diverse chess positions for the model to learn from. 
Despite this relatively small sample size, the model showed substantial improvements over the base model 
across all evaluation metrics. A larger and more diverse training set would likely yield further performance gains.
- Extended training and convergence analysis: I would also extend training for additional epochs to identify 
the true convergence point of the model and achieve better overall training stability. Currently, the training 
was limited by computational constraints, but longer training could reveal whether the model can achieve 
even better chess performance.
- Enhanced reasoning through annotation: As mentioned previously, I plan to annotate the training dataset with 
detailed reasoning from foundation models like GPT-4 or Claude. These annotations would explain why specific 
moves are superior to alternatives, providing richer training signals that could enhance the model's strategic 
understanding and decision-making capabilities.

#############################################################################