#!/usr/bin/env python3
"""
Data processing for chess GRPO training
Loads and preprocesses the stockfish evaluations dataset
"""
import json
import pandas as pd
import chess
from datasets import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessDataProcessor:
    def __init__(self, data_path: str, min_depth: int = 10, train_samples: Optional[int] = None):
        """
        Initialize chess data processor
        
        Args:
            data_path: Path to stockfish_evaluations.jsonl
            min_depth: Minimum depth for filtering quality positions
            train_samples: Maximum number of training samples to use (None for all available)
        """
        self.data_path = data_path
        self.min_depth = min_depth
        self.train_samples = train_samples
        
    def _parse_evaluation(self, eval_str: str) -> float:
        """
        Parse evaluation string to float
        Handles both numeric evaluations and mate scores
        
        Args:
            eval_str: Evaluation string (e.g., "0.58", "M18", "M1")
            
        Returns:
            Numeric evaluation value
        """
        eval_str = str(eval_str).strip()
        
        if eval_str.startswith('M'):
            # Mate score - convert to large positive/negative value
            try:
                mate_moves = int(eval_str[1:])
                # Positive mate = good for current side, negative = bad
                # Use large values to represent mate advantage
                return 10.0 if mate_moves > 0 else -10.0
            except ValueError:
                return 0.0
        else:
            # Regular numeric evaluation
            try:
                return float(eval_str)
            except ValueError:
                return 0.0

    def _validate_position(self, fen: str, move: str) -> bool:
        """Validate chess position and move"""
        try:
            board = chess.Board(fen)
            move_obj = board.parse_san(move)
            return move_obj in board.legal_moves
        except:
            return False



#     def create_chess_prompt(self, fen: str) -> str:
#         """Create standardized prompt for chess position with structured output"""
        
#         # Parse turn info
#         try:
#             board = chess.Board(fen)
#             turn = "White" if board.turn == chess.WHITE else "Black"
#         except:
#             turn = "Unknown"
        
#         return f"""You are a professional chess grandmaster. Analyze the following position and provide your best move.
# Position (FEN): {fen}
# Turn: {turn}
# Your response MUST follow this EXACT format, without any extra text, PGN, or game history. 
# Example Format of your response:
# Best move: [The single best move in Standard Algebraic Notation, e.g., Ra7]
# <reasoning>
# [Explain the strategic and tactical reasons for your move. Address the opponent's threats and your own opportunities.]
# </reasoning>
# END OF RESPONSE
# Your response for the given position:"""

    def create_chess_prompt(self, fen: str) -> str:
        """Create standardized prompt for chess position with structured output"""
        
        # Parse turn info
        try:
            board = chess.Board(fen)
            turn = "White" if board.turn == chess.WHITE else "Black"
        except:
            turn = "Unknown"
        
        return f"""You are a professional chess grandmaster. Analyze the following position and provide your best move.

Position (FEN): {fen}
Turn: {turn}

Your response MUST follow this EXACT format, without any extra text, PGN, or game history. 
------------------------------------------------------
Standard Algebraic Notation is as follow
Basic Format: [Piece][Disambiguation][Capture][Destination][Promotion][Check/Checkmate]
1. Piece Identifier (Optional - 0-1 characters)

K = King
Q = Queen
R = Rook
B = Bishop
N = Knight
No letter = Pawn

2. Disambiguation (Optional - 1-2 characters)
Used when multiple pieces of the same type can move to the same square:

File disambiguation: a-h (e.g., Rad1 - Rook from a-file to d1)
Rank disambiguation: 1-8 (e.g., R1d2 - Rook from rank 1 to d2)
Full square: a1-h8 (e.g., Ra1d1 - rarely needed)

3. Capture Indicator (Optional - 1 character)

x = Indicates a capture (e.g., Bxc7, axb5)

4. Destination Square (Required - 2 characters)

File: a-h (columns)
Rank: 1-8 (rows)
Examples: e4, a1, h8

5. Pawn Promotion (Optional - 2 characters)

= followed by piece: =Q, =R, =B, =N
Example: e8=Q (pawn promotes to queen)

6. Check/Checkmate (Optional - 1-2 characters)

+ = Check
# = Checkmate

Examples from Your Data

Bxc7: Bishop captures on c7
Qxc7: Queen captures on c7
Ra8: Rook moves to a8
Re8#: Rook moves to e8, delivering checkmate
Nf3: Knight moves to f3
b3+: Pawn moves to b3, giving check

Special Moves
Castling (3-5 characters)

O-O = Kingside castling
O-O-O = Queenside castling

------------------------------------------------------
Example Format of your response:

Best move: [The single best move in Standard Algebraic Notation, e.g., Ra7]
<reasoning>
[Explain the strategic and tactical reasons for your move. Address the opponent's threats and your own opportunities.]
<reasoning>
END OF RESPONSE


Your response for the given position:
"""

 

    def prepare_datasets(self, val_size: float = 0.05) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Prepare train/val/test datasets following assignment instructions
        Test set = last 1,000 rows from RAW dataset (before filtering)
        Training set size controlled by train_samples parameter
        
        Args:
            val_size: Validation set proportion of training data (default 5%)
            
        Returns:
            train_dataset, val_dataset, test_dataset (test = last 1,000 raw rows)
        """
        # Step 1: First, collect test set (last 1000 rows) - do this FIRST
        logger.info("Collecting test set (last 1,000 rows from raw dataset)...")
        
        with open(self.data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
            
        logger.info(f"Total lines in raw dataset: {total_lines}")
        test_start_line = total_lines - 1000
        
        test_data = []
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= test_start_line:
                    # Test set: last 1000 rows (minimal filtering, just valid JSON)
                    try:
                        entry = json.loads(line.strip())
                        entry['evaluation'] = self._parse_evaluation(entry['evaluation'])
                        test_data.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Test set collected: {len(test_data)} samples")
        
        # Step 2: Now collect training/validation data (everything before test set)
        logger.info("Collecting training/validation data...")
        train_val_data = []
        train_val_count = 0
        
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                # Stop before test set
                if i >= test_start_line:
                    break
                    
                try:
                    entry = json.loads(line.strip())
                    
                    # Apply quality filters to training data
                    if entry.get('depth', 0) >= self.min_depth:
                        if self._validate_position(entry['fen'], entry['best_move']):
                            entry['evaluation'] = self._parse_evaluation(entry['evaluation'])
                            train_val_data.append(entry)
                            train_val_count += 1
                            
                            # Stop if we have enough training samples
                            if self.train_samples and train_val_count >= self.train_samples:
                                break
                                
                except json.JSONDecodeError:
                    continue
        
        # Create DataFrames
        test_df = pd.DataFrame(test_data)
        train_val_df = pd.DataFrame(train_val_data)
        
        logger.info(f"Test set (last 1000 raw rows): {len(test_df)}")
        logger.info(f"Training/validation data after filtering: {len(train_val_df)}")
        
        # Step 3: Split training/validation data
        if len(train_val_df) == 0:
            train_df = pd.DataFrame()
            val_df = pd.DataFrame()
        else:
            val_samples = int(len(train_val_df) * val_size)
            val_samples = max(50, min(val_samples, len(train_val_df) // 20))  # At least 50, at most 5%
            
            train_df = train_val_df.head(len(train_val_df) - val_samples).copy()
            val_df = train_val_df.tail(val_samples).copy()
        
        # Step 4: Create prompts for all datasets
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            if len(split_df) > 0:
                split_df.loc[:, 'prompt'] = split_df['fen'].apply(self.create_chess_prompt)
        
        logger.info(f"Final dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df) if len(train_df) > 0 else Dataset.from_dict({})
        val_dataset = Dataset.from_pandas(val_df) if len(val_df) > 0 else Dataset.from_dict({})
        test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, val_dataset, test_dataset

def load_chess_datasets(data_path: str, train_samples: Optional[int] = None, min_depth: int = 10):
    """
    Convenience function to load chess datasets following assignment requirements
    
    Args:
        data_path: Path to stockfish evaluations file
        train_samples: Maximum number of training+validation samples (None for all available)
        min_depth: Minimum depth filter
    
    Returns:
        train_dataset, val_dataset, test_dataset (test = last 1,000 raw rows)
    """
    processor = ChessDataProcessor(data_path, min_depth=min_depth, train_samples=train_samples)
    return processor.prepare_datasets()

if __name__ == "__main__":
    # Test the data processing
    data_path = "../data/stockfish_evaluations.jsonl"
    
    # Load with specific training size for testing
    train_ds, val_ds, test_ds = load_chess_datasets(data_path, train_samples=5000, min_depth=8)
    
    print("Sample data:")
    if len(train_ds) > 0:
        print(train_ds[0])
    else:
        print("No training data found!")