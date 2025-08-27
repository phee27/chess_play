#!/usr/bin/env python3
"""
Utility functions for chess GRPO training
Contains reward function using Stockfish engine and helper utilities
"""
import chess
import chess.engine
import re
import numpy as np
import logging
import os
import subprocess
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StockfishManager:
    """Manage Stockfish engine for reward calculation"""
    
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 15, time_limit: float = 1.0):
        """
        Initialize Stockfish engine
        
        Args:
            stockfish_path: Path to stockfish binary (None to auto-detect)
            depth: Search depth for evaluation
            time_limit: Time limit per evaluation in seconds
        """
        self.stockfish_path = self._find_stockfish_path(stockfish_path)
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        
        # Initialize engine
        self._init_engine()
    
    def _find_stockfish_path(self, provided_path: Optional[str]) -> str:
        """Find Stockfish binary path"""
        if provided_path and os.path.exists(provided_path):
            return provided_path
            
        # Common paths to check
        common_paths = [
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish", 
            "/usr/games/stockfish",  # Added this path based on your feedback
            "/opt/homebrew/bin/stockfish",
            "stockfish",  # If in PATH
            "/workspace/stockfish/stockfish",  # Common in cloud environments
        ]
        
        for path in common_paths:
            try:
                # Use --help as a simple check if the binary exists and is executable
                if subprocess.run([path, "--help"], capture_output=True, timeout=5, check=True).returncode == 0:
                    logger.info(f"Found Stockfish at: {path}")
                    return path
            except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
                continue
        
        raise FileNotFoundError("Stockfish binary not found. Please provide the correct path.")

    
    def _init_engine(self):
        """Initialize the Stockfish engine"""
        if not self.stockfish_path:
            logger.warning("No Stockfish path available, using fallback scoring")
            return
            
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            logger.info("Stockfish engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Stockfish: {e}")
            self.engine = None
    
    def evaluate_position(self, fen: str) -> Optional[float]:
        """
        Evaluate a chess position using Stockfish
        
        Args:
            fen: Board position in FEN notation
            
        Returns:
            Numerical score from Stockfish (positive = good for side to move, None if failed)
        """
        if not self.engine:
            logger.warning("Stockfish engine is not initialized. Cannot evaluate position.")
            return None

        try:
            board = chess.Board(fen)
            
            # Evaluate current position
            info = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth, time=self.time_limit)
            )
            
            score = info["score"].relative
            
            # Convert score to float (from side-to-move perspective)
            if score.is_mate():
                # Mate score: positive if we're giving mate, negative if getting mated
                mate_moves = score.mate()
                if mate_moves > 0:
                    return 10.0 + (1.0 / mate_moves)  # Faster mate is better
                else:
                    return -10.0 - (1.0 / abs(mate_moves))  # Getting mated is bad
            else:
                # Regular centipawn score, convert to reasonable range
                cp_score = score.score() / 100.0  # Convert centipawns to pawns
                return max(-10.0, min(10.0, cp_score))  # Clamp to [-10, 10]
                
        except Exception as e:
            logger.warning(f"Stockfish position evaluation failed: {e}")
            return None
    
    def close(self):
        """Close the Stockfish engine"""
        if self.engine:
            self.engine.quit()
            self.engine = None

# Global Stockfish manager instance
_stockfish_manager = None

def get_stockfish_manager() -> StockfishManager:
    """Get global Stockfish manager instance"""
    global _stockfish_manager
    if _stockfish_manager is None:
        try:
            _stockfish_manager = StockfishManager()
        except FileNotFoundError as e:
            logger.error(e)
            _stockfish_manager = StockfishManager(stockfish_path="")
    return _stockfish_manager

def extract_reasoning_and_move(response: str) -> tuple[str, str]:
    """
    Extract reasoning and move from model response
    
    Args:
        response: Model's text output
        
    Returns:
        Tuple of (reasoning, move)
    """
    response = response.strip()
    
    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>(.*?)<reasoning>', response, re.DOTALL | re.IGNORECASE)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract move - look for "Best move: X" pattern
    move_match = re.search(r'(?:best\s+move|move):\s*([^\s\n.!?]+)', response, re.IGNORECASE)
    if move_match:
        move = move_match.group(1).strip()
    else:
        # Fallback: extract any chess-like move pattern
        patterns = [
            r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b',  # Long algebraic notation
            r'\b([NBRQK]?[a-h]?[1-8]?x?[a-h][1-8](?:=[NBRQ])?[+#]?)\b',  # Standard algebraic notation
            r'\b(O-O-O|O-O)\b',  # Castling
        ]
        
        move = ""
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                move = matches[-1]  # Take the last match
                break
    
    return reasoning.strip(), move.strip()



# def chess_reward_function(
#     prompt: str, 
#     response: str, 
#     ground_truth: Dict[str, Any]
# ) -> float:
#     """
#     Calculate reward based on change in Stockfish evaluation from the moving side's perspective.
#     Also provides a bonus for predicting the best move.
    
#     Args:
#         prompt: The chess position prompt (unused here, but part of the GRPO signature).
#         response: Model's response containing reasoning and predicted move.
#         ground_truth: Dictionary with stockfish data (fen, best_move, etc.).
        
#     Returns:
#         Reward value based on evaluation improvement and move accuracy.
#     """
#     stockfish_manager = get_stockfish_manager()
#     if not stockfish_manager.engine:
#         logger.error("Stockfish engine not available, cannot calculate reward.")
#         return 0.0, None

#     # 1. Extract necessary data from ground truth and model response
#     initial_fen = ground_truth.get('fen')
#     ground_truth_move = ground_truth.get('best_move')
    
#     # Extract model's predicted move from its response
#     _, predicted_move = extract_reasoning_and_move(response)

#     if not initial_fen or not ground_truth_move or not predicted_move:
#         logger.warning("Missing data for reward calculation.")
#         return -10.0, None # Penalty for incomplete data

#     # 2. Get initial evaluation
#     try:
#         initial_board = chess.Board(initial_fen)
#         initial_score = stockfish_manager.evaluate_position(initial_fen)
#         if initial_score is None:
#             return -10.0, None # Penalty if initial evaluation fails
#     except Exception as e:
#         logger.error(f"Failed to load initial FEN: {e}")
#         return -100.0, None # Big penalty for bad FEN

#     # 3. Get evaluation after the predicted move
#     try:
#         board_after_move = chess.Board(initial_fen)
#         move = board_after_move.parse_san(predicted_move)
        
#         if move not in board_after_move.legal_moves:
#             logger.warning(f"Illegal move predicted: {predicted_move}")
#             return -5.0, predicted_move # Penalty for illegal move
            
#         board_after_move.push(move)
#         final_score = stockfish_manager.evaluate_position(board_after_move.fen())
#         if final_score is None:
#             return -10.0, None # Penalty if final evaluation fails
#     except Exception as e:
#         logger.warning(f"Invalid move or FEN after move: {e}")
#         return -5.0, None # Penalty for move parsing errors

#     # 4. Calculate the reward based on the change in evaluation
#     # A positive change is good for the side that just moved.
#     # Stockfish evaluation is always relative to the side to move.
#     if initial_board.turn == chess.WHITE:
#         # For White, a higher positive score is better
#         positional_reward = final_score - initial_score
#     else: # Black's turn
#         # For Black, a lower negative score is better, so we reverse the change
#         positional_reward = initial_score - final_score
    
#     # 5. Add a bonus for playing the actual best move
#     accuracy_bonus = 0.0
#     if predicted_move == ground_truth_move:
#         accuracy_bonus = 1.0 # A configurable bonus for the perfect move
    
#     # 6. Combine rewards
#     total_reward = positional_reward + accuracy_bonus
    
#     # 7. Add penalty for incorrect output format
#     format_penalty = 0.0
#     end_phrase = "END OF RESPONSE"
#     if end_phrase not in response:
#         format_penalty -= 2.0  # Penalty for not including the phrase
#     elif response.strip().endswith(end_phrase) is False:
#         format_penalty -= 2.0  # Penalty for text after the phrase

#     # 8. Combine all rewards and return
#     total_reward += format_penalty
    
#     return total_reward, predicted_move



def chess_reward_function(
    prompt: str, 
    response: str, 
    ground_truth: Dict[str, Any]
) -> Tuple[float, str]:
    """
    Simplified version that focuses mainly on position evaluation.
    Good for initial experiments to see if the approach works.
    """
    stockfish_manager = get_stockfish_manager()
    if not stockfish_manager.engine:
        return 0, None
    
    initial_fen = ground_truth.get('fen')
    predicted_move = extract_reasoning_and_move(response)[1]
    
    if not initial_fen or not predicted_move:
        return 0, None
    
    try:
        board = chess.Board(initial_fen)
        moving_side = board.turn
        
        # Check legality
        move = board.parse_san(predicted_move)
        if move not in board.legal_moves:
            return 0, predicted_move
        
        # Execute and evaluate
        board.push(move)
        final_eval = stockfish_manager.evaluate_position(board.fen())
        
        if final_eval is None:
            return 0, predicted_move
        
        # Simple position-based reward
        if moving_side == chess.WHITE:
            reward = final_eval / 10.0  # Convert centipawns to pawn units
        else:
            reward = -final_eval / 10.0
        
        # Small baseline to encourage exploration
        reward += 0.2
        
        return reward, predicted_move
        
    except Exception as e:
        logger.warning(f"Simple reward calculation failed: {e}")
        return 0, predicted_move


        

def format_chess_position_for_display(fen: str) -> str:
    """Format chess position for human-readable display"""
    try:
        board = chess.Board(fen)
        return str(board)
    except:
        return f"Invalid FEN: {fen}"

# Cleanup function
def cleanup_stockfish():
    """Cleanup Stockfish resources"""
    global _stockfish_manager
    if _stockfish_manager:
        _stockfish_manager.close()
        _stockfish_manager = None

if __name__ == "__main__":
    # Test the reward function with the new chess position
    sample_data = {
        "fen": "8/4r3/2R2pk1/6pp/3P4/6P1/5K1P/8 b - -",
        "best_move": "Ra7",
        "best_line": "Ra7 Ke3 Ra3+ Ke4 Ra2 h4 gxh4 gxh4 Rh2 Rc1"
    }

    test_responses = [
        # Response with the correct best move
        "<reasoning>This is an endgame position where the rook on c6 is a major threat. I need to get the rook out of the way to alleviate pressure. The best move is to check the king and force an exchange to simplify the position.</reasoning>\n\nBest move: Ra7",
        # Response with a legal, but likely suboptimal move
        "Best move: Kg7",
        # Response with an illegal move
        "Best move: Rg7",
        # Response with a malformed move
        "I will move my king. The best move is: kf7"
    ]
    
    print("Testing reward function:")
    for i, response in enumerate(test_responses, 1):
        reward = chess_reward_function("", response, sample_data)
        reasoning, move = extract_reasoning_and_move(response)
        print(f"Test {i}: Move='{move}', Reasoning={len(reasoning)} chars, Reward={reward:.3f}")
    
    # Cleanup
    cleanup_stockfish()
