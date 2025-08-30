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
    
    def evaluate_position(self, fen: str, depth: int) -> Optional[float]:
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
                chess.engine.Limit(depth=depth, time=self.time_limit)
            )

            score = info["score"].white()  
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

def chess_reward_function(
    prompt: str, 
    response: str, 
    ground_truth: Dict[str, Any],
    move_evaluations: Dict[str, Dict]):
    """
    Optimized chess reward function that uses pre-computed move evaluations.
    
    Args:
        prompt: Chess position prompt
        response: Model's response
        ground_truth: Ground truth data
        move_evaluations: Pre-computed move evaluations and rankings
        
    Returns:
        Tuple of (reward value, predicted_move)
    """
    # Extract necessary data
    initial_fen = ground_truth.get('fen')
    ground_truth_move = ground_truth.get('best_move')
    _, predicted_move = extract_reasoning_and_move(response)
    
    # Return 0 if missing data
    if not initial_fen or not ground_truth_move or not predicted_move:
        logger.warning("Missing data for reward calculation.")
        return 0.0, None
    
    try:
        board = chess.Board(initial_fen)
        legal_moves = [board.san(move) for move in board.legal_moves]
    except Exception as e:
        logger.error(f"Failed to load initial FEN: {e}")
        return 0.0, None
    
    # Base reward for valid data
    reward = 1.0
    
    # Check if predicted move is legal
    if predicted_move in legal_moves:
        reward += 1.0  # +1 for legal move
    else:
        logger.warning(f"Illegal move predicted: {predicted_move}")
        return reward, predicted_move
    
    # Use pre-computed evaluations if available
    if not move_evaluations:
        logger.warning("No move evaluations available")
        return reward, predicted_move
    
    # Check if predicted move was evaluated
    if predicted_move not in move_evaluations:
        logger.warning(f"Predicted move {predicted_move} not in evaluations")
        return reward, predicted_move
    
    # Get ranking information
    move_info = move_evaluations[predicted_move]
    predicted_rank = move_info['rank']
    total_moves = move_info['total_moves']
    
    # Calculate ranking reward
    if total_moves == 1:
        ranking_reward = 10.0
    else:
        ranking_reward = 10 - ((predicted_rank - 1) * 9.0) / (total_moves - 1)
    
    reward += ranking_reward
    
    # Bonus for matching ground truth
    if predicted_move == ground_truth_move:
        reward += 10.0
    
    return reward, predicted_move
        

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


    sample_data = {
        "fen": "8/8/8/3pk3/R3p3/3r4/5K2/8 w - -",
        "best_move": "Ra7",
        "best_line": "Ra7 Ke3 Ra3+ Ke4 Ra2 h4 gxh4 gxh4 Rh2 Rc1",
        # "depth": 42
    }

    test_responses = [
        "Best move: Ra7\n<reasoning>\nThis is an endgame position where the rook on c6 is a major threat. I need to get the rook out of the way to alleviate pressure. The best move is to check the king and force an exchange to simplify the position.</reasoning>",
    ]
    
    print("Testing reward function:")
    for i, response in enumerate(test_responses, 1):
        reward, predicted_move = chess_reward_function("", response, sample_data)  
        reasoning, move = extract_reasoning_and_move(response)
        print(f"Test {i}: Move='{move}', Predicted='{predicted_move}', Reasoning={len(reasoning)} chars, Reward={reward:.3f}")
    
    # Cleanup
    cleanup_stockfish()
