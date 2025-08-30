import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import json
import chess
import os
import sys
from datetime import datetime

# Add src to path since we're in eval/ and src/ is at same level
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your existing modules
from data_processing import ChessDataProcessor, load_chess_datasets
from utils import (
    extract_reasoning_and_move, 
    get_stockfish_manager, 
    cleanup_stockfish,
    StockfishManager
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessModelEvaluator:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", is_local_checkpoint: bool = False):
        """Initialize the chess model evaluator using existing infrastructure
        
        Args:
            model_name: Either HuggingFace model name or local checkpoint path
            is_local_checkpoint: True if model_name is a local fine-tuned checkpoint path
        """
        self.model_name = model_name
        self.is_local_checkpoint = is_local_checkpoint
        self.model = None
        self.tokenizer = None
        
        # Use your existing StockfishManager
        self.stockfish_manager = get_stockfish_manager()
        
        # Load model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """Load model exactly like your test script"""
        if self.is_local_checkpoint:
            print(f"Loading fine-tuned checkpoint: {self.model_name}")
            
            # For LoRA checkpoints, we need to load base model + adapter
            from peft import PeftModel
            
            # Load base model first
            base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
            print(f"Loading base model: {base_model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load the fine-tuned adapter
            print(f"Loading adapter from: {self.model_name}")
            self.model = PeftModel.from_pretrained(base_model, self.model_name)
            
            print(f"Fine-tuned model loaded successfully!")
            
        else:
            print(f"Loading base model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model with same settings as your test script
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            print(f"Base model loaded successfully!")
        
        print(f"Device: {next(self.model.parameters()).device}")

        # Show memory usage like your test script
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
    
    def prepare_test_dataset(self, data_path: str, test_csv_path: str = "test_dataset.csv") -> pd.DataFrame:
        """Prepare and save test dataset using your ChessDataProcessor"""
        if os.path.exists(test_csv_path):
            print(f"Loading existing test dataset from {test_csv_path}")
            return pd.read_csv(test_csv_path)
        
        print("Preparing test dataset using ChessDataProcessor...")
        
        # Use your existing data loading function
        _, _, test_dataset = load_chess_datasets(data_path, train_samples=1000, min_depth=10)
        
        if len(test_dataset) == 0:
            raise ValueError("Test dataset is empty!")
        
        # Convert to DataFrame and save
        test_df = pd.DataFrame(test_dataset)
        test_df.to_csv(test_csv_path, index=False)
        
        print(f"Test dataset prepared and saved to {test_csv_path}")
        print(f"Test dataset size: {len(test_df)}")
        
        return test_df
    
    def generate_response(self, prompt: str) -> str:
        """Generate response exactly like your test script"""
        # Same tokenization and device handling as your script
        inputs = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_token_count = inputs['input_ids'].shape[1]

        # Same generation settings as your test script
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=200,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Same response extraction as your test script
        generated_tokens = outputs[0][input_token_count:]
        generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_response.strip()
    
    def evaluate_move_accuracy(self, test_df: pd.DataFrame, sample_size: int = None) -> tuple:
        """Evaluate move accuracy using your existing functions"""
        if sample_size and sample_size < len(test_df):
            sample_df = test_df.sample(n=sample_size, random_state=42)
        else:
            sample_df = test_df
            
        results = {
            'correct_moves': 0,
            'legal_moves': 0,
            'total_moves': 0,
            'parse_failures': 0
        }
        
        # Storage for detailed results
        detailed_results = []
        
        print(f"Evaluating move accuracy on {len(sample_df)} positions...")
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Move Accuracy"):
            try:
                # Use the prompt that was already created by your ChessDataProcessor
                prompt = row['prompt']
                ground_truth_move = row['best_move']
                fen = row['fen']
                
                # Generate response using the model
                response = self.generate_response(prompt)
                
                # Use your existing function to extract move
                reasoning, predicted_move = extract_reasoning_and_move(response)
                
                # Store detailed result for this position
                position_result = {
                    'fen': fen,
                    'ground_truth_move': ground_truth_move,
                    'predicted_move': predicted_move,
                    'reasoning': reasoning,
                    'response': response,
                    'legal_moves': None,  # Will fill if move is valid
                    'is_legal': False,
                    'is_correct': False,
                    'parse_success': bool(predicted_move)
                }
                
                results['total_moves'] += 1
                
                if not predicted_move:
                    results['parse_failures'] += 1
                    detailed_results.append(position_result)
                    continue
                
                # Use your existing validation logic
                try:
                    board = chess.Board(fen)
                    legal_moves = [board.san(move) for move in board.legal_moves]
                    position_result['legal_moves'] = ', '.join(legal_moves)
                    
                    if predicted_move in legal_moves:
                        results['legal_moves'] += 1
                        position_result['is_legal'] = True
                        
                        if predicted_move == ground_truth_move:
                            results['correct_moves'] += 1
                            position_result['is_correct'] = True
                    
                except Exception as e:
                        logger.warning(f"Failed to validate move {predicted_move}: {e}")
                        detailed_results.append(position_result)
                        continue
                
                # Add detailed result to storage
                detailed_results.append(position_result)
                        
            except Exception as e:
                logger.error(f"Error processing position: {e}")
                continue
        
        # Calculate metrics
        accuracy_metrics = {
            'move_accuracy': results['correct_moves'] / max(results['total_moves'], 1),
            'legal_move_rate': results['legal_moves'] / max(results['total_moves'], 1), 
            'parse_success_rate': (results['total_moves'] - results['parse_failures']) / max(results['total_moves'], 1),
            'total_evaluated': results['total_moves'],
            'correct_moves': results['correct_moves'],
            'legal_moves': results['legal_moves'],
            'parse_failures': results['parse_failures']
        }
        
        # Create detailed DataFrame
        detailed_df = pd.DataFrame(detailed_results)
        return accuracy_metrics, detailed_df
    


    def evaluate_stockfish_mse(
        self, 
        test_df: pd.DataFrame, 
        sample_size: int = None, 
        move_details_df: pd.DataFrame = None, ) -> tuple:
        """Calculate MSE between model predicted move evaluations and ground truth best move evaluations"""
        # Use move_details_df as the basis for MSE calculation
        if move_details_df is not None and not move_details_df.empty:
            sample_df = move_details_df
        else:
            # Fallback to original logic if no move details provided
            if sample_size and sample_size < len(test_df):
                sample_df = test_df.sample(n=sample_size, random_state=42)
            else:
                sample_df = test_df
            
        squared_errors = []
        valid_evaluations = 0
        
        # Storage for detailed Stockfish results
        stockfish_details = []
        
        print(f"Evaluating Stockfish MSE vs best moves on {len(sample_df)} positions...")
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Stockfish MSE vs Best Move"):
            try:
                original_fen = row['fen']
                predicted_move = row.get('predicted_move', '')
                ground_truth_best_move = None
                depth = 15  # Default depth
                
                # Get ground truth best move from current row
                ground_truth_best_move = row.get('ground_truth_move')
                depth = row.get('depth', 15)
                
                if ground_truth_best_move is None:
                    logger.warning(f"No ground truth best move found in row")
                    continue
                
                # Apply ground truth best move to get evaluation baseline
                try:
                    board_gt = chess.Board(original_fen)
                    gt_move_obj = board_gt.parse_san(ground_truth_best_move)
                    board_gt.push(gt_move_obj)
                    gt_new_fen = board_gt.fen()
                    gt_eval = self.stockfish_manager.evaluate_position(gt_new_fen, depth)
                    gt_move_success = True
                except Exception as e:
                    logger.error(f"Failed to apply ground truth move {ground_truth_best_move}: {e}")
                    continue
                
                if gt_eval is None:
                    logger.warning(f"Failed to evaluate ground truth position: {gt_new_fen}")
                    continue
                
                # Skip illegal or missing predicted moves
                if not (predicted_move and row.get('is_legal', False)):
                    stockfish_result = {
                        'original_fen': original_fen,
                        'predicted_move': predicted_move,
                        'ground_truth_best_move': ground_truth_best_move,
                        'predicted_new_fen': None,
                        'gt_new_fen': gt_new_fen,
                        'predicted_move_applied_success': False,
                        'gt_eval': gt_eval,
                        'model_eval': None,
                        'depth': depth,
                        'squared_error': None,
                        'evaluation_success': False
                    }
                    stockfish_details.append(stockfish_result)
                    logger.warning(f"Skipping illegal/missing predicted move: {predicted_move}")
                    continue
                
                # Apply predicted move to get new position
                try:
                    board_pred = chess.Board(original_fen)
                    pred_move_obj = board_pred.parse_san(predicted_move)
                    board_pred.push(pred_move_obj)
                    pred_new_fen = board_pred.fen()
                    pred_move_success = True
                except Exception as e:
                    logger.error(f"Unexpected error applying legal predicted move {predicted_move}: {e}")
                    continue
                
                # Evaluate the predicted move's resulting position with Stockfish
                model_eval = self.stockfish_manager.evaluate_position(pred_new_fen, depth)
                
                # Store detailed result
                stockfish_result = {
                    'original_fen': original_fen,
                    'predicted_move': predicted_move,
                    'ground_truth_best_move': ground_truth_best_move,
                    'predicted_new_fen': pred_new_fen,
                    'gt_new_fen': gt_new_fen,
                    'predicted_move_applied_success': pred_move_success,
                    'gt_eval': gt_eval,
                    'model_eval': model_eval,
                    'depth': depth,
                    'squared_error': None,
                    'evaluation_success': model_eval is not None
                }
                
                if model_eval is not None:
                    squared_error = (model_eval - gt_eval) ** 2
                    squared_errors.append(squared_error)
                    stockfish_result['squared_error'] = squared_error
                    valid_evaluations += 1
                else:
                    logger.warning(f"Failed to evaluate predicted position: {pred_new_fen}")
    
                # Add to detailed storage
                stockfish_details.append(stockfish_result)
                    
            except Exception as e:
                logger.error(f"Error evaluating position: {e}")
                continue
        
        if valid_evaluations == 0:
            logger.error("No valid evaluations obtained!")
            return float('inf'), pd.DataFrame()
            
        mse = np.mean(squared_errors)
        print(f"Stockfish MSE vs Best Move: {mse:.4f} (based on {valid_evaluations} valid evaluations)")
        
        # Create detailed DataFrame
        stockfish_df = pd.DataFrame(stockfish_details)
        return mse, stockfish_df


    
    
    # def run_evaluation(
    #     self, 
    #     data_path: str, 
    #     sample_size: int = 100, 
    #     test_csv_path: str = "test_dataset.csv",
    #     move_details_path: str = None
    # ):
    #     """Run complete evaluation using your existing data loading"""
        
    #     # Step 1: Prepare test dataset (uses cached CSV if exists)
    #     test_df = self.prepare_test_dataset(data_path, test_csv_path)
        
    #     print(f"Test dataset size: {len(test_df)}")
        
    #     if len(test_df) == 0:
    #         raise ValueError("Test dataset is empty!")
        
    #     print("\n" + "="*50)
    #     print("CHESS MODEL EVALUATION")
    #     print("="*50)
        
    #     # 2. Move Accuracy Evaluation
    #     print("\n1. MOVE ACCURACY EVALUATION")
    #     print("-" * 30)
    #     move_metrics, move_details_df = self.evaluate_move_accuracy(test_df, sample_size)
        
    #     print(f"Move Accuracy: {move_metrics['move_accuracy']:.3f} ({move_metrics['correct_moves']}/{move_metrics['total_evaluated']})")
    #     print(f"Legal Move Rate: {move_metrics['legal_move_rate']:.3f} ({move_metrics['legal_moves']}/{move_metrics['total_evaluated']})")
    #     print(f"Parse Success Rate: {move_metrics['parse_success_rate']:.3f}")
        
    #     # 3. Stockfish MSE Evaluation  
    #     print(f"\n2. STOCKFISH EVALUATION MSE")
    #     print("-" * 30)
    #     stockfish_mse, stockfish_details_df = self.evaluate_stockfish_mse(test_df, sample_size, move_details_df)
        
    #     # Save detailed DataFrames
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     model_identifier = self.model_name.replace('/', '_').replace('..', '').replace('models_', '')
        
    #     # Save move accuracy details
    #     move_details_file = f"move_details_{model_identifier}_{timestamp}.csv"
    #     move_details_df.to_csv(move_details_file, index=False)
    #     print(f"Move details saved to: {move_details_file}")
        
    #     # Save Stockfish evaluation details  
    #     stockfish_details_file = f"stockfish_details_{model_identifier}_{timestamp}.csv"
    #     stockfish_details_df.to_csv(stockfish_details_file, index=False)
    #     print(f"Stockfish details saved to: {stockfish_details_file}")
        
    #     # Combine results
    #     evaluation_results = {
    #         'model_name': self.model_name,
    #         'is_local_checkpoint': self.is_local_checkpoint,
    #         'test_dataset_size': len(test_df),
    #         'sample_size_used': sample_size,
    #         'move_accuracy': move_metrics['move_accuracy'],
    #         'legal_move_rate': move_metrics['legal_move_rate'], 
    #         'parse_success_rate': move_metrics['parse_success_rate'],
    #         'stockfish_mse': stockfish_mse,
    #         'detailed_move_metrics': move_metrics,
    #         'move_details_file': move_details_file,
    #         'stockfish_details_file': stockfish_details_file,
    #         'detailed_records_count': len(move_details_df)
    #     }
        
    #     print(f"\n" + "="*50)
    #     print("EVALUATION SUMMARY")
    #     print("="*50)
    #     print(f"Model: {self.model_name}")
    #     print(f"Test samples: {sample_size if sample_size else len(test_df)}")
    #     print(f"Move Accuracy: {evaluation_results['move_accuracy']:.3f}")
    #     print(f"Legal Move Rate: {evaluation_results['legal_move_rate']:.3f}")  
    #     print(f"Stockfish MSE: {evaluation_results['stockfish_mse']:.4f}")
        
    #     return evaluation_results

    def run_evaluation(
        self, 
        data_path: str, 
        sample_size: int = 100, 
        test_csv_path: str = "test_dataset.csv", 
        move_details_path: str = None
    ):
        """Run complete evaluation using your existing data loading"""
        
        # Step 1: Prepare test dataset (uses cached CSV if exists)
        test_df = self.prepare_test_dataset(data_path, test_csv_path)
        
        print(f"Test dataset size: {len(test_df)}")
        
        if len(test_df) == 0:
            raise ValueError("Test dataset is empty!")
        
        print("\n" + "="*50)
        print("CHESS MODEL EVALUATION")
        print("="*50)
        
        # 2. Move Accuracy Evaluation - Skip if move_details_path provided
        if move_details_path and os.path.exists(move_details_path):
            print(f"\n1. LOADING EXISTING MOVE DETAILS")
            print("-" * 30)
            move_details_df = pd.read_csv(move_details_path)
            print(f"Loaded move details from: {move_details_path}")
            print(f"Move details size: {len(move_details_df)}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_identifier = self.model_name.replace('/', '_').replace('..', '').replace('models_', '')
            
            # Calculate metrics from loaded data
            total_evaluated = len(move_details_df)
            correct_moves = move_details_df['is_correct'].sum() if 'is_correct' in move_details_df.columns else 0
            legal_moves = move_details_df['is_legal'].sum() if 'is_legal' in move_details_df.columns else 0
            parse_success = move_details_df['parsing_success'].sum() if 'parsing_success' in move_details_df.columns else total_evaluated
            
            move_metrics = {
                'total_evaluated': total_evaluated,
                'correct_moves': correct_moves,
                'legal_moves': legal_moves,
                'move_accuracy': correct_moves / total_evaluated if total_evaluated > 0 else 0,
                'legal_move_rate': legal_moves / total_evaluated if total_evaluated > 0 else 0,
                'parse_success_rate': parse_success / total_evaluated if total_evaluated > 0 else 0
            }
            
            move_details_file = move_details_path  # Use existing file
        else:
            print("\n1. MOVE ACCURACY EVALUATION")
            print("-" * 30)
            move_metrics, move_details_df = self.evaluate_move_accuracy(test_df, sample_size)
            
            # Save move accuracy details
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_identifier = self.model_name.replace('/', '_').replace('..', '').replace('models_', '')
            move_details_file = f"move_details_{model_identifier}_{timestamp}.csv"
            move_details_df.to_csv(move_details_file, index=False)
            print(f"Move details saved to: {move_details_file}")
        
        print(f"Move Accuracy: {move_metrics['move_accuracy']:.3f} ({move_metrics['correct_moves']}/{move_metrics['total_evaluated']})")
        print(f"Legal Move Rate: {move_metrics['legal_move_rate']:.3f} ({move_metrics['legal_moves']}/{move_metrics['total_evaluated']})")
        print(f"Parse Success Rate: {move_metrics['parse_success_rate']:.3f}")
        
        # 3. Stockfish MSE Evaluation  
        print(f"\n2. STOCKFISH EVALUATION MSE")
        print("-" * 30)
        stockfish_mse, stockfish_details_df = self.evaluate_stockfish_mse(test_df, sample_size, move_details_df)
        
        # Save Stockfish evaluation details
        if 'move_details_file' not in locals():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_identifier = self.model_name.replace('/', '_').replace('..', '').replace('models_', '')
        
        
        stockfish_details_file = f"stockfish_details_{model_identifier}_{timestamp}.csv"
        stockfish_details_df.to_csv(stockfish_details_file, index=False)
        print(f"Stockfish details saved to: {stockfish_details_file}")
        
        # Combine results
        evaluation_results = {
            'model_name': self.model_name,
            'is_local_checkpoint': self.is_local_checkpoint,
            'test_dataset_size': len(test_df),
            'sample_size_used': sample_size,
            'move_accuracy': move_metrics['move_accuracy'],
            'legal_move_rate': move_metrics['legal_move_rate'], 
            'parse_success_rate': move_metrics['parse_success_rate'],
            'stockfish_mse': stockfish_mse,
            'detailed_move_metrics': move_metrics,
            'move_details_file': move_details_file,
            'stockfish_details_file': stockfish_details_file,
            'detailed_records_count': len(move_details_df)
        }
        
        print(f"\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {self.model_name}")
        print(f"Test samples: {sample_size if sample_size else len(test_df)}")
        print(f"Move Accuracy: {evaluation_results['move_accuracy']:.3f}")
        print(f"Legal Move Rate: {evaluation_results['legal_move_rate']:.3f}")  
        print(f"Stockfish MSE: {evaluation_results['stockfish_mse']:.4f}")
        
        return evaluation_results


def main():
    """Main evaluation function"""
    # Configuration - update these paths as needed
    DATA_PATH = "../data/stockfish_evaluations.jsonl"  # Path to your raw data
    TEST_CSV_PATH = "test_dataset.csv"  # Will be created/cached in eval/ directory
    SAMPLE_SIZE = 1000 # Number of test positions to evaluate (None for all)
    
    # Model configuration - choose one:
    
    # Option 1: Evaluate base Llama model
    # evaluator = ChessModelEvaluator()
    
    # # Option 2: Evaluate fine-tuned checkpoint
    # CHECKPOINT_PATH = "../models/chess-grpo-final-new-reward/checkpoint-1500"  # Update this path
    CHECKPOINT_PATH = "../models/chess-grpo-final-fixed-reward-resume/checkpoint-500"
    evaluator = ChessModelEvaluator(CHECKPOINT_PATH, is_local_checkpoint=True)
    
    try:
        # Run evaluation using your existing infrastructure
        results = evaluator.run_evaluation(
            DATA_PATH, 
            SAMPLE_SIZE, 
            TEST_CSV_PATH, 
        )
        
        # Save results with appropriate filename
        model_identifier = results['model_name'].replace('/', '_').replace('meta-llama_', '')
        results_file = f"evaluation_results_{model_identifier}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        print(f"\nDetailed results saved:")
        print(f"- Move accuracy details: {results.get('move_details_file', 'N/A')}")
        print(f"- Stockfish evaluation details: {results.get('stockfish_details_file', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # Use your existing cleanup function
        cleanup_stockfish()


if __name__ == "__main__":
    main()