#!/usr/bin/env python3
"""
Simple GRPO training script for chess move prediction
Uses TRL library for GRPO implementation
"""
import os
import sys
import argparse
import logging
import torch
import warnings
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import AutoModelForCausalLMWithValueHead, GRPOConfig, GRPOTrainer
from datasets import Dataset
import numpy as np

# Assuming these are available in your environment
from data_processing import load_chess_datasets
from utils import chess_reward_function, get_stockfish_manager, cleanup_stockfish
import chess
import chess.engine

warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible with gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class RewardEvaluationCallback(TrainerCallback):
    """
    A custom callback to evaluate the model's reward on the test set periodically.
    """
    def __init__(self, test_dataset: Dataset, tokenizer: AutoTokenizer, call_step: int):
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.reward_fn_name = "reward_eval"
        self.callback_step = call_step

    def evaluate_on_test_set(self, trainer: GRPOTrainer) -> None:
        """Performs a full evaluation run on the test dataset."""
        logger.info("Starting evaluation on the test set...")
        total_rewards = []
        pred_moves = []
        best_moves = []
        
        # Disable progress bar and logging during this custom evaluation
        original_disable_tqdm = trainer.args.disable_tqdm
        original_report_to = trainer.args.report_to
        trainer.args.disable_tqdm = True
        trainer.args.report_to = "none"

        for i, sample in enumerate(self.test_dataset):
            # Generate a response from the model
            input_ids = self.tokenizer(sample['prompt'], return_tensors="pt").input_ids.to(trainer.model.device)
            output_ids = trainer.model.generate(
                input_ids,
                do_sample=True,
                max_new_tokens=150,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            input_length = input_ids.shape[1]
            generated_tokens = output_ids[0][input_length:]  # Get only the new tokens
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Calculate the reward using our custom function

            reward, predicted_m = chess_reward_function(
                prompt=sample['prompt'],
                response=response_text,
                ground_truth={
                    'fen': sample['fen'],
                    'best_move': sample['best_move']
                }
            )
            total_rewards.append(reward)
            pred_moves.append(predicted_m)
            best_moves.append( sample['best_move'])

        # Calculate average and log to W&B
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        # Log the metrics with a unique name to avoid conflicts
        trainer.log({
            f"{self.reward_fn_name}/mean": avg_reward,
            f"{self.reward_fn_name}/std": std_reward,
        })
        logger.info(f"Predcited best moves: {str(pred_moves)}| Best moves: {str(best_moves)} | Reward: {str(reward)}")
        logger.info(f"Evaluation finished at step {trainer.state.global_step}| Average reward: {avg_reward:.4f}| Len test set: {len(total_rewards)} |")

        # Restore original settings
        trainer.args.disable_tqdm = original_disable_tqdm
        trainer.args.report_to = original_report_to

    def on_step_end(self, args, state, control, **kwargs):
        """
        Runs at the end of each training step.
        Fixed: Access trainer through the callback handler's trainer reference.
        """
        # Check if it's time to evaluate
        if state.global_step > 0 and state.global_step % self.callback_step == 0:
            if hasattr(self, 'trainer'):
                self.evaluate_on_test_set(self.trainer)
            else:
                logger.warning("Trainer reference not available in callback")

def main():
    """
    Main function to run the GRPO training loop.
    """
    config = {
        'model_name': "meta-llama/Llama-3.1-8B-Instruct",
        'data_path': "data/stockfish_evaluations.jsonl",
        'output_dir': "/workspace/chess_play/models/chess-grpo-finetune",
        'train_samples': 15000,  
        'min_depth': 10,
        'num_epochs': 5,
        'batch_size': 2,
        'mini_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'learning_rate': 4e-5,
        'max_length': 512,
        'max_new_tokens': 200,  # Longer for reasoning
        'temperature': 0.3,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'use_8bit': True,
        'logging_steps': 10,
        'save_steps': 250,
        'total_steps': 1300,
        'gradient_checkpointing': False,
        'test_dataset_every': 1500,
        'top_testset_to_test': 25,
    }

    logger.info("Starting Chess GRPO Training...")
    logger.info(f"Config: {config}")

    # Load datasets
    logger.info("Loading chess datasets...")
    train_dataset, val_dataset, test_dataset = load_chess_datasets(
        config['data_path'],
        train_samples=config['train_samples'],
        min_depth=config['min_depth']
    )

    if len(train_dataset) == 0:
        raise ValueError("No training data found!")

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    quantization_config = None
    if config['use_8bit']:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16
        )

    logger.info(f"Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)
    logger.info("LoRA applied to model")
    model.print_trainable_parameters()

    def evaluate_all_legal_moves(fen: str, depth: int, stockfish_manager):
        """Evaluate all legal moves for a position once."""
        try:
            board = chess.Board(fen)
            moving_side = board.turn
            legal_moves = [board.san(move) for move in board.legal_moves]
            
            move_evaluations = []
            
            for legal_move_san in legal_moves:
                test_board = board.copy()
                try:
                    move = test_board.parse_san(legal_move_san)
                    test_board.push(move)
                    evaluation = stockfish_manager.evaluate_position(test_board.fen(), depth)
                    
                    if evaluation is not None:
                        move_evaluations.append((legal_move_san, evaluation))
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate move {legal_move_san}: {e}")
                    continue
            
            # Sort moves by evaluation
            if moving_side == chess.WHITE:
                move_evaluations.sort(key=lambda x: x[1], reverse=True)
            else:
                move_evaluations.sort(key=lambda x: x[1])
            
            # Return as dict with rankings
            result = {}
            for rank, (move_san, eval_score) in enumerate(move_evaluations, 1):
                result[move_san] = {
                    'evaluation': eval_score,
                    'rank': rank,
                    'total_moves': len(move_evaluations)
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to evaluate position {fen}: {e}")
            return {}


    def reward_fn(prompts, completions, **kwargs):
        """
        Reward function with batch optimization for identical positions.
        """
        rewards = []
        predicted_moves = []
        
        # Get additional data from kwargs
        fens = kwargs.get('fen', [None] * len(prompts))
        best_moves = kwargs.get('best_move', [None] * len(prompts))
        evaluations = kwargs.get('evaluation', [None] * len(prompts))
        best_lines = kwargs.get('best_line', [None] * len(prompts))
        depths = kwargs.get('depth', [None] * len(prompts))
        stockfish_manager = get_stockfish_manager()
        if not stockfish_manager.engine:
            logger.error("Stockfish engine not available")
            return [0.0] * len(prompts)
        
        # Since all positions are identical, evaluate once for the entire batch
        first_fen = fens[0] if fens[0] else ''
        first_depth = depths[0] if depths[0] else 15
        
        if first_fen:
            # Evaluate all legal moves once for this position
            move_evaluations = evaluate_all_legal_moves(first_fen, first_depth, stockfish_manager)
            logger.info(f"Evaluated {len(move_evaluations)} moves for position")
            
            # Extract legal moves and their evaluations for logging
            all_legal_moves = list(move_evaluations.keys()) if move_evaluations else []
            eval_scores = {move: data['evaluation'] for move, data in move_evaluations.items()} if move_evaluations else {}
            
        else:
            move_evaluations = {}
            all_legal_moves = []
            eval_scores = {}
        
        logger.info(f"fens : {str(fens)} \n best moves: {str(best_moves)} \n depths: {str(depths)} \n eval: {str(evaluations)} \n all_legal : {all_legal_moves} \n eval_for_all_legal: {eval_scores}")
        
        # Process each completion using the same evaluation
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            ground_truth = {
                'fen': fens[i] if fens[i] is not None else '',
                'best_move': best_moves[i] if best_moves[i] is not None else '',
                'evaluation': evaluations[i] if evaluations[i] is not None else 0,
                'best_line': best_lines[i] if best_lines[i] is not None else '',
                'depth': depths[i] if depths[i] is not None else 15
            }
            
            # Use the pre-computed evaluations
            reward, predicted_move = chess_reward_function(
                prompt, completion, ground_truth, move_evaluations
            )
            rewards.append(float(reward))  
            predicted_moves.append(predicted_move)
        
        logger.info(f"reward: {str(rewards)}")
        logger.info(f"predicted moves: {str(predicted_moves)}")
        
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            logger.info(f"Batch Average Reward: {avg_reward:.4f}")
        
        logger.info("--------------------------")
        return rewards

    # Create GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=config['output_dir'],
        learning_rate=config['learning_rate'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        max_steps=config['total_steps'],
        num_train_epochs=config['num_epochs'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        temperature=config['temperature'],
        max_prompt_length=config['max_length'],
        max_completion_length=config['max_new_tokens'],
        gradient_checkpointing=False,
        # Additional useful parameters
        log_completions=True,
        num_completions_to_print=5,
        remove_unused_columns=False,  # Keep all dataset columns for reward function
        report_to="wandb",
    )

    # Initialize GRPO trainer with correct parameter names
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,  
        processing_class=tokenizer,  
        train_dataset=train_dataset,  
        reward_funcs=[reward_fn],  
    )


    device = next(model.parameters()).device
    logger.info(f"Training on device: {device}")

    # Start training
    logger.info("Starting automated training...")
    grpo_trainer.train()

    # Save final model
    logger.info(f"Saving final model to {config['output_dir']}")
    os.makedirs(config['output_dir'], exist_ok=True)
    grpo_trainer.save_model()

    logger.info("Training completed successfully!")

    # Cleanup
    cleanup_stockfish()

if __name__ == "__main__":
    main()