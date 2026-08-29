import os
import sys
import argparse
from typing import List, Optional

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch

from ai.mcts import MCTS
from ai.neural import AlphaTiger, load_checkpoint, save_checkpoint
from ai.train import * # type: ignore

def train(
    curriculum: List[CurriculumStage],
    config: TrainerConfig = TrainerConfig(),
    resume_path: Optional[str] = None,
    final_model_name: str = "final.pt",
) -> AlphaTiger:
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = AlphaTiger().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    buffer    = ReplayBuffer(config.buffer_size)

    global_iter = 0
    if resume_path and os.path.exists(resume_path):
        global_iter = load_checkpoint(model, optimizer, resume_path)
        print(f"Resumed ({model}) from iteration {global_iter}  ({resume_path}) on ({device})")

    for stage in curriculum:
        print(f"\n{'='*60}")
        print(f"  Stage: {stage.name}")
        print(f"  Iterations: {stage.iterations}  |  Simulations: {stage.simulations}")
        print(f"  Temperature: {stage.temperature}  (greedy after move {stage.temperature_cutoff})")
        print(f"{'='*60}")

        games_played = 0
        
        model.share_memory() 
        num_workers = 8
        
        ctx = mp.get_context("spawn")

        while games_played < stage.iterations:
            batch_size = min(num_workers, stage.iterations - games_played)
            
            model.eval()
            
            with ProcessPoolExecutor(max_workers=batch_size, mp_context=ctx) as executor:
                futures = []
                for _ in range(batch_size):
                    mcts_worker = MCTS(model)
                    futures.append(
                        executor.submit(
                            self_play_game,
                            mcts_worker,
                            stage.state_factory,
                            stage.temperature,
                            stage.temperature_cutoff,
                            stage.simulations,
                        )
                    )
                
                batch_samples = [] 
                for future in futures:
                    samples = future.result()
                    buffer.add(samples)
                    batch_samples.extend(samples)
                    global_iter += 1
            
            games_played += batch_size

            total_loss = val_loss = pol_loss = 0.0
            steps = 0

            model.train()
            if len(buffer) >= config.min_buffer_size:
                total_train_steps = config.train_steps_per_iter * batch_size
                
                for _ in range(total_train_steps):
                    batch = buffer.sample(config.batch_size)
                    tl, vl, pl = train_step(model, optimizer, batch, device)
                    total_loss += tl
                    val_loss   += vl
                    pol_loss   += pl
                    steps      += 1

            prefix = f"[{stage.name}] iter {games_played:>4}/{stage.iterations} | buf {len(buffer):>6}"
            if steps:
                if len(batch_samples) == 0:
                    print("DEBUG: Batch ended with zero samples! Check your GameState initialization.")
                else:
                    print(
                        f"{prefix} | loss {total_loss/steps:.4f} "
                        f"(val {val_loss/steps:.4f}  pol {pol_loss/steps:.4f})"
                        f" | batch samples {len(batch_samples)} | winner {'british' if batch_samples[0][-1] == 1 else 'mysore'}"
                    )
            else:
                print(f"{prefix} | warming up ({len(buffer)}/{config.min_buffer_size})")

            # ── Checkpoint ───────────────────────────────────────────────────
            if global_iter % config.save_every == 0 or games_played == stage.iterations:
                path = os.path.join(config.checkpoint_dir, f"ckpt_{global_iter:06d}.pt")
                save_checkpoint(model, optimizer, global_iter, path)
                print(f"  ↳ saved {path}")

    final_path = os.path.join(config.checkpoint_dir, final_model_name)
    save_checkpoint(model, optimizer, global_iter, final_path)
    print(f"\nTraining complete — final model saved to {final_path}")
    return model

class CustomStateFactory:
    """A top-level, pickleable factory wrapper to inject the starting game state string."""
    def __init__(self, orig_factory, state_str):
        self.orig_factory = orig_factory
        self.state_str = state_str
        
    def __call__(self):
        state = self.orig_factory()
        state = state.read_str(self.state_str)
        return state

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--state_file", type=str, default=None, help="Path to file containing the game string representation")
    parser.add_argument("--model_name", type=str, default="final.pt", help="Name of the final saved model file")
    custom_args, remaining_argv = parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining_argv
    
    args, curriculum, config = setup_training_run("AlphaTiger Multiprocessing Trainer")
    
    if custom_args.state_file and os.path.exists(custom_args.state_file):
        with open(custom_args.state_file, 'r') as f:
            state_str = f.read().strip()
            
        for stage in curriculum:
            stage.state_factory = CustomStateFactory(stage.state_factory, state_str)

    train(curriculum, config=config, resume_path=args.resume, final_model_name=custom_args.model_name)