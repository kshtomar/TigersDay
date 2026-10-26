import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from ai.arena import play_match
from ai.neural import AlphaTiger, ONNXAlphaTiger, load_checkpoint, save_checkpoint
from game.constants import DEFAULT_MODEL

LR = 1e-3
BATCH_SIZE = 512
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 3
VAL_SPLIT = 0.2
SHARD_GAMES = 100
CACHE_DIR = Path('checkpoints/distill')
OUTPUT_PATH = CACHE_DIR / 'student.pt'
OUTCOME_WEIGHT = 0.75
FIELDS = ('states', 'masks', 'values', 'policies', 'game_ids')


def evaluate_positions(model, pos):
    values, logits = model.session.run(model.output_names, {model.input_name: pos.astype(np.float32)})
    return values.ravel(), logits


def collect_games(models, count, sims, seed, offset=0):
    states, masks, values, policies, game_ids = [], [], [], [], []
    for game_id in range(offset, offset + count):
        game_seed = seed + game_id

        if len(models) == 1:
            m_idx, b_idx = 0, 0
        else:
            m_idx = game_id % 2
            b_idx = 1 - m_idx

        winner, _, _, (pos, msk) = play_match(
            models[m_idx], models[b_idx], sims, sims, seed=game_seed, record=True
        )
        winner_model = models[m_idx if winner == -1 else b_idx]

        teacher_values, logits = evaluate_positions(winner_model, pos)
        masked_logits = torch.from_numpy(logits).masked_fill(~torch.from_numpy(msk), -1e9)
        policy = torch.softmax(masked_logits, dim=-1).numpy()

        states.append(pos)
        masks.append(msk)
        # Blend true outcome with teacher predicted value
        values.append((OUTCOME_WEIGHT * winner + (1 - OUTCOME_WEIGHT) * teacher_values).astype(np.float32))
        policies.append(policy.astype(np.float32))
        game_ids.append(np.full(len(pos), game_id, dtype=np.int64))

    return {
        'states': np.concatenate(states), 'masks': np.concatenate(masks),
        'values': np.concatenate(values), 'policies': np.concatenate(policies),
        'game_ids': np.concatenate(game_ids)
    }


def _collect_shard(paths, offset, count, sims, seed, output):
    models = [ONNXAlphaTiger(p) for p in paths]
    data = collect_games(models, count, sims, seed, offset=offset)
    tmp = Path(output).with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **data)
    tmp.replace(output)


def collect_dataset(paths, games, sims, seed, workers):
    teacher = "_".join(Path(p).stem for p in paths)
    shard_dir = CACHE_DIR / f"shards_{teacher}_{sims}s_{seed}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shards, pending = [], []
    for offset in range(0, games, SHARD_GAMES):
        count = min(SHARD_GAMES, games - offset)
        out = shard_dir / f"{offset:06d}.npz"
        shards.append(out)
        # Reuse if teacher already played
        if not out.exists():
            pending.append((offset, count, str(out)))

    if pending:
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(_collect_shard, paths, off, cnt, sims, seed, out)
                       for off, cnt, out in pending]
            for f in as_completed(futures):
                f.result()

    return sorted(shards)


def load_dataset(shards):
    directory = shards[0].parent / 'mapped'
    directory.mkdir(parents=True, exist_ok=True)
    counts = []
    for path in shards:
        with np.load(path) as data:
            counts.append(len(data['values']))
    total = sum(counts)
    with np.load(shards[0]) as data:
        for key in FIELDS:
            example = data[key]
            mapped = np.lib.format.open_memmap(directory / f'{key}.npy', mode='w+',
                dtype=example.dtype, shape=(total, *example.shape[1:]))
            mapped._mmap.close()
    offset = 0
    for path, count in zip(shards, counts):
        with np.load(path) as data:
            for key in FIELDS:
                mapped = np.load(directory / f'{key}.npy', mmap_mode='r+')
                mapped[offset:offset + count] = data[key]
                mapped.flush()
                mapped._mmap.close()
        offset += count
    return {key: directory / f'{key}.npy' for key in FIELDS}


def read_rows(path, indices):
    mapped = np.load(path, mmap_mode='r')
    try:
        return mapped[indices]
    finally:
        # Global random reads otherwise accumulate resident pages across epochs.
        mapped._mmap.close()


def stream_batches(data, indices):
    for start in range(0, len(indices), BATCH_SIZE):
        idx = indices[start:start + BATCH_SIZE]
        x, m, v, p = (torch.from_numpy(read_rows(data[key], idx)) for key in FIELDS[:-1])
        yield x, m, v[:, None], p


def compute_loss(model, x, m, v, p):
    pv, lg = model(x, value_logits=True)
    lv = F.binary_cross_entropy_with_logits(pv, (v + 1) / 2)
    lp = F.kl_div(F.log_softmax(lg.masked_fill(~m, -1e9), dim=-1), p, reduction='batchmean')
    return lv, lp


@torch.no_grad()
def evaluate(student, data, indices, device):
    student.eval()
    tot_v, tot_p, count = 0.0, 0.0, 0
    for x, m, v, p in stream_batches(data, indices):
        x, m, v, p = (t.to(device) for t in (x, m, v, p))
        lv, lp = compute_loss(student, x, m, v, p)
        tot_v += lv.item() * len(x)
        tot_p += lp.item() * len(x)
        count += len(x)
    return tot_v / count, tot_p / count


def train_student(student, data, device, seed):
    rng = np.random.default_rng(seed)
    game_ids = np.load(data['game_ids'])
    games = np.unique(game_ids)
    rng.shuffle(games)
    val_games = games[:int(len(games) * VAL_SPLIT)]
    val_mask = np.isin(game_ids, val_games)
    validation, training = np.flatnonzero(val_mask), np.flatnonzero(~val_mask)

    opt = torch.optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    base_v, base_p = evaluate(student, data, validation, device)
    best_score, best_ep = 1.0, 0
    best_weights = copy.deepcopy(student.state_dict())
    best_optimizer = copy.deepcopy(opt.state_dict())

    print(f"Baseline validation: BCE={base_v:.4f}, KL={base_p:.4f}")

    for ep in range(1, MAX_EPOCHS + 1):
        student.train()
        rng.shuffle(training)
        for x, m, v, p in stream_batches(data, training):
            x, m, v, p = (t.to(device) for t in (x, m, v, p))
            lv, lp = compute_loss(student, x, m, v, p)
            opt.zero_grad()
            (lv + lp).backward()
            opt.step()

        # Early stop if validation loss not improving
        val_v, val_p = evaluate(student, data, validation, device)
        score = 0.5 * (val_v / base_v + val_p / base_p)
        improved = score < best_score

        if improved:
            best_score, best_ep = score, ep
            best_weights = copy.deepcopy(student.state_dict())
            best_optimizer = copy.deepcopy(opt.state_dict())

        print(f"Epoch {ep:2d}: BCE={val_v:.4f}, KL={val_p:.4f}, Score={score:.4f}{' *' if improved else ''}", flush=True)

        if ep - best_ep >= PATIENCE:
            break

    student.load_state_dict(best_weights)
    opt.load_state_dict(best_optimizer)
    student.eval()
    return opt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers', nargs='+', default=[DEFAULT_MODEL])
    parser.add_argument('--student', default=None)
    parser.add_argument('--games', type=int, required=True)
    parser.add_argument('--sims', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=110100)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()

    if args.games % SHARD_GAMES != 0:
        parser.error(f"Games must be multiple of {SHARD_GAMES}")
    if args.games < SHARD_GAMES / VAL_SPLIT:
        parser.error(f"Games must be at least {SHARD_GAMES / VAL_SPLIT}")
    if len(args.teachers) not in (1, 2):
        parser.error("Expected 1 or 2 teachers")

    torch.set_num_threads(1)
    device = torch.device(args.device)

    started = time.monotonic()
    shards = collect_dataset(args.teachers, args.games, args.sims, args.seed, args.workers)
    data = load_dataset(shards)
    print(f"Dataset ready in {(time.monotonic() - started) / 60:.1f}m ({len(shards)} shards)")

    torch.manual_seed(args.seed)
    student = AlphaTiger().to(device)
    if args.student:
        load_checkpoint(student, None, args.student)

    opt = train_student(student, data, device, args.seed)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(student, opt, iteration=0, path=OUTPUT_PATH)
    print(f"Saved checkpoint to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
