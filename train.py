"""Meta-train a plastic key-value memory with a frozen Qwen3 model.

Compatible with data_hybrid/*.jsonl from augment_memory.py.
New architecture: old checkpoints / comparison scripts using Memory(d) MLP
are NOT compatible without updating their model definitions.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer


class Memory(nn.Module):
    """Associative memory. Query each token against trainable keys; read values."""
    def __init__(self, d, n_memories=64, temperature=0.15):
        super().__init__()
        self.d = d
        self.temperature = temperature
        self.keys = nn.Parameter(torch.randn(n_memories, d) * 0.02)
        self.values = nn.Parameter(torch.randn(n_memories, d) * 1e-3)

    def forward(self, x):
        # x: (..., D), output: (..., D). Float32 for numerical stability.
        q = F.normalize(x.float(), dim=-1, eps=1e-6)
        k = F.normalize(self.keys, dim=-1, eps=1e-6)
        scores = q @ k.T / self.temperature
        weights = torch.softmax(scores, dim=-1)
        return weights @ self.values


class BeliefUpdate(nn.Module):
    def __init__(self, d, n_heads=8):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d)
        self.output = nn.Linear(d, d)
        nn.init.normal_(self.output.weight, std=1e-4)
        nn.init.zeros_(self.output.bias)



    def forward(self, x, y):
        """
        x : (B, Lx, D)
        y : (B, Ly, D)

        return : (B, Lx,D)
        """

        x = x.float()
        y = y.float()

        # Les tokens de X interrogent ceux de Y
        attended, _ = self.cross_attention(
            query=x,
            key=y,
            value=y,
            need_weights=False,
        )

        # Représentation de X enrichie par Y
        v = self.output(self.norm(x + attended))


        return v


def read_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Run prepare_claire.py / augment_memory.py.')
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def encode(tokenizer, text, device, max_tokens):
    return tokenizer(text, return_tensors='pt', truncation=True,
                     max_length=max_tokens, add_special_tokens=False).input_ids.to(device)


def run_episode(episode, model, tokenizer, mem, updater, device, args, meta=True):
    # Parameter copies stay connected to the meta-learned initialization.
    params = {name: p.clone() for name, p in mem.named_parameters()}
    weighted_losses = []
    weights = []
    delta_ratios = []
    synthetic_count = 0
    embedding = model.get_input_embeddings()
    valid_turn = 0

    for turn in episode['turns']:
        xids = encode(tokenizer, turn['input'], device, args.max_input)
        yids = encode(tokenizer, turn['target'], device, args.max_target)
        if xids.numel() == 0 or yids.numel() == 0:
            continue
        valid_turn += 1
        x = embedding(xids).detach()
        y = embedding(yids).detach()
        xm, ym = x.float(), y.float()
        xmean, ymean = xm.mean(dim=1), ym.mean(dim=1)

        # Read memory BEFORE observing the target response.
        delta = functional_call(mem, params, (xm,))
        adjusted = x + (args.alpha * delta).to(x.dtype)
        labels = torch.cat((torch.full_like(xids, -100), yids), dim=1)
        inputs = torch.cat((adjusted, y), dim=1)
        ce = model(inputs_embeds=inputs, labels=labels, use_cache=False).loss

        relative_delta_sq = ((args.alpha * delta).square().sum(dim=-1)
                             / xm.detach().square().sum(dim=-1).clamp_min(1e-8)).mean()
        delta_ratios.append(relative_delta_sq.detach().sqrt().item())
        if valid_turn > args.warmup:
            is_recall = turn.get('kind') == 'synthetic_recall'
            synthetic_count += int(is_recall)
            weight = (valid_turn - args.warmup) * (args.synthetic_weight if is_recall else 1.)
            weighted_losses.append(weight * (ce + args.delta_reg * relative_delta_sq))
            weights.append(weight)




        # Cible générée par BeliefUpdate
        target = updater(xm, ym)

        # Loss interne
        inner = F.mse_loss(delta, target)

        # Gradient sur les paramètres de Memory
        gradients = torch.autograd.grad(
            inner,
            tuple(params.values()),
            create_graph=meta and not args.first_order,
        )
        if meta and not args.first_order:
            params = {name: p - args.inner_lr * g
                      for (name, p), g in zip(params.items(), gradients)}
        else:
            # First-order approximation: preserve gradient to initial K,V,
            # but NOT through the updater's update rule. For proper updater
            # meta-training use the default second-order mode.
            params = {name: p - args.inner_lr * g.detach()
                      for (name, p), g in zip(params.items(), gradients)}

    if not weighted_losses:
        return None
    return (sum(weighted_losses) / sum(weights),
            sum(delta_ratios) / len(delta_ratios), synthetic_count)


def evaluate(data, model, tok, mem, updater, device, args):
    values = []
    for episode in data[:args.eval_episodes]:
        result = run_episode(episode, model, tok, mem, updater, device, args, meta=False)
        if result is not None:
            values.append(result[0].detach().item())
    return sum(values) / len(values) if values else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='data_hybrid')
    ap.add_argument('--model', default='Qwen/Qwen3-0.6B')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--episodes-per-epoch', type=int, default=100)
    ap.add_argument('--eval-episodes', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--max-input', type=int, default=48)
    ap.add_argument('--max-target', type=int, default=48)
    ap.add_argument('--inner-lr', type=float, default=0.05)
    ap.add_argument('--outer-lr', type=float, default=1e-4)
    ap.add_argument('--memory-outer-lr', type=float, default=1e-5)
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--synthetic-weight', type=float, default=3.)
    ap.add_argument('--delta-reg', type=float, default=0.01)
    ap.add_argument('--n-memories', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=0.15)
    ap.add_argument('--belief-hidden', type=int, default=256)
    ap.add_argument('--first-order', action='store_true',
                    help='Cheaper approximation; does NOT meta-train BeliefUpdate from future CE.')
    ap.add_argument('--checkpoint', default='belief_update_kv_best.pt')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    train = read_data(Path(args.data_dir) / 'train.jsonl')
    val = read_data(Path(args.data_dir) / 'val.jsonl')
    test = read_data(Path(args.data_dir) / 'test.jsonl')
    print('Episodes:', len(train), len(val), len(test), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device).eval()
    model.requires_grad_(False)
    d = model.get_input_embeddings().embedding_dim
    mem = Memory(d, args.n_memories, args.temperature).to(device)
    updater = BeliefUpdate(d, n_heads=8).to(device)
    optimizer = torch.optim.AdamW([
        {'params': mem.parameters(), 'lr': args.memory_outer_lr},
        {'params': updater.parameters(), 'lr': args.outer_lr},
    ])
    print(f'Memory: {args.n_memories} keys + {args.n_memories} values, d={d}; '
          f'BeliefUpdate input={3*d}; second_order={not args.first_order}', flush=True)
    if args.first_order:
        print('WARNING: --first-order blocks the future-CE gradient to BeliefUpdate. '
              'Use second-order for the intended meta-learning.', flush=True)
    best = float('inf')
    for epoch in range(args.epochs):
        random.shuffle(train)
        losses = []
        for index, episode in enumerate(train[:args.episodes_per_epoch], 1):
            optimizer.zero_grad(set_to_none=True)
            result = run_episode(episode, model, tokenizer, mem, updater, device, args, meta=True)
            if result is None:
                continue
            loss, delta_ratio, recalls = result
            loss.backward()

            nn.utils.clip_grad_norm_(list(mem.parameters()) + list(updater.parameters()), 1.)
            optimizer.step()
            losses.append(loss.detach().item())
            if index % 10 == 0:
                print(f'  {index}/{min(len(train), args.episodes_per_epoch)} '
                      f'loss={sum(losses)/len(losses):.4f} '
                      f'| ||alpha*delta||/||x||={delta_ratio:.6f} '
                      f'| synthetic_recalls={recalls}', flush=True)
        validation = evaluate(val, model, tokenizer, mem, updater, device, args)
        avg = sum(losses) / len(losses) if losses else float('nan')
        print(f'Epoch {epoch+1}/{args.epochs} train={avg:.4f} val={validation:.4f}', flush=True)
        if math.isfinite(validation) and validation < best:
            best = validation
            torch.save({
                'updater': updater.state_dict(),
                'memory_init': mem.state_dict(),
                'config': vars(args),
                'architecture': 'key_value_attention_v1',
            }, args.checkpoint)
            print(f'  Saved best: {args.checkpoint}', flush=True)
    if Path(args.checkpoint).exists() and math.isfinite(best):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        mem.load_state_dict(ckpt['memory_init'])
        updater.load_state_dict(ckpt['updater'])
    print('Final test loss:', evaluate(test, model, tokenizer, mem, updater, device, args))
    print('NOTE: This is teacher-assisted CE, not an exact-match recall metric.')


if __name__ == '__main__':
    main()
