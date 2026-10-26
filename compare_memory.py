"""Compare Qwen3-0.6B alone vs Qwen3-0.6B + learned neural memory.
Place next to train.py and data/test.jsonl.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import Memory, BeliefUpdate, encode, read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='belief_update_best.pt')
    parser.add_argument('--test', default='data/test.jsonl')
    parser.add_argument('--episodes', type=int, default=30, help='0 = all episodes')
    parser.add_argument('--output', default='comparison_test.csv')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = SimpleNamespace(**ckpt['config'])
    model_name = cfg.model
    print(f'Model: {model_name} | device: {device}')
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device).eval()
    model.requires_grad_(False)
    embedding = model.get_input_embeddings()
    d = embedding.embedding_dim
    mem = Memory(d).to(device).eval()
    updater = BeliefUpdate(d).to(device).eval()
    mem.load_state_dict(ckpt['memory_init'])
    updater.load_state_dict(ckpt['updater'])
    mem.requires_grad_(False)  # functional params below are differentiable
    updater.requires_grad_(False)

    data = read_data(args.test)
    if args.episodes > 0:
        data = data[:args.episodes]
    rows = []
    for ep_idx, ep in enumerate(data, 1):
        # IMPORTANT: same initial memory for each episode, as during training.
        params = {n: p.detach().clone().requires_grad_(True) for n, p in mem.named_parameters()}
        for turn_idx, turn in enumerate(ep['turns']):
            xids = encode(tok, turn['input'], device, cfg.max_input)
            yids = encode(tok, turn['target'], device, cfg.max_target)
            if not xids.numel() or not yids.numel():
                continue
            with torch.no_grad():
                x = embedding(xids).detach()
                y = embedding(yids).detach()
            xm, ym = x.float(), y.float()
            xmean, ymean = xm.mean(dim=1), ym.mean(dim=1)
            labels = torch.cat([torch.full_like(xids, -100), yids], dim=1)
            with torch.no_grad():
                baseline = model(inputs_embeds=torch.cat([x, y], dim=1),
                                 labels=labels, use_cache=False).loss.item()
            # This model's forward is frozen but its input remains differentiable.
            delta = functional_call(mem, params, (xm,))
            adjusted = x + (cfg.alpha * delta).to(x.dtype)
            with torch.no_grad():
                memory_ce = model(inputs_embeds=torch.cat([adjusted.detach(), y], dim=1),
                                  labels=labels, use_cache=False).loss.item()
            # Same online teacher update as train.py, AFTER evaluating this turn.
            current = functional_call(mem, params, (xmean,))
            with torch.no_grad():
                target = updater(xmean, ymean, current.detach())
            inner_loss = F.mse_loss(current, target)
            grads = torch.autograd.grad(inner_loss, tuple(params.values()))
            params = {n: (p - cfg.inner_lr * g).detach().requires_grad_(True)
                      for (n, p), g in zip(params.items(), grads)}
            rows.append({
                'episode': ep_idx, 'turn': turn_idx + 1,
                'baseline_ce': baseline, 'memory_ce': memory_ce,
                'gain_ce': baseline - memory_ce,
                'delta_norm': delta.detach().norm(dim=-1).mean().item(),
                'embedding_norm': xm.detach().norm(dim=-1).mean().item(),
                'input': turn['input'][:200], 'target': turn['target'][:200],
            })
        ep_rows = [r for r in rows if r['episode'] == ep_idx]
        if ep_rows:
            gains = [r['gain_ce'] for r in ep_rows if r['turn'] > cfg.warmup]
            print(f'Episode {ep_idx}/{len(data)} | gain post-warmup: '
                  f'{sum(gains)/len(gains):+.5f}' if gains else f'Episode {ep_idx}: no scored turns',
                  flush=True)
    if not rows:
        raise RuntimeError('No valid turns found in test dataset')
    with open(args.output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    scored = [r for r in rows if r['turn'] > cfg.warmup]
    if not scored:
        print('No post-warmup turns: increase episode length or reduce warmup.')
        return
    avg = lambda k: sum(r[k] for r in scored) / len(scored)
    b, m = avg('baseline_ce'), avg('memory_ce')
    print('\n=== PAIRED COMPARISON ON SAME TEST TURNS ===')
    print(f'Episodes: {len(data)} | evaluated turns (after warmup): {len(scored)}')
    print(f'Qwen alone       CE: {b:.6f} | perplexity: {math.exp(b):.3f}')
    print(f'Qwen + memory    CE: {m:.6f} | perplexity: {math.exp(m):.3f}')
    print(f'Gain (positive = memory better): {b-m:+.6f} CE')
    print(f'Fraction of turns improved: {sum(r["gain_ce"]>0 for r in scored)/len(scored):.1%}')
    print(f'Mean ||delta|| / ||embedding||: {sum(r["delta_norm"] / max(r["embedding_norm"], 1e-9) for r in scored)/len(scored):.6f}')
    print(f'Per-turn results saved to: {args.output}')
    print('NOTE: Ground-truth targets are used to update memory AFTER each turn.')
    print('This tests teacher-assisted online adaptation, not autonomous recall or free generation.')


if __name__ == '__main__':
    main()
