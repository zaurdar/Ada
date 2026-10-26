"""Paired CE ablation: frozen Qwen vs static KV vs online-updated KV memory.

Place beside the CURRENT train.py (KV Memory, cross-attention BeliefUpdate).
Memory is reset for every episode; ground-truth targets are used only AFTER scoring.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer
from train import Memory, BeliefUpdate, encode, read_data


def mean(values):
    return sum(values) / len(values) if values else float('nan')


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', default='belief_update_kv_best.pt')
    ap.add_argument('--n-memories', type=int, default=None,
                    help='Number of KV slots (default: detect from checkpoint)')
    ap.add_argument('--test', default='data_hybrid/test.jsonl')
    ap.add_argument('--episodes', type=int, default=30, help='0 = all episodes')
    ap.add_argument('--include-synthetic', action='store_true',
                    help='Include synthetic turns (default: natural turns only)')
    ap.add_argument('--max-turn', type=int, default=0, help='0 = all turns')
    ap.add_argument('--output-prefix', default='ablation_two_kv')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--min-count', type=int, default=1)
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = SimpleNamespace(**ckpt['config'])
    print(f'Model: {cfg.model} | device: {device} | warmup: {cfg.warmup}', flush=True)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, dtype=torch.float16 if device.type == 'cuda' else torch.float32
    ).to(device).eval()
    model.requires_grad_(False)
    embedding = model.get_input_embeddings()
    d = embedding.embedding_dim

    checkpoint_slots = ckpt['memory_init']['keys'].shape[0]
    n_memories = args.n_memories if args.n_memories is not None else checkpoint_slots
    if n_memories != checkpoint_slots:
        ap.error(f'--n-memories={n_memories} but checkpoint contains {checkpoint_slots} slots')
    mem = Memory(d, n_memories=n_memories,
                 temperature=getattr(cfg, 'temperature', 0.15)).to(device).eval()
    mem.load_state_dict(ckpt['memory_init'])
    mem.requires_grad_(False)

    # Current train.py: BeliefUpdate(d, n_heads=8), output (B, Lx, D).
    updater = BeliefUpdate(d, n_heads=8).to(device).eval()
    updater.load_state_dict(ckpt['updater'])
    updater.requires_grad_(False)
    print(f'KV slots: {n_memories} | d={d} | alpha={cfg.alpha} | inner_lr={cfg.inner_lr}', flush=True)

    data = read_data(args.test)
    if not args.include_synthetic:
        # Remove synthetic turns BEFORE scoring and BEFORE online updates.
        # Keep natural turns in their original order within each episode.
        filtered = []
        removed = 0
        for episode in data:
            natural_turns = [t for t in episode['turns']
                             if not str(t.get('kind', '')).startswith('synthetic')]
            removed += len(episode['turns']) - len(natural_turns)
            if natural_turns:
                filtered.append({**episode, 'turns': natural_turns})
        data = filtered
        print(f'Natural-only evaluation: excluded {removed} synthetic turns; '
              f'{len(data)} episodes remain.', flush=True)
    if args.episodes > 0:
        data = data[:args.episodes]

    # Static and dynamic branches start from exactly the same trained KV initialization.
    static_params = {name: p.detach().clone() for name, p in mem.named_parameters()}

    rows = []
    for ep_idx, ep in enumerate(data, 1):
        # Fresh differentiable parameters for this episode, without meta graph.
        params = {name: p.detach().clone().requires_grad_(True)
                  for name, p in mem.named_parameters()}
        local = []
        valid_turn = 0
        for turn_idx, turn in enumerate(ep['turns']):
            if args.max_turn and turn_idx >= args.max_turn:
                break
            xids = encode(tok, turn['input'], device, cfg.max_input)
            yids = encode(tok, turn['target'], device, cfg.max_target)
            if xids.numel() == 0 or yids.numel() == 0:
                continue
            valid_turn += 1
            with torch.no_grad():
                x = embedding(xids).detach()
                y = embedding(yids).detach()
            xm, ym = x.float(), y.float()
            labels = torch.cat((torch.full_like(xids, -100), yids), dim=1)

            # Score BEFORE the update with this turn's ground truth.
            with torch.no_grad():
                ce_base = model(inputs_embeds=torch.cat((x, y), dim=1),
                                labels=labels, use_cache=False).loss.item()
                delta_static = functional_call(mem, static_params, (xm,))
                adjusted_static = x + (cfg.alpha * delta_static).to(x.dtype)
                ce_static = model(inputs_embeds=torch.cat((adjusted_static, y), dim=1),
                                  labels=labels, use_cache=False).loss.item()
                delta = functional_call(mem, params, (xm,))
                adjusted = x + (cfg.alpha * delta).to(x.dtype)
                ce_dynamic = model(inputs_embeds=torch.cat((adjusted, y), dim=1),
                                   labels=labels, use_cache=False).loss.item()
                ratio = (((cfg.alpha * delta).square().sum(dim=-1)
                          / xm.square().sum(dim=-1).clamp_min(1e-8)).mean().sqrt().item())

            row = {
                'episode': ep_idx, 'turn': valid_turn,
                'post_warmup': int(valid_turn > cfg.warmup),
                'synthetic_recall': int(turn.get('kind') == 'synthetic_recall'),
                'qwen_ce': ce_base, 'static_ce': ce_static, 'dynamic_ce': ce_dynamic,
                'static_gain': ce_base - ce_static,
                'adaptation_gain': ce_static - ce_dynamic,
                'dynamic_gain': ce_base - ce_dynamic,
                'dynamic_delta_ratio': ratio,
                'input': turn['input'][:200], 'target': turn['target'][:200],
            }
            rows.append(row)
            local.append(row)

            # Teacher-assisted update AFTER scoring. Identical to current train.py:
            # inner = mse(delta, updater(xm, ym)), then gradient step on K and V.
            # Recompute delta with autograd enabled: scoring used no_grad().
            with torch.enable_grad():
                delta_grad = functional_call(mem, params, (xm,))
                with torch.no_grad():
                    target = updater(xm, ym)
                if delta_grad.shape != target.shape:
                    raise ValueError(f'Memory output {delta_grad.shape} != BeliefUpdate target {target.shape}; '
                                     'check that train.py uses per-token MSE.')
                inner = F.mse_loss(delta_grad, target)
                grads = torch.autograd.grad(inner, tuple(params.values()))
                params = {name: (p - cfg.inner_lr * g).detach().requires_grad_(True)
                          for (name, p), g in zip(params.items(), grads)}

        scored = [r for r in local if r['post_warmup']]
        print(f'Episode {ep_idx}/{len(data)} | turns={len(local)} | '
              f'post-warmup CE gain={mean([r["dynamic_gain"] for r in scored]):+.6f}', flush=True)

    if not rows:
        raise RuntimeError('No valid test turns found.')

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    details_path = prefix.with_name(prefix.name + '_turns.csv')
    summary_path = prefix.with_name(prefix.name + '_by_position.csv')
    save_csv(details_path, rows)

    groups = defaultdict(list)
    for row in rows:
        groups[row['turn']].append(row)
    positions = []
    for turn, rr in sorted(groups.items()):
        positions.append({
            'turn': turn, 'count': len(rr),
            'post_warmup': int(turn > cfg.warmup),
            'qwen_ce': mean([r['qwen_ce'] for r in rr]),
            'static_ce': mean([r['static_ce'] for r in rr]),
            'static_gain': mean([r['static_gain'] for r in rr]),
            'adaptation_gain': mean([r['adaptation_gain'] for r in rr]),
            'dynamic_ce': mean([r['dynamic_ce'] for r in rr]),
            'dynamic_gain': mean([r['dynamic_gain'] for r in rr]),
        })
    save_csv(summary_path, positions)

    print('\n=== MEAN CE BY CONVERSATION POSITION (lower is better) ===')
    print('Turn | N episodes | Qwen     | Static   | Dynamic  | Static gain | Adaptation gain')
    for p in positions:
        print(f'{p["turn"]:>4} | {p["count"]:>10} | {p["qwen_ce"]:>8.5f} | '
              f'{p["dynamic_ce"]:>8.5f} | {p["dynamic_gain"]:+.6f}')

    print('\n=== OVERALL PAIRED RESULTS ===')
    subsets = [
        ('All valid turns', rows),
        (f'After warmup (>{cfg.warmup})', [r for r in rows if r['post_warmup']]),
        *(([('Synthetic recall only', [r for r in rows if r['synthetic_recall']])])
          if args.include_synthetic else []),
        ('Non-recall turns', [r for r in rows if not r['synthetic_recall']]),
    ]
    for name, subset in subsets:
        if not subset:
            print(f'{name}: no turns')
            continue
        base = mean([r['qwen_ce'] for r in subset])
        static = mean([r['static_ce'] for r in subset])
        dynamic = mean([r['dynamic_ce'] for r in subset])
        print(f'{name}: N={len(subset)}')
        print(f'  Qwen: {base:.6f} | Static: {static:.6f} | Dynamic: {dynamic:.6f}')
        print(f'  Static vs Qwen: {base - static:+.6f} CE')
        print(f'  Adaptation (Static - Dynamic): {static - dynamic:+.6f} CE')
        print(f'  Dynamic vs Qwen: {base - dynamic:+.6f} CE')
        print(f'  Fraction dynamic better than static: '
              f'{sum(r["adaptation_gain"] > 0 for r in subset) / len(subset):.1%}')
        print(f'  Fraction dynamic better than Qwen: '
              f'{sum(r["dynamic_gain"] > 0 for r in subset) / len(subset):.1%}')

    print(f'\nSaved: {details_path} and {summary_path}')
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            pp = [p for p in positions if p['count'] >= args.min_count]
            if pp:
                xx = [p['turn'] for p in pp]
                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.plot(xx, [p['qwen_ce'] for p in pp], marker='o', label='Qwen seul')
                ax.plot(xx, [p['static_ce'] for p in pp], marker='o', label='Qwen + Memory statique')
                ax.plot(xx, [p['dynamic_ce'] for p in pp], marker='o', label='Qwen + Memory dynamique')
                ax.axvline(cfg.warmup + 0.5, linestyle='--', color='gray', alpha=.6,
                           label=f'Fin warm-up ({cfg.warmup} tours)')
                ax.set(xlabel='Position dans la conversation', ylabel='Cross-Entropy moyenne',
                       title='Comparaison de la CE par position')
                ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
                plot_path = prefix.with_name(prefix.name + '_by_position.png')
                fig.savefig(plot_path, dpi=170); plt.close(fig)

                fig, ax = plt.subplots(figsize=(10, 5.5))
                ax.plot(xx, [p['static_gain'] for p in pp], marker='o', label='Gain statique vs Qwen')
                ax.plot(xx, [p['adaptation_gain'] for p in pp], marker='o', label='Gain adaptation (statique - dynamique)')
                ax.plot(xx, [p['dynamic_gain'] for p in pp], marker='o', alpha=.5, label='Gain total vs Qwen')
                ax.axhline(0, color='gray', linewidth=1)
                ax.axvline(cfg.warmup + 0.5, linestyle='--', color='gray', alpha=.6)
                ax.set(xlabel='Position dans la conversation', ylabel='Gain de CE (positif = mieux)',
                       title='Gain de la mémoire par position')
                ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
                gain_path = prefix.with_name(prefix.name + '_gains.png')
                fig.savefig(gain_path, dpi=170); plt.close(fig)
                print(f'Plots: {plot_path}, {gain_path}')
        except ImportError:
            print('matplotlib not installed: plots skipped')

    print('\nNOTE: Static Memory never updates within an episode; Dynamic Memory updates after each turn.')
    print('NOTE: Positive adaptation gain = online updates outperform trained static memory.')
    print('NOTE: CE excludes correction regularization and inner MSE.')
    print('NOTE: Dynamic Memory observes the ground-truth response only AFTER scoring each turn.')
    print('NOTE: Synthetic recall CE is not exact-match recall accuracy.')


if __name__ == '__main__':
    main()
