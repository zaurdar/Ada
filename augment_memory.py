"""Mix mostly natural Claire conversations with sparse explicit recall probes.

Reads existing data/{train,val,test}.jsonl; writes data_hybrid/{...}.jsonl.
Never modifies source files. Uses disjoint random facts per split/episode.
"""
import argparse
import json
import random
import secrets
import string
from pathlib import Path

NOUNS = ['dossier', 'carnet', 'projet', 'colis', 'badge', 'enveloppe', 'classeur', 'rendez-vous']


def augment(ep, rng, gap):
    turns = [dict(t) for t in ep['turns']]
    if len(turns) < gap + 3:
        return None
    # Cryptographically random opaque value: cannot be inferred from semantics.
    token = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(7))
    noun = rng.choice(NOUNS)
    key = f'code du {noun}'
    fact = {
        'input': f'Petite information à retenir pour plus tard : le {key} est {token}.',
        'target': f"D'accord, je retiens le code du {noun}.",
        'kind': 'synthetic_fact', 'memory_key': key, 'memory_value': token,
    }
    probe = {
        'input': f'Question de mémoire : quel est le {key} que je t’ai donné plus tôt ? Réponds uniquement avec le code.',
        'target': token,
        'kind': 'synthetic_recall', 'memory_key': key, 'memory_value': token,
    }
    # Place fact before warmup, and recall after at least gap natural turns.
    # At least 2 natural turns before fact when possible.
    fact_at = min(2, max(0, len(turns) - gap - 2))
    recall_at = fact_at + gap  # index in ORIGINAL natural turns
    out = turns[:fact_at] + [fact] + turns[fact_at:recall_at] + [probe] + turns[recall_at:]
    return {**ep, 'turns': out, 'has_synthetic_recall': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', default='data')
    p.add_argument('--output-dir', default='data_hybrid')
    p.add_argument('--fraction', type=float, default=0.25, help='Fraction of episodes enriched; others remain untouched')
    p.add_argument('--gap', type=int, default=3, help='Natural exchanges between fact and recall')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not 0 <= a.fraction <= 1:
        p.error('--fraction must be in [0, 1]')
    if a.gap < 1:
        p.error('--gap must be >= 1')
    source, dest = Path(a.input_dir), Path(a.output_dir)
    if source.resolve() == dest.resolve():
        p.error('input-dir and output-dir must differ to preserve natural data')
    dest.mkdir(parents=True, exist_ok=True)
    for split_i, split in enumerate(('train', 'val', 'test')):
        src = source / f'{split}.jsonl'
        if not src.exists():
            p.error(f'Missing {src}; run prepare_claire.py first')
        rng = random.Random(a.seed + split_i * 10007)
        with src.open(encoding='utf-8') as f:
            episodes = [json.loads(line) for line in f if line.strip()]
        eligible = [i for i, ep in enumerate(episodes) if len(ep.get('turns', [])) >= a.gap + 3]
        selected = set(rng.sample(eligible, min(round(len(episodes) * a.fraction), len(eligible))))
        changed = 0
        with (dest / f'{split}.jsonl').open('w', encoding='utf-8') as f:
            for i, ep in enumerate(episodes):
                if i in selected:
                    ep = augment(ep, rng, a.gap)
                    changed += 1
                f.write(json.dumps(ep, ensure_ascii=False) + '\n')
        natural_turns = sum(len(ep['turns']) for ep in episodes)
        print(f'{split}: {len(episodes)} episodes, {changed} enriched, '
              f'{natural_turns} natural turns + {2 * changed} synthetic turns; '
              f'natural turn share={natural_turns/(natural_turns + 2*changed):.1%}')
    print(f'Written to {dest.resolve()}')


if __name__ == '__main__':
    main()
