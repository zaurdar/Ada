#!/usr/bin/env python3
"""Convert SpokenWOZ dialogues to the exact JSONL episode format used by compare_long_kv.py.

20-50 spoken turns = 10-25 consecutive USER->SYSTEM pairs by default.
The resulting JSONL is frozen/reproducible across all checkpoint evaluations.

Input: official SpokenWOZ text JSON (mapping dialogue ID -> {log: [...]})
       or a JSONL of dialogue objects. No generated or synthetic dialogue is added.
"""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset


def text_of(turn):
    for k in ('text', 'utterance', 'transcript', 'content'):
        v = turn.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ''


def role_of(turn):
    role = str(turn.get('speaker', turn.get('role', ''))).strip().lower()
    if role in ('user', 'human', 'customer', '0'):
        return 'user'
    if role in ('system', 'assistant', 'agent', 'bot', '1'):
        return 'system'
    return None


def dialogue_records(obj):
    """Yield (id, record) for canonical SpokenWOZ mapping or common alternatives."""
    if isinstance(obj, dict):
        if isinstance(obj.get('log'), (list, dict)) or isinstance(obj.get('turns'), (list, dict)):
            yield str(obj.get('dialogue_id', obj.get('id', 'unknown'))), obj
        elif isinstance(obj.get('dialogues'), (list, dict)):
            yield from dialogue_records(obj['dialogues'])
        else:
            for key, value in obj.items():
                if isinstance(value, dict) and ('log' in value or 'turns' in value):
                    yield str(key), value
    elif isinstance(obj, list):
        for i, rec in enumerate(obj):
            if isinstance(rec, dict):
                if 'log' in rec or 'turns' in rec:
                    yield str(rec.get('dialogue_id', rec.get('id', i))), rec
                else:
                    yield from dialogue_records(rec)


def turns_of(record):
    turns = record.get('log', record.get('turns', []))
    if isinstance(turns, dict):
        # Canonical JSON often uses an array; preserve numeric order for dict-shaped logs.
        try:
            keys = sorted(turns, key=lambda x: int(x))
        except (ValueError, TypeError):
            keys = list(turns)
        turns = [turns[k] for k in keys]
    if not isinstance(turns, list):
        return []
    return turns


def extract_pairs(record):
    turns = turns_of(record)
    pairs = []
    pending = None
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            pending = None
            continue
        text = text_of(turn)
        role = role_of(turn)
        if role is None:
            # SpokenWOZ follows MultiWOZ: log alternates user, system, starting user.
            role = 'user' if i % 2 == 0 else 'system'
        if not text:
            pending = None
            continue
        if role == 'user':
            pending = text
        elif role == 'system':
            if pending:
                pairs.append({'input': pending, 'target': text})
            pending = None
    return pairs


def read_records(path):
    if path.suffix.lower() == '.jsonl':
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():
                    yield from dialogue_records(json.loads(line))
    else:
        with path.open(encoding='utf-8-sig') as f:
            yield from dialogue_records(json.load(f))

def read_huggingface(split="dev"):
    print("Chargement de SpokenWOZ depuis Hugging Face...")

    ds = load_dataset(
        "pirxus/spokenwoz-whisper",
        split=split,
        streaming=True
    )

    from datasets import Audio

    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
        ds = ds.remove_columns("audio")

    dialogues = defaultdict(list)

    for row in ds:
        dialogue_id = row["wav_id"]

        dialogues[dialogue_id].append({
            "turn_index": row["turn_index"],
            "input": row["text"],
            "target": row["agent_text"]
        })

    print(f"{len(dialogues)} conversations récupérées.")

    for dialogue_id, turns in dialogues.items():
        turns.sort(key=lambda t: t["turn_index"])

        pairs = [
            {"input": t["input"], "target": t["target"]}
            for t in turns
            if t["input"] and t["target"]
        ]

        yield dialogue_id, pairs

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=None)

    p.add_argument(
        '--split',
        default='dev',
        choices=['train', 'dev', 'test']
    )
    p.add_argument('--output', default='spokenwoz_20_50.jsonl', type=Path)
    p.add_argument('--min-turns', type=int, default=20, help='Minimum spoken utterances, not pairs')
    p.add_argument('--max-turns', type=int, default=50, help='Maximum spoken utterances, not pairs')
    p.add_argument('--episodes', type=int, default=150, help='0 = all eligible')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.min_turns < 2 or args.max_turns < args.min_turns:
        p.error('Require 2 <= --min-turns <= --max-turns')
    min_pairs = (args.min_turns + 1) // 2
    max_pairs = args.max_turns // 2
    candidates = []
    scanned = 0
    if args.input is None:
        source = read_huggingface(args.split)
    else:
        source = (
            (dialogue_id, extract_pairs(record))
            for dialogue_id, record in read_records(args.input)
        )

    for dialogue_id, pairs in source:
        scanned += 1

        if len(pairs) < min_pairs:
            continue

        candidates.append({
            'source_id': dialogue_id,
            'turns': pairs[:max_pairs]
        })
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[:args.episodes] if args.episodes else candidates
    if not selected:
        raise SystemExit(f'No conversations with >= {min_pairs} pairs among {scanned} dialogues. '
                         'Try --min-turns 10 or check the JSON format.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    counts = [len(x['turns']) for x in selected]
    print(f'Scanned {scanned} dialogues; {len(candidates)} eligible; selected {len(selected)}.')
    print(f'Pairs per conversation: min={min(counts)}, max={max(counts)}, '
          f'mean={sum(counts)/len(counts):.1f}')
    print(f'Saved frozen evaluation set: {args.output.resolve()}')
    print('Reuse this SAME JSONL for every checkpoint (do not regenerate between runs).')

if __name__ == '__main__':
    main()
