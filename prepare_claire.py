import argparse, hashlib, json, re
from pathlib import Path
from datasets import load_dataset

PAT = re.compile(r'^\[([^\]]+:)\]\s*(.*)$')

def turns(text):
    out=[]
    for line in text.splitlines():
        m=PAT.match(line.strip())
        if m and len(m.group(2).strip()) >= 8:
            out.append({'speaker':m.group(1)[:-1], 'text':m.group(2).strip()})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='data');p.add_argument('--train',type=int,default=1000)
    p.add_argument('--val',type=int,default=150);p.add_argument('--test',type=int,default=150)
    p.add_argument('--min-turns',type=int,default=7);p.add_argument('--max-turns',type=int,default=14)
    a=p.parse_args();dest=Path(a.out);dest.mkdir(parents=True,exist_ok=True)
    counts={s:0 for s in ['train','val','test']}
    files={s:(dest/f'{s}.jsonl').open('w',encoding='utf8') for s in counts}
    limits={'train':a.train,'val':a.val,'test':a.test}
    try:
        ds=load_dataset('OpenLLM-France/Claire-Dialogue-French-0.1',sample_by='paragraph',streaming=True)
        for original_split in ('train','test'):
            for row in ds[original_split]:
                raw=row['text']; ts=turns(raw)
                if len(ts)<a.min_turns or len({x['speaker'] for x in ts})<2: continue
                h=int(hashlib.sha256(raw.encode('utf8')).hexdigest()[:12],16)
                # Source test is never used in train or validation.
                split='test' if original_split=='test' else ('val' if h%10==0 else 'train')
                if counts[split]>=limits[split]:continue
                # Require alternating speakers for adjacent prediction examples.
                ts=[t for t in ts if t['text'] and len(t['text'])<=1000]
                if len(ts)<a.min_turns:continue
                start=(h//10)%max(1,len(ts)-a.max_turns+1)
                ts=ts[start:start+a.max_turns]
                pairs=[{'input':ts[i]['text'], 'target':ts[i+1]['text'], 'speaker':ts[i]['speaker'], 'target_speaker':ts[i+1]['speaker']}
                       for i in range(len(ts)-1) if ts[i]['speaker']!=ts[i+1]['speaker']]
                if len(pairs)<5:continue
                files[split].write(json.dumps({'id':hashlib.sha256(raw.encode()).hexdigest()[:16],'turns':pairs},ensure_ascii=False)+'\n')
                counts[split]+=1
                if all(counts[s]>=limits[s] for s in counts):break
            if all(counts[s]>=limits[s] for s in counts):break
    finally:
        for f in files.values():f.close()
    print('Saved:',counts)
    if counts['test']==0:print('Warning: no test samples; check dataset availability.')
if __name__=='__main__':main()
