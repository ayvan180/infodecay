from datasets import load_dataset
import json

print('Downloading HotpotQA (distractor setting)...')
ds = load_dataset('hotpot_qa', 'distractor', split='validation')

# Filter out yes/no questions — keep only factoid questions
# Yes/no answers cause IDI problems because they match everything
factoid = [s for s in ds if s.get('answer', '').lower()
           not in ('yes', 'no', 'true', 'false')]

samples = factoid[:500]
with open('data/hotpotqa_dev_500.jsonl', 'w', encoding='utf-8') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')

print(f'Saved {len(samples)} factoid samples to data/hotpotqa_dev_500.jsonl')
print('Done!')