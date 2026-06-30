import csv

with open('output/team_XUINO.csv') as f:
    rows = list(csv.DictReader(f))

print('Total rows:', len(rows))
print()
print('TOP 10:')
for r in rows[:10]:
    rank = r['rank']
    cid = r['candidate_id']
    score = float(r['score'])
    reasoning = r['reasoning'][:90]
    print(f'  #{rank:>3}  {cid}  {score:.4f}  {reasoning}')

print()
print('BOTTOM 3:')
for r in rows[-3:]:
    rank = r['rank']
    cid = r['candidate_id']
    score = float(r['score'])
    reasoning = r['reasoning'][:90]
    print(f'  #{rank:>3}  {cid}  {score:.4f}  {reasoning}')