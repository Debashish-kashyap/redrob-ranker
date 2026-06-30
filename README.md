# Redrob Hackathon — Candidate Ranker (Team XUINO)

Senior AI Engineer (Founding Team) @ Redrob AI — Intelligent Candidate Discovery & Ranking Challenge.

## Setup

Python 3.9+ required. Install dependencies:

```
pip install -r requirements.txt
```

## One-time setup (needs internet, done once)

```
python download_model.py
python precompute.py --candidates data/candidates.jsonl --model models/all-MiniLM-L6-v2
```

This downloads the sentence-transformers model (~22MB) and precomputes embeddings
for all candidates (~5-15 min depending on CPU). Saves `embeddings.npy` and
`candidate_ids.json`.

## Reproduce submission (offline, <5 min, CPU only)

```
python rank.py --candidates data/candidates.jsonl --embeddings embeddings.npy --ids candidate_ids.json --out output/team_XUINO.csv
```

If embeddings are not available, `rank.py` automatically falls back to TF-IDF
semantic scoring (no precompute step needed, still <2 min):

```
python rank.py --candidates data/candidates.jsonl --out output/team_XUINO.csv
```

## Validate

```
python validate_submission.py output/team_XUINO.csv
```

## Architecture

Three-layer hybrid ranking system:

**Layer 1 — Semantic Embeddings (30% weight)**
sentence-transformers/all-MiniLM-L6-v2 encodes each candidate's headline,
summary, current role, and top skills into a 384-dim vector. Four JD aspect
queries (core technical, evaluation/ML, production engineering, full JD) are
encoded and weighted cosine similarity is computed. This catches plain-language
candidates who describe real retrieval/ranking systems without buzzwords.

**Layer 2 — Rule-based Scoring (70% weight across 4 components)**
- Title and career track (32%): disqualifies off-domain titles, penalizes
  IT-services-only careers, rewards product company experience, detects
  job-hopping and title-chasing
- Skills (28%): matches JD-relevant keywords weighted by proficiency x
  duration x endorsements — catches keyword stuffers claiming "expert"
  with zero months of usage
- Experience (22%): ideal 5-9yr range, detects production ML signal in
  career descriptions
- Location (18%): Pune/Noida preferred, acceptable India cities scored,
  relocation willingness considered

**Layer 3 — Behavioral Multiplier**
Applied after layers 1+2. Uses all 23 redrob_signals: last_active_date
recency, open_to_work_flag, recruiter_response_rate, notice_period_days,
interview_completion_rate, github_activity_score, verification flags,
preferred_work_mode.

**Honeypot detection**
6 consistency checks: expert skill with 0 duration, impossible career
timeline, 10+ expert skills, implausible single-role duration, years-of-
experience inconsistent with oldest career entry, high endorsements with
0 duration. Requires 2+ flags to disqualify a candidate.

## Performance

- Runtime: ~25 seconds for 100,000 candidates (with precomputed embeddings)
- Precompute: ~5-15 minutes once, offline
- Honeypots in top 100: 0 (limit is <10)
- All scoring is rule-based + local embeddings — no external API calls,
  no GPU, fully reproducible offline

## Files

| File | Purpose |
|---|---|
| `rank.py` | Main ranking script — produces the submission CSV |
| `precompute.py` | One-time embedding precomputation |
| `download_model.py` | Downloads sentence-transformers model locally |
| `validate_submission.py` | Official format validator |
| `app.py` | HuggingFace Spaces sandbox demo (Gradio) |
| `submission_metadata.yaml` | Team and methodology metadata |

## Notes

This system is tuned to the specific Senior AI Engineer JD provided in this
hackathon (skill lists, location preferences, and disqualifier rules are
JD-specific). A production version would extract these parameters
dynamically from arbitrary JD text using the same embedding approach
already used for candidate matching.
