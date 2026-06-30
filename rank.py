#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker
Senior AI Engineer (Founding Team) @ Redrob AI

Architecture: Three-layer hybrid scoring system
  Layer 1 — Semantic Embeddings (30% weight)
             sentence-transformers/all-MiniLM-L6-v2
             Multi-aspect JD encoding: core technical + evaluation + production + full JD
             Catches plain-language Tier 5 candidates with no buzzwords
             Precomputed offline → loads in milliseconds at ranking time

  Layer 2 — Rule-based scoring (55% weight)
             Title + career trajectory (product vs IT services, job-hopping)
             Skills with duration/endorsement trust weighting
             Experience depth + production signals in descriptions
             Location fit

  Layer 3 — Behavioral multiplier (applied after layers 1+2)
             All 23 redrob_signals: recency, response rate, notice period,
             open_to_work, GitHub, interview completion, verification

  Honeypot detection: 6 consistency checks, disqualifies on 2+ flags

Runtime: ~5-10 seconds for 100K candidates (embeddings pre-loaded from disk)
Precompute: ~3-8 minutes once offline (saved to embeddings.npy)

Usage:
    # First time (with internet, done once):
    python download_model.py

    # Precompute embeddings (done once, ~5 min):
    python precompute.py --candidates data/candidates.jsonl --model models/all-MiniLM-L6-v2

    # Rank (fast, <2 min, no network):
    python rank.py --candidates data/candidates.jsonl --embeddings embeddings.npy --ids candidate_ids.json --out team_XUINO.csv

    # Fallback (no embeddings, uses TF-IDF only):
    python rank.py --candidates data/candidates.jsonl --out team_XUINO.csv
"""

import argparse
import csv
import gzip
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TODAY = datetime(2026, 6, 27)

# ── Multi-aspect JD queries ───────────────────────────────────────────────────
# Each aspect targets a different dimension of what the JD is actually asking for.
# We encode all 4, compute cosine similarity for each, then weighted-average.
# This gives richer semantic signal than a single query.

JD_ASPECTS = {
    "core_technical": """
        Production embeddings retrieval ranking recommendation systems search
        vector database dense retrieval hybrid search sentence transformers BERT
        FAISS Pinecone Weaviate Qdrant Milvus Elasticsearch OpenSearch
        information retrieval semantic search approximate nearest neighbor
        embedding drift index refresh retrieval quality regression
        learning to rank reranking cross encoder bi-encoder
    """,

    "evaluation_ml": """
        Evaluation framework ranking systems NDCG MRR MAP mean average precision
        offline evaluation online AB testing recruiter engagement metrics
        relevance labeling offline online correlation precision recall
        experiment design statistical significance metric design
        XGBoost LightGBM gradient boosting feature engineering
        learning to rank neural ranking models
    """,

    "production_engineering": """
        Shipped deployed production real users scale latency throughput
        inference serving API endpoint machine learning pipeline
        product company startup iterate ship fast engineering
        Python code quality software engineering applied machine learning
        end to end ML system deployment monitoring data pipeline
        million users queries per second production ML engineer
    """,

    "full_jd": """
        Senior AI Engineer founding team Redrob AI talent intelligence platform
        Series A machine learning embeddings retrieval ranking recommendation search
        vector database hybrid search evaluation NDCG MRR MAP AB testing
        sentence transformers FAISS Pinecone Weaviate Qdrant Elasticsearch
        learning to rank XGBoost LightGBM fine-tuning LoRA QLoRA
        Python product company startup scrappy ship deployed production
        NLP natural language processing transformers hugging face
        candidate job description matching recruiter search shortlist
        Pune Noida India hybrid 5 to 9 years experience
        not IT services not pure research not title chaser writes code
    """
}

# Weights for each JD aspect in the final semantic score
JD_ASPECT_WEIGHTS = {
    "core_technical":      0.35,
    "evaluation_ml":       0.25,
    "production_engineering": 0.25,
    "full_jd":             0.15,
}

# ── Title classification ──────────────────────────────────────────────────────
HARD_DISQUALIFIER_TITLES = {
    "marketing manager", "accountant", "hr manager", "human resources manager",
    "graphic designer", "civil engineer", "mechanical engineer",
    "customer support", "customer service", "operations manager",
    "business analyst", "project manager", "java developer", ".net developer",
    "net developer", "product manager", "sales manager", "finance manager",
    "legal counsel", "chartered accountant", "supply chain manager",
    "procurement manager", "content writer", "ux designer", "ui designer",
    "scrum master", "agile coach", "program manager",
}

GOOD_TITLE_KEYWORDS = [
    "machine learning", "ml engineer", "ai engineer", "nlp engineer",
    "data scientist", "applied scientist", "research engineer",
    "recommendation", "search engineer", "retrieval", "ranking engineer",
    "applied ml", "deep learning", "software engineer", "backend engineer",
    "full stack", "platform engineer", "data engineer",
]

PURE_RESEARCH_TITLES = {
    "research scientist", "phd student", "postdoc",
    "principal researcher", "research fellow",
}

# ── Company classification ────────────────────────────────────────────────────
IT_SERVICES_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "persistent systems", "kpit",
    "mastek", "niit technologies", "zensar", "cyient",
}

PRODUCT_COMPANY_KEYWORDS = [
    "swiggy", "zomato", "flipkart", "meesho", "ola", "rapido", "cred",
    "razorpay", "zerodha", "groww", "paytm", "phonepe", "nykaa",
    "freshworks", "zoho", "chargebee", "browserstack", "postman",
    "google", "microsoft", "amazon", "meta", "apple", "netflix",
    "uber", "airbnb", "stripe", "databricks", "openai", "anthropic",
    "mad street den", "lenskart", "myntra", "bigbasket", "dunzo",
    "urban company", "cars24", "delhivery", "porter", "slice",
    "smallcase", "leadsquared", "whatfix", "unacademy", "byju",
    "vedantu", "sharechat", "moj", "dailyhunt", "inmobi", "truecaller",
    "sarvam", "krutrim", "haptik", "rephrase", "locobuzz", "observe",
    "sprinklr", "clevertap", "moengage", "webengage", "niramai",
    "saarthi", "observe.ai", "niki", "vernacular", "skit",
]

# ── Skill keywords ────────────────────────────────────────────────────────────
MUST_HAVE_SKILLS = {
    "embeddings", "embedding", "sentence transformers", "sentence-transformers",
    "bge", "e5", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "vector database", "vector db",
    "vector search", "hybrid search", "information retrieval", "retrieval",
    "ranking", "reranking", "re-ranking", "learning to rank", "bm25",
    "tfidf", "tf-idf", "ndcg", "mrr", "map", "a/b testing",
    "python", "hugging face", "huggingface", "transformers",
    "fine-tuning", "fine tuning", "lora", "qlora", "peft",
    "xgboost", "lightgbm", "scikit-learn", "sklearn", "pytorch",
    "mlflow", "mlops", "nlp", "natural language processing",
    "recommendation", "recommendation system", "collaborative filtering",
    "feature engineering", "chroma", "chromadb",
}

CV_SPEECH_ONLY = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "speech recognition", "tts", "text-to-speech",
    "asr", "gans",
}

# ── Location ──────────────────────────────────────────────────────────────────
PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "gurgaon", "gurugram", "faridabad",
    "greater noida", "ncr", "new delhi",
}
ACCEPTABLE_INDIA_LOCATIONS = {
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai",
    "kolkata", "ahmedabad", "jaipur", "lucknow", "indore",
}

# ── Score blend weights ───────────────────────────────────────────────────────
SEMANTIC_WEIGHT = 0.30   # sentence-BERT embedding similarity
RULE_WEIGHT     = 0.70   # rule-based components below

RULE_COMPONENT_WEIGHTS = {
    "title_career": 0.32,
    "skills":       0.28,
    "experience":   0.22,
    "location":     0.18,
}


# ─────────────────────────────────────────────────────────────────────────────
# SEMANTIC SCORING — JD Auto-understanding via sentence embeddings
# ─────────────────────────────────────────────────────────────────────────────

def encode_jd_aspects(model) -> dict:
    """
    Encode each JD aspect into a normalized embedding vector.
    These are the 'query vectors' for semantic candidate matching.
    """
    print("  Encoding JD aspects...", flush=True)
    aspect_embeddings = {}
    for aspect, text in JD_ASPECTS.items():
        emb = model.encode([text.strip()], normalize_embeddings=True)
        aspect_embeddings[aspect] = emb[0]
        print(f"    ✓ {aspect}", flush=True)
    return aspect_embeddings


def compute_semantic_scores_from_embeddings(
    candidate_embeddings: np.ndarray,
    aspect_embeddings: dict
) -> np.ndarray:
    """
    Compute weighted multi-aspect semantic similarity scores.
    candidate_embeddings: shape (N, 384), already L2-normalized
    Returns: shape (N,) scores in [0, 1]
    """
    scores = np.zeros(len(candidate_embeddings), dtype=np.float32)

    for aspect, weight in JD_ASPECT_WEIGHTS.items():
        query_vec = aspect_embeddings[aspect].astype(np.float32)
        # Dot product = cosine similarity (both are normalized)
        aspect_scores = candidate_embeddings @ query_vec
        scores += weight * aspect_scores

    # Normalize to [0, 1]
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)

    return scores


def compute_tfidf_scores_fallback(candidates: list) -> np.ndarray:
    """TF-IDF fallback when embeddings are not available."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    full_jd = " ".join(JD_ASPECTS.values())

    def build_text(c):
        parts = [
            c["profile"].get("headline", ""),
            c["profile"].get("summary", ""),
            c["profile"].get("current_title", ""),
        ]
        for role in c.get("career_history", []):
            parts.append(role.get("title", ""))
            parts.append(role.get("description", ""))
        for s in c.get("skills", []):
            repeat = {"beginner":1,"intermediate":2,"advanced":3,"expert":5}.get(
                s.get("proficiency","beginner"), 1)
            parts.extend([s["name"]] * repeat)
        return " ".join(filter(None, parts))

    corpus = [full_jd] + [build_text(c) for c in candidates]
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=10000,
                          sublinear_tf=True, min_df=2, stop_words="english")
    matrix = vec.fit_transform(corpus)
    sims = cosine_similarity(matrix[0], matrix[1:])[0]
    max_s = sims.max()
    return (sims / max_s) if max_s > 0 else sims


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> tuple:
    flags = []
    profile = candidate["profile"]
    career  = candidate["career_history"]
    skills  = candidate["skills"]

    expert_zero = [s["name"] for s in skills
                   if s["proficiency"] == "expert" and s.get("duration_months", 1) == 0]
    if expert_zero:
        flags.append(f"expert_zero_duration:{expert_zero[:2]}")

    total_career = sum(r.get("duration_months", 0) for r in career)
    claimed      = profile["years_of_experience"] * 12
    if total_career > claimed + 18:
        flags.append(f"impossible_timeline:{total_career:.0f}>{claimed:.0f}mo")

    if sum(1 for s in skills if s["proficiency"] == "expert") >= 10:
        flags.append("too_many_experts")

    for role in career:
        if role.get("duration_months", 0) > 240:
            flags.append(f"implausible_role:{role['company']}")

    if career:
        try:
            oldest = min(datetime.strptime(r["start_date"], "%Y-%m-%d")
                         for r in career if r.get("start_date"))
            implied = (TODAY - oldest).days / 30.44
            if claimed > implied + 24:
                flags.append(f"yoe_vs_career:{claimed:.0f}>{implied:.0f}mo")
        except (ValueError, TypeError):
            pass

    endorsed_zero = [s["name"] for s in skills
                     if s.get("endorsements", 0) > 20 and s.get("duration_months", 1) == 0]
    if endorsed_zero:
        flags.append(f"endorsed_zero_dur:{endorsed_zero[:2]}")

    return len(flags) >= 2, flags


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BASED SCORING COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def score_title_and_career(candidate: dict) -> tuple:
    profile = candidate["profile"]
    career  = candidate["career_history"]
    notes   = []

    title_lower = profile["current_title"].lower().strip()

    if any(dt in title_lower for dt in HARD_DISQUALIFIER_TITLES):
        return 0.05, [f"off-domain title '{profile['current_title']}'"]

    if any(rt in title_lower for rt in PURE_RESEARCH_TITLES):
        title_score = 0.35
        notes.append("pure research title — production signal absent")
    elif any(gt in title_lower for gt in GOOD_TITLE_KEYWORDS):
        title_score = 1.0
        notes.append(f"'{profile['current_title']}' — strong AI/ML/Search title")
    else:
        title_score = 0.55
        notes.append(f"neutral title '{profile['current_title']}'")

    if not career:
        return title_score * 0.5, notes

    total_months = max(1, sum(r.get("duration_months", 0) for r in career))
    it_months = product_months = research_months = 0
    job_hop_penalty = 0.0
    has_code_role = False

    for role in career:
        company  = role.get("company", "").lower()
        industry = role.get("industry", "").lower()
        dur      = role.get("duration_months", 0)
        title_r  = role.get("title", "").lower()

        if any(it in company for it in IT_SERVICES_COMPANIES) or industry == "it services":
            it_months += dur
        if any(pc in company for pc in PRODUCT_COMPANY_KEYWORDS):
            product_months += dur
        if any(rt in title_r for rt in PURE_RESEARCH_TITLES):
            research_months += dur
        if not role.get("is_current", False) and 0 < dur < 12:
            job_hop_penalty += 0.04
        if role.get("is_current", False) and any(
            ct in title_r for ct in ["engineer", "developer", "scientist", "architect"]
        ):
            has_code_role = True

    it_ratio      = it_months / total_months
    product_ratio = product_months / total_months
    research_ratio= research_months / total_months

    if product_ratio > 0.5:
        career_score = 1.0
        notes.append(f"majority career at product companies ({product_ratio*100:.0f}%)")
    elif product_ratio > 0.25:
        career_score = 0.75
        notes.append(f"product company experience ({product_ratio*100:.0f}%)")
    elif it_ratio > 0.8:
        career_score = 0.15
        notes.append("career almost entirely in IT services")
    elif it_ratio > 0.5:
        career_score = 0.35
        notes.append("majority career in IT services")
    else:
        career_score = 0.55

    if research_ratio > 0.5:
        career_score *= 0.5
        notes.append("significant pure research background")
    if not has_code_role:
        career_score *= 0.8
        notes.append("current role may not involve writing code")

    job_hop_penalty = min(job_hop_penalty, 0.20)
    if job_hop_penalty > 0.08:
        notes.append("multiple short tenures — title-chasing signal")

    final = (title_score * 0.5 + career_score * 0.5) - job_hop_penalty
    return max(0.0, min(1.0, final)), notes


def score_skills(candidate: dict) -> tuple:
    skills = candidate["skills"]
    notes  = []

    if not skills:
        return 0.0, ["no skills listed"]

    skill_lookup = {s["name"].lower(): s for s in skills}
    matched, total_score = [], 0.0

    for must in MUST_HAVE_SKILLS:
        key = next((sn for sn in skill_lookup if must in sn or sn in must), None)
        if key:
            s         = skill_lookup[key]
            dur       = s.get("duration_months", 0)
            end       = s.get("endorsements", 0)
            prof_mult = {"beginner":0.4,"intermediate":0.7,"advanced":0.9,"expert":1.0}.get(
                        s.get("proficiency","beginner"), 0.5)
            dur_trust = min(1.0, dur / 24.0) if dur > 0 else 0.1
            end_bonus = min(0.3, end / 100.0)
            val       = prof_mult * dur_trust + end_bonus
            matched.append((s["name"], val))
            total_score += val

    normalized = min(1.0, total_score / 10.0)

    if matched:
        seen, deduped = set(), []
        for n, v in sorted(matched, key=lambda x: -x[1]):
            if n.lower() not in seen:
                seen.add(n.lower())
                deduped.append((n, v))
        notes.append(
            f"{len(deduped)} JD-relevant skills: "
            + ", ".join(n for n, _ in deduped[:5])
        )
    else:
        return 0.05, ["no JD-relevant skills found"]

    cv_only = {sn for sn in skill_lookup if any(cv in sn for cv in CV_SPEECH_ONLY)}
    nlp_ir  = {sn for sn in skill_lookup if any(
        k in sn for k in ["nlp","retrieval","embedding","ranking","recommendation","search","transformer"]
    )}
    if cv_only and not nlp_ir:
        normalized *= 0.5
        notes.append("skills skew CV/Speech without NLP/IR")

    if not any("python" in sn for sn in skill_lookup):
        normalized *= 0.85
        notes.append("no Python listed")

    return max(0.0, min(1.0, normalized)), notes


def score_experience(candidate: dict) -> tuple:
    profile = candidate["profile"]
    career  = candidate["career_history"]
    notes   = []

    yoe = profile["years_of_experience"]
    if   5  <= yoe <= 9:  yoe_score = 1.00; notes.append(f"{yoe:.1f}yr (ideal 5-9)")
    elif 4  <= yoe < 5:   yoe_score = 0.85; notes.append(f"{yoe:.1f}yr (slightly below ideal)")
    elif 9  < yoe <= 12:  yoe_score = 0.80; notes.append(f"{yoe:.1f}yr (above ideal)")
    elif 3  <= yoe < 4:   yoe_score = 0.60; notes.append(f"{yoe:.1f}yr (below minimum)")
    elif yoe > 12:        yoe_score = 0.65; notes.append(f"{yoe:.1f}yr (very senior)")
    else:                 yoe_score = 0.20; notes.append(f"{yoe:.1f}yr (too junior)")

    prod_kw = ["shipped","deployed","production","real users","a/b test","latency",
               "throughput","inference","serving","api","scale","million","qps","sla"]
    ml_kw   = ["embedding","retrieval","ranking","recommendation","search","nlp",
               "transformer","fine-tun","vector","index","ndcg","mrr","map","recall"]

    prod_sig = ml_sig = 0
    for role in career:
        desc = role.get("description", "").lower()
        prod_sig += sum(1 for kw in prod_kw if kw in desc)
        ml_sig   += sum(1 for kw in ml_kw   if kw in desc)

    if ml_sig >= 5:
        yoe_score = min(1.0, yoe_score * 1.15)
        notes.append("strong ML depth in career descriptions")
    elif ml_sig == 0:
        yoe_score *= 0.7
        notes.append("no ML keywords in career history")

    if prod_sig >= 3:
        yoe_score = min(1.0, yoe_score * 1.10)
        notes.append("production deployment evidence")

    return max(0.0, min(1.0, yoe_score)), notes


def score_location(candidate: dict) -> tuple:
    profile  = candidate["profile"]
    signals  = candidate["redrob_signals"]
    notes    = []
    location = profile.get("location", "").lower()
    country  = profile.get("country",  "").lower()
    relocate = signals.get("willing_to_relocate", False)

    if country != "india":
        if relocate:
            notes.append(f"outside India ({profile['country']}), willing to relocate")
            return 0.35, notes
        notes.append(f"outside India ({profile['country']})")
        return 0.10, notes

    if any(pl in location for pl in PREFERRED_LOCATIONS):
        notes.append(f"preferred location ({profile['location']})")
        return 1.0, notes
    if any(al in location for al in ACCEPTABLE_INDIA_LOCATIONS):
        notes.append(f"good India location ({profile['location']})")
        return 0.80, notes
    if relocate:
        notes.append(f"India ({profile['location']}), willing to relocate")
        return 0.65, notes
    notes.append(f"India ({profile['location']}), not willing to relocate")
    return 0.45, notes


def score_behavioral(candidate: dict) -> tuple:
    signals = candidate["redrob_signals"]
    notes   = []
    score   = 1.0

    try:
        days_ago = (TODAY - datetime.strptime(signals["last_active_date"], "%Y-%m-%d")).days
        if   days_ago <= 14:  notes.append(f"very recently active ({days_ago}d ago)")
        elif days_ago <= 30:  score *= 0.97; notes.append(f"active {days_ago}d ago")
        elif days_ago <= 60:  score *= 0.90
        elif days_ago <= 90:  score *= 0.80; notes.append(f"inactive {days_ago}d (borderline)")
        elif days_ago <= 180: score *= 0.60; notes.append(f"inactive {days_ago}d (concerning)")
        else:                 score *= 0.35; notes.append(f"inactive {days_ago}d (likely unavailable)")
    except (ValueError, TypeError, KeyError):
        score *= 0.80

    if not signals.get("open_to_work_flag", False):
        score *= 0.85; notes.append("not open to work")
    else:
        notes.append("open to work")

    rrr = signals.get("recruiter_response_rate", 0.5)
    if   rrr >= 0.7: notes.append(f"high response rate ({rrr:.0%})")
    elif rrr >= 0.4: score *= 0.95
    elif rrr >= 0.2: score *= 0.85; notes.append(f"low response rate ({rrr:.0%})")
    else:            score *= 0.65; notes.append(f"very low response rate ({rrr:.0%}) — ghost risk")

    notice = signals.get("notice_period_days", 60)
    if   notice <= 30:  notes.append(f"short notice ({notice}d)")
    elif notice <= 60:  score *= 0.95
    elif notice <= 90:  score *= 0.85; notes.append(f"notice {notice}d")
    else:               score *= 0.70; notes.append(f"notice {notice}d (very long)")

    if signals.get("interview_completion_rate", 0.7) < 0.5:
        score *= 0.85; notes.append("low interview completion")

    gh = signals.get("github_activity_score", -1)
    if   gh == -1:  score *= 0.95; notes.append("no GitHub linked")
    elif gh >= 50:  notes.append(f"active GitHub ({gh})")

    if not signals.get("verified_email", True):  score *= 0.92
    if not signals.get("verified_phone", True):  score *= 0.92
    if signals.get("preferred_work_mode", "flexible") == "remote":
        score *= 0.90; notes.append("prefers remote (JD is hybrid)")

    return max(0.20, min(1.0, score)), notes


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate(candidate: dict, semantic_score: float) -> tuple:
    trap, trap_flags = is_honeypot(candidate)
    if trap:
        return 0.001, (
            f"Profile has {len(trap_flags)} consistency issues "
            f"({trap_flags[0].split(':')[0]}). Excluded as likely dataset artefact."
        )

    title_score, title_notes = score_title_and_career(candidate)
    skill_score, skill_notes = score_skills(candidate)
    exp_score,   exp_notes   = score_experience(candidate)
    loc_score,   loc_notes   = score_location(candidate)
    beh_mult,    beh_notes   = score_behavioral(candidate)

    if title_score <= 0.05:
        rule_base = 0.05
    else:
        rule_base = (
            RULE_COMPONENT_WEIGHTS["title_career"] * title_score
            + RULE_COMPONENT_WEIGHTS["skills"]      * skill_score
            + RULE_COMPONENT_WEIGHTS["experience"]  * exp_score
            + RULE_COMPONENT_WEIGHTS["location"]    * loc_score
        )

    # Blend: 70% rule-based (trap-resistant) + 30% semantic (meaning-aware)
    blended = RULE_WEIGHT * rule_base + SEMANTIC_WEIGHT * semantic_score

    # Behavioral multiplier applied last
    composite = blended * beh_mult

    # ── Reasoning string ──────────────────────────────────────────────────
    p       = candidate["profile"]
    signals = candidate["redrob_signals"]
    yoe     = p["years_of_experience"]
    title   = p["current_title"]
    company = p["current_company"]
    notice  = signals.get("notice_period_days", "?")

    if title_score <= 0.05:
        s1 = f"Title '{title}' at {company} is off-domain for Senior AI Engineer."
    elif title_score > 0.7 and skill_score > 0.5:
        s1 = (f"{title} at {company} with {yoe:.1f}yr experience; "
              f"{skill_notes[0] if skill_notes else 'relevant skills present'}.")
    elif semantic_score > 0.6 and skill_score < 0.3:
        s1 = (f"{title} at {company} ({yoe:.1f}yr); career descriptions show "
              f"strong semantic alignment with JD (score {semantic_score:.2f}) "
              f"despite limited explicit keyword matches.")
    else:
        s1 = (f"{title} at {company} ({yoe:.1f}yr); "
              f"{title_notes[0] if title_notes else 'career reviewed'}.")

    concerns, positives = [], []
    if notice > 60:
        concerns.append(f"notice period {notice}d")
    for n in beh_notes:
        if any(w in n for w in ["inactive", "response rate", "ghost", "remote"]):
            concerns.append(n); break
    if loc_score < 0.5:
        concerns.append(loc_notes[0] if loc_notes else "location mismatch")

    if not concerns:
        for n in beh_notes:
            if any(w in n for w in ["open to work", "response", "active", "GitHub"]):
                positives.append(n); break
        if exp_notes and any("production" in n for n in exp_notes):
            positives.append("production deployment evidence")
        if loc_score >= 0.8:
            positives.append(loc_notes[0] if loc_notes else "good location")

    s2 = ("Concerns: " + "; ".join(concerns[:2]) + ".") if concerns else \
         ("Positives: " + "; ".join(positives[:2]) + ".") if positives else \
         f"Semantic fit score {semantic_score:.2f}."

    reasoning = f"{s1} {s2}"
    return max(0.0, min(1.0, composite)), reasoning[:300]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str) -> list:
    p = Path(path)
    opener = (lambda: gzip.open(p, "rt", encoding="utf-8")) if p.suffix == ".gz" \
             else (lambda: open(p, "r", encoding="utf-8"))
    candidates = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def run(candidates_path: str, out_path: str,
        embeddings_path: str = None, ids_path: str = None):

    t_start = time.time()

    print(f"Loading candidates from {candidates_path}...", flush=True)
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates):,} candidates.", flush=True)

    # ── Semantic scoring ──────────────────────────────────────────────────
    use_embeddings = embeddings_path and ids_path and \
                     Path(embeddings_path).exists() and Path(ids_path).exists()

    if use_embeddings:
        print(f"\n[Step 1/3] Loading precomputed embeddings from {embeddings_path}...", flush=True)
        embeddings = np.load(embeddings_path)
        with open(ids_path) as f:
            emb_ids = json.load(f)

        # Build id→index map for fast lookup
        id_to_idx = {cid: i for i, cid in enumerate(emb_ids)}

        # Load model to encode JD aspects only (fast — 4 short texts)
        print("  Loading model for JD encoding...", flush=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(
            str(Path(embeddings_path).parent / "models" / "all-MiniLM-L6-v2")
            if (Path(embeddings_path).parent / "models" / "all-MiniLM-L6-v2").exists()
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        aspect_embeddings = encode_jd_aspects(model)

        print("  Computing multi-aspect semantic scores...", flush=True)
        # Reorder embeddings to match current candidates list
        ordered_embs = np.zeros((len(candidates), embeddings.shape[1]), dtype=np.float32)
        missing = 0
        for i, c in enumerate(candidates):
            idx = id_to_idx.get(c["candidate_id"])
            if idx is not None:
                ordered_embs[i] = embeddings[idx]
            else:
                missing += 1

        if missing > 0:
            print(f"  Warning: {missing} candidates missing from embeddings file", flush=True)

        semantic_scores = compute_semantic_scores_from_embeddings(ordered_embs, aspect_embeddings)
        print(f"  Semantic scores: range [{semantic_scores.min():.4f}, {semantic_scores.max():.4f}]", flush=True)
        mode = "sentence-BERT embeddings"

    else:
        print(f"\n[Step 1/3] Computing TF-IDF semantic scores (fallback)...", flush=True)
        print("  Tip: Run precompute.py first for better sentence-BERT semantic scoring.", flush=True)
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        semantic_scores = compute_tfidf_scores_fallback(candidates)
        print(f"  TF-IDF scores: range [{semantic_scores.min():.4f}, {semantic_scores.max():.4f}]", flush=True)
        mode = "TF-IDF (fallback)"

    # ── Rule-based + blend scoring ────────────────────────────────────────
    print(f"\n[Step 2/3] Scoring all candidates (rules + {mode})...", flush=True)
    scored        = []
    honeypot_count = 0

    for i, (c, sem_s) in enumerate(zip(candidates, semantic_scores)):
        if i > 0 and i % 10000 == 0:
            print(f"  {i:,} / {len(candidates):,}...", flush=True)

        score, reasoning = score_candidate(c, float(sem_s))
        scored.append((score, c["candidate_id"], reasoning))
        if score <= 0.001:
            honeypot_count += 1

    print(f"  Done. Detected {honeypot_count} likely honeypots.", flush=True)

    # ── Rank and write ────────────────────────────────────────────────────
    print(f"\n[Step 3/3] Selecting top 100 and writing CSV...", flush=True)
    scored.sort(key=lambda x: (-x[0], x[1]))
    top100 = scored[:100]

    hp_top100 = sum(1 for s, _, _ in top100 if s <= 0.001)
    print(f"  Honeypots in top 100: {hp_top100} (must be <10)", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning) in enumerate(top100, start=1):
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    elapsed = time.time() - t_start
    print(f"\n✓ Submission written to {out_path}")
    print(f"✓ Semantic mode: {mode}")
    print(f"✓ Total runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"\nTop 5 candidates:")
    for rank, (score, cid, reasoning) in enumerate(top100[:5], start=1):
        print(f"  #{rank}: {cid}  score={score:.4f}")
        print(f"         {reasoning[:110]}...")


def main():
    parser = argparse.ArgumentParser(description="Redrob Hackathon candidate ranker v3")
    parser.add_argument("--candidates",  required=True,
                        help="Path to candidates.jsonl or .jsonl.gz")
    parser.add_argument("--out",         required=True,
                        help="Output CSV path")
    parser.add_argument("--embeddings",  default=None,
                        help="Path to precomputed embeddings.npy (optional)")
    parser.add_argument("--ids",         default=None,
                        help="Path to candidate_ids.json (optional)")
    args = parser.parse_args()
    run(args.candidates, args.out, args.embeddings, args.ids)


if __name__ == "__main__":
    main()