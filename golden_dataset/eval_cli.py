#!/usr/bin/env python
"""
RAG Evaluation CLI
==================
Tests every question in golden_dataset/rag_eval_questions.json against the
live FastAPI backend and prints a colour-coded report.

Usage:
    uv run python golden_dataset/eval_cli.py
    uv run python golden_dataset/eval_cli.py --url http://localhost:8000 --file golden_dataset/rag_eval_questions.json
    uv run python golden_dataset/eval_cli.py --id ahkam_001        # run single question
    uv run python golden_dataset/eval_cli.py --source ahkam.pdf    # run one file's questions
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import httpx

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

API_BASE     = "http://localhost:8000"
QUERY_PATH   = "/api/v1/query"
DATASET_FILE = Path(__file__).parent / "rag_eval_questions.json"
TIMEOUT      = 120.0  # seconds per question (LLM can be slow)


# ── Keyword scoring ────────────────────────────────────────────────────────────

def keyword_score(answer: str, expected_phrases: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Check how many of the expected key phrases appear in the answer.
    Returns (score 0-1, hits, misses).
    """
    hits, misses = [], []
    lower_answer = answer.lower()
    for phrase in expected_phrases:
        # Normalise: remove tashkeel for comparison
        normalised_phrase = _strip_tashkeel(phrase.lower())
        normalised_answer = _strip_tashkeel(lower_answer)
        if normalised_phrase in normalised_answer:
            hits.append(phrase)
        else:
            misses.append(phrase)
    total = len(expected_phrases)
    score = len(hits) / total if total else 1.0
    return score, hits, misses


def _strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) for fuzzy matching."""
    # Unicode range for Arabic diacritics: U+064B – U+065F
    return re.sub(r"[\u064B-\u065F]", "", text)


import httpx
from openai import OpenAI

# ── LLM scoring ────────────────────────────────────────────────────────────────

def llm_score(question: str, ground_truth: str, answer: str, api_key: str) -> float:
    """
    Use GPT-4o as a judge to score the answer against the ground truth.
    Returns a score between 0.0 and 1.0.
    """
    client = OpenAI(api_key=api_key)
    prompt = f"""
    You are a legal expert judge evaluating a RAG (Retrieval-Augmented Generation) system.
    Evaluate the following answer against the provided ground truth for the given question.
    
    Question: {question}
    Ground Truth: {ground_truth}
    RAG Answer: {answer}
    
    Scoring Guidelines:
    - 1.0: Perfect. Covers all key points from the ground truth accurately.
    - 0.8: Mostly correct. Covers major points but misses minor details.
    - 0.5: Partially correct. Covers some points but misses significant information or has some inaccuracies.
    - 0.2: Poor. Barely addresses the question or has major hallucinations.
    - 0.0: Irrelevant or completely wrong.
    
    Provide ONLY the numeric score as a float (e.g., 0.8). No explanation.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        score_str = response.choices[0].message.content.strip()
        return float(re.search(r"(\d\.\d|\d)", score_str).group(1))
    except Exception as exc:
        print(f"{RED}LLM judging failed: {exc}{RESET}")
        return 0.0


# ── API caller ─────────────────────────────────────────────────────────────────

def query_api(question: str, base_url: str) -> dict:
    """POST to /api/v1/query and return the JSON response."""
    url = base_url.rstrip("/") + QUERY_PATH
    payload = {"query": question}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


# ── Printing helpers ───────────────────────────────────────────────────────────

def hr(char="─", width=80):
    print(DIM + char * width + RESET)

def print_header(text: str):
    print()
    print(BOLD + CYAN + "═" * 80 + RESET)
    print(BOLD + CYAN + f"  {text}" + RESET)
    print(BOLD + CYAN + "═" * 80 + RESET)

def print_question_result(
    q: dict,
    answer: str,
    score: float,
    hits: list[str],
    misses: list[str],
    latency_ms: float,
    cached: bool,
    error: str | None,
    index: int,
    total: int,
    mode: str,
):
    status_icon = "✅" if score >= 0.6 else ("⚠️ " if score >= 0.3 else "❌")
    color = GREEN if score >= 0.6 else (YELLOW if score >= 0.3 else RED)

    print()
    hr()
    print(
        f"{BOLD}[{index}/{total}] {q['id']}{RESET}  "
        f"{DIM}({q['source']}){RESET}  "
        f"{color}{status_icon} Score: {score:.0%}{RESET}  "
        f"{DIM}axes: {', '.join(q.get('difficulty_axes', []))}{RESET}"
    )
    hr()

    print(f"\n{BOLD}❓ Question:{RESET}")
    print(f"   {q['question']}\n")

    if error:
        print(f"{RED}⚡ Error: {error}{RESET}\n")
        return

    print(f"{BOLD}💬 Answer{RESET} {DIM}({'cached' if cached else f'{latency_ms:.0f}ms'}){RESET}:")
    # Wrap answer to 76 chars, indent 3 spaces
    words = answer.split()
    line, lines = "   ", []
    for w in words:
        if len(line) + len(w) + 1 > 79:
            lines.append(line)
            line = "   " + w
        else:
            line += (" " if line.strip() else "") + w
    lines.append(line)
    print("\n".join(lines))

    if mode == "keyword":
        print(f"\n{BOLD}🔍 Keyword Check:{RESET}")
        for h in hits:
            print(f"   {GREEN}✓{RESET}  {h}")
        for m in misses:
            print(f"   {RED}✗{RESET}  {m}")
    else:
        print(f"\n{BOLD}🤖 LLM Judge Score: {color}{score:.0%}{RESET}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument("--url",    default=API_BASE,        help="FastAPI base URL")
    parser.add_argument("--file",   default=str(DATASET_FILE), help="Path to questions JSON")
    parser.add_argument("--id",     default=None,            help="Run a single question by id")
    parser.add_argument("--source", default=None,            help="Filter to one source file")
    parser.add_argument("--threshold", type=float, default=0.6, help="Pass threshold (0-1)")
    parser.add_argument("--mode",   choices=["keyword", "llm"], default="keyword", help="Scoring mode")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key for LLM judge")
    args = parser.parse_args()

    dataset_path = Path(args.file)
    if not dataset_path.exists():
        print(f"{RED}Dataset file not found: {dataset_path}{RESET}")
        sys.exit(1)

    data    = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data["questions"]

    # Filtering
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
        if not questions:
            print(f"{RED}No question with id '{args.id}'{RESET}")
            sys.exit(1)
    if args.source:
        questions = [q for q in questions if args.source in q.get("source", "")]

    print_header(f"LegalMind RAG Evaluation  ({len(questions)} questions)")
    print(f"  API  : {CYAN}{args.url}{RESET}")
    print(f"  File : {dataset_path.name}")
    print(f"  Mode : {BOLD}{args.mode}{RESET}")
    print(f"  Pass : score ≥ {args.threshold:.0%}")

    # Verify API is up
    try:
        httpx.get(args.url.rstrip("/") + "/api/v1/health", timeout=5).raise_for_status()
        print(f"  Health: {GREEN}OK{RESET}")
    except Exception as exc:
        print(f"  Health: {RED}UNREACHABLE ({exc}){RESET} — answers will fail")

    # LLM judge prep
    openai_key = args.openai_key
    if args.mode == "llm" and not openai_key:
        # Try to find it in environment or .env
        import os
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            # Try parsing .env
            env_path = Path(".env")
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        openai_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        
        if not openai_key:
            print(f"{RED}Error: --mode llm requires --openai-key or OPENAI_API_KEY environment variable{RESET}")
            sys.exit(1)

    total   = len(questions)
    passed  = 0
    results = []

    for i, q in enumerate(questions, 1):
        error     = None
        answer    = ""
        latency   = 0.0
        cached    = False
        score     = 0.0
        hits: list[str]   = []
        misses: list[str] = []

        try:
            t0   = time.monotonic()
            resp = query_api(q["question"], args.url)
            latency = (time.monotonic() - t0) * 1000
            answer  = resp.get("answer", "")
            cached  = resp.get("cached", False)
            if cached:
                latency = resp.get("latency_ms", latency)
            
            if args.mode == "keyword":
                score, hits, misses = keyword_score(answer, q.get("expected_answer_contains", []))
            else:
                score = llm_score(q["question"], q["ground_truth"], answer, openai_key)
        except Exception as exc:
            error = str(exc)
            score = 0.0

        if score >= args.threshold:
            passed += 1

        results.append({
            "id":     q["id"],
            "source": q["source"],
            "score":  score,
            "passed": score >= args.threshold,
            "error":  error,
        })

        print_question_result(q, answer, score, hits, misses, latency, cached, error, i, total, args.mode)

    # ── Summary ────────────────────────────────────────────────────────────────
    print_header("Summary")

    pass_rate = passed / total if total else 0
    color = GREEN if pass_rate >= 0.8 else (YELLOW if pass_rate >= 0.5 else RED)
    print(f"  Overall pass rate: {color}{BOLD}{passed}/{total}  ({pass_rate:.0%}){RESET}\n")

    # Per-source breakdown
    sources = sorted({r["source"] for r in results})
    for src in sources:
        src_results = [r for r in results if r["source"] == src]
        src_pass    = sum(1 for r in src_results if r["passed"])
        src_total   = len(src_results)
        avg_score   = sum(r["score"] for r in src_results) / src_total if src_total else 0
        denominator = src_total if src_total else 1
        c = GREEN if (src_pass / denominator) >= 0.6 else YELLOW
        print(f"  {src:<40}  {c}{src_pass}/{src_total} passed  avg={avg_score:.0%}{RESET}")

    # Failed questions list
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  {RED}Failed questions:{RESET}")
        for r in failed:
            print(f"    {RED}✗{RESET}  {r['id']}  score={r['score']:.0%}"
                  + (f"  [{r['error'][:60]}]" if r.get("error") else ""))

    print()
    sys.exit(0 if pass_rate >= args.threshold else 1)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
