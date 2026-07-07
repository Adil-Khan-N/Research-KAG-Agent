"""
Run the full RAGAS evaluation.
Estimated time: 45-90 minutes (rate limiting on free Gemini tier).

Run: python scratch/run_ragas_eval.py

Options:
  --quick    Run only 9 questions (3 per type) for a fast test
  --full     Run all 40 questions
"""

import sys
import logging
logging.basicConfig(level=logging.WARNING)

quick = "--quick" in sys.argv
full = "--full" in sys.argv

subset_size = 9 if quick else (40 if full else 15)

print(f"Running RAGAS evaluation with {subset_size} questions")
print(f"Mode: {'quick' if quick else 'full' if full else 'standard'}")
print(f"Estimated time: {subset_size * 3 * 15 / 60:.0f}-"
      f"{subset_size * 3 * 20 / 60:.0f} minutes\n")

from eval.ragas_eval import run_full_evaluation

results = run_full_evaluation(
    subset_size=subset_size,
    delay_between_calls=15.0,  # 15s = safe for 5 req/min free tier
    save_samples=True,
)

print("\nDone. Check docs/ragas_results.md for the full table.")