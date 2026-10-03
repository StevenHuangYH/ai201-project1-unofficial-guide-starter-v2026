"""
Scorer for Unit 2 evaluations.

Used by run_eval.py to judge whether an answer meets expectations.
"""


def judge(question: str, expects: str, answer: str, results: list) -> bool:
    """
    Judge whether the generated answer contains the expected text or phrase.
    """
    if not expects or not answer:
        return False
    return expects.strip().lower() in answer.lower()
