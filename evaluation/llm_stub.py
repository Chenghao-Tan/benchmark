"""
Optional LLM-based realism score (OpenAI API). Stub returns NaN until configured.
"""


def llm_realism_score_stub(
    factual_row: object,
    counterfactual_row: object,
    feature_names: list[str],
    api_key: str | None = None,
) -> float:
    """
    Placeholder for a 1–10 realism rating from an LLM.
    When api_key is None, returns NaN (no network call).
    """
    del factual_row, counterfactual_row, feature_names, api_key
    return float("nan")
