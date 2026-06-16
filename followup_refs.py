import re


_FOLLOWUP_REF_RE = re.compile(
    r"""
    (?:                          # optional leading verb phrase
        (?:answer|tell\s+me(?:\s+about)?|explain|elaborate\s+on|what\s+about|give\s+me)\s+
    )?
    (?:
        [Qq](?P<qnum>[1-5])\b               # Q1 ... Q5
      | (?P<word>first|second|third|fourth|fifth)  # "the first one"
        \s+(?:question|one|follow[- ]?up)?
      | (?:question\s+)?(?P<num>[1-5])(?:st|nd|rd|th)?      # "question 1" / "1st question" / "2"
        \s+(?:question|one|follow[- ]?up)?
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

_WORD_TO_IDX = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def resolve_followup_query(user_input: str, memory: list) -> str:
    """
    If the user's input is a shorthand reference to a previous follow-up
    question (e.g. "Q1", "answer Q2", "question 1", "the second one"),
    resolve it to the actual question text stored in the most recent memory turn.

    Returns the original user_input unchanged when:
    - no follow-up reference is detected
    - the referenced question index is out of range
    - no follow-up questions are stored in memory
    """
    stripped = user_input.strip()
    m = _FOLLOWUP_REF_RE.search(stripped)
    fallback_question_num = None
    if not m:
        question_num_match = re.search(r"(?:^|\b)question\s+(?P<num>[1-5])\b", stripped, re.IGNORECASE)
        if question_num_match:
            fallback_question_num = question_num_match.group("num")
    if not m:
        if fallback_question_num is None:
            return user_input

    if fallback_question_num is not None:
        idx = int(fallback_question_num) - 1
    elif m.group("qnum"):
        idx = int(m.group("qnum")) - 1
    elif m.group("word"):
        idx = _WORD_TO_IDX.get(m.group("word").lower(), -1)
    elif m.group("num"):
        idx = int(m.group("num")) - 1
    else:
        return user_input

    if idx < 0:
        return user_input

    followups: list[str] = []
    for turn in reversed(memory):
        if isinstance(turn, dict) and turn.get("followups"):
            followups = turn["followups"]
            break

    if followups and 0 <= idx < len(followups):
        resolved = followups[idx]
        print(f"[memory] Follow-up reference '{stripped}' → '{resolved}'")
        return resolved

    return user_input