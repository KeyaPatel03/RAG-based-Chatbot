import re


_UNIVERSITY_OF_RE = re.compile(
    r"\bUniversity of\s+[A-Za-z][\w&'.-]*(?:\s+[A-Za-z][\w&'.-]*)*"
    r"(?=\s+(?:have|has|need|needs|want|wants|require|requires|required|"
    r"admission|admissions|deadline|deadlines|financial|scholarship|scholarships|"
    r"apply|application|applications|campus|major|majors|tuition|essay|essays|"
    r"transfer|transfers|open|closed|offer|offers|look|looks|start|starts|"
    r"what|who|when|where|why|how|is|are|was|were|do|does|did)\b|[?!.]\Z|\Z)",
    re.IGNORECASE,
)

_NAMED_INSTITUTION_LOWER_RE = re.compile(
    r"\b[a-z][\w&'.-]*(?:\s+[a-z][\w&'.-]*)*\s+"
    r"(?:university|college|institute|school|community college|state university)\b"
    r"(?=\s+(?:have|has|need|needs|want|wants|require|requires|required|"
    r"admission|admissions|deadline|deadlines|financial|scholarship|scholarships|"
    r"apply|application|applications|campus|major|majors|tuition|essay|essays|"
    r"transfer|transfers|open|closed|offer|offers|look|looks|start|starts|"
    r"what|who|when|where|why|how|is|are|was|were|do|does|did)\b|[?!.]\Z|\Z)",
)

_NAMED_INSTITUTION_RE = re.compile(
    r"\b(?:[A-Z]{2,8}|[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*)*)\s+"
    r"(?:University|College|Institute|School|Community College|State University)\b"
    r"(?=\s+(?:have|has|need|needs|want|wants|require|requires|required|"
    r"admission|admissions|deadline|deadlines|financial|scholarship|scholarships|"
    r"apply|application|applications|campus|major|majors|tuition|essay|essays|"
    r"transfer|transfers|open|closed|offer|offers|look|looks|start|starts|"
    r"what|who|when|where|why|how|is|are|was|were|do|does|did)\b|[?!.]\Z|\Z)",
)

_UC_RE = re.compile(r"\bUC\s+[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*)*", re.IGNORECASE)

_INSTITUTION_MARKERS = (
    "course",
    "courses",
    "class",
    "classes",
    "apply",
    "admission",
    "admissions",
    "acceptance",
    "requirements",
    "deadline",
    "deadlines",
    "financial aid",
    "scholarship",
    "scholarships",
    "campus",
    "transfer",
    "major",
    "majors",
    "tuition",
    "essay",
    "housing",
    "major",
    "majors",
    "tuition",
    "program",
    "programs",
)

_GENERIC_WORDS = {
    "university",
    "college",
    "institute",
    "school",
    "community",
    "state",
    "of",
    "the",
    "and",
    "for",
    "at",
    "campus",
}

_TRAILING_QUERY_WORDS = {
    "have",
    "has",
    "need",
    "needs",
    "want",
    "wants",
    "require",
    "requires",
    "required",
    "admission",
    "admissions",
    "deadline",
    "deadlines",
    "financial",
    "scholarship",
    "scholarships",
    "apply",
    "application",
    "applications",
    "campus",
    "major",
    "majors",
    "tuition",
    "essay",
    "essays",
    "transfer",
    "transfers",
    "open",
    "closed",
    "offer",
    "offers",
    "look",
    "looks",
    "start",
    "starts",
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
}


def _normalize_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]


def _trim_trailing_query_words(phrase: str) -> str:
    words = phrase.split()
    while words and words[-1].lower() in _TRAILING_QUERY_WORDS:
        words.pop()
    return " ".join(words)


def _extract_institution_phrase(query: str) -> str:
    cleaned_query = re.sub(r"[,:;()\[\]{}]", " ", query)
    for pattern in (_UNIVERSITY_OF_RE, _UC_RE, _NAMED_INSTITUTION_RE, _NAMED_INSTITUTION_LOWER_RE):
        match = pattern.search(cleaned_query)
        if match:
            return _trim_trailing_query_words(match.group(0).strip())

    lowered = query.lower()
    if not any(marker in lowered for marker in _INSTITUTION_MARKERS):
        return ""

    acronym_after_preposition = re.search(r"\b(?:by|at|for|from|about|of)\s+(?P<acro>[A-Z]{2,8})\b", query)
    if acronym_after_preposition:
        return acronym_after_preposition.group("acro")

    acronym_match = re.search(r"\b[A-Z]{2,8}\b", query)
    if acronym_match:
        return acronym_match.group(0)

    return ""


def _institution_signatures(phrase: str) -> tuple[str, str, list[str]]:
    words = re.findall(r"[A-Za-z0-9]+", phrase)
    compact = re.sub(r"[^a-z0-9]+", "", phrase.lower())

    initials = ""
    if len(words) > 1:
        initials = "".join(
            word[0].lower()
            for word in words
            if word and word.lower() not in {"of", "the", "and", "for", "at"}
        )

    tokens = [
        token
        for token in _normalize_tokens(phrase)
        if token not in _GENERIC_WORDS and len(token) >= 3
    ]

    return compact, initials, tokens


def filter_sources_for_query(query: str, source_urls: list[str]) -> list[str]:
    """
    Keep only sources related to the specific university named in the query.

    When the query does not name a university or similar institution, the
    original source list is returned unchanged.
    """
    phrase = _extract_institution_phrase(query)
    if not phrase:
        return source_urls

    compact, initials, tokens = _institution_signatures(phrase)
    if not compact and not initials and not tokens:
        return source_urls

    filtered: list[str] = []
    for url in source_urls:
        candidate = url.lower().strip()
        candidate_compact = re.sub(r"[^a-z0-9]+", "", candidate)
        candidate_tokens = set(_normalize_tokens(candidate))

        if compact and compact in candidate_compact:
            filtered.append(url)
            continue

        if initials and initials in candidate_compact:
            filtered.append(url)
            continue

        if tokens and any(token in candidate_tokens or token in candidate_compact for token in tokens):
            filtered.append(url)

    return filtered


def should_omit_sources_section(query: str, source_urls: list[str]) -> bool:
    """Return True when a university-specific query has no matching sources."""
    return bool(_extract_institution_phrase(query)) and not filter_sources_for_query(query, source_urls)