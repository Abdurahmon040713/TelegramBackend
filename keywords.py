"""
Negative keyword loading and pattern matching — optimised for Uzbek-language text.

Design goals
────────────
• Treat Uzbek digraphs (o', g', sh, ch, ng …) as SINGLE units when building
  regex patterns — the old char-by-char approach split "o'" into separate "o"
  and "'" tokens, so apostrophe-normalized input was never matched.

• Accept any Unicode apostrophe variant in the INPUT (', ', ʻ, ʼ, `, ′ …).
  All variants are canonicalised to ASCII "'" before matching.

• Transliterate Cyrillic-script Uzbek to Latin BEFORE matching.  This lets
  a single Latin keyword list catch input typed in either script.
  Example: "сўкин" → "so'kin" → matched by keyword "so'kinish" pattern.

• Support leet-speak substitutions (0→o, 3→e, 5→s …) and mixed-Cyrillic
  injections (е→e, о→o, с→s …) via an extended LEET_MAP.

Matching flow
─────────────
  1. Input text  → _normalize_for_matching() → Latin, canonical apostrophe, lower
  2. Each keyword → _build_pattern()         → regex with flexible char classes
  3. is_toxic_by_keywords(text) checks the normalised text against all patterns.
"""
import logging
import os
import re

logger = logging.getLogger(__name__)

# Absolute path to the word file — works regardless of the working directory
# from which uvicorn/gunicorn is launched.
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WORD_FILE = os.path.join(_HERE, "data", "uz_negative_words.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Uzbek Cyrillic → Latin transliteration table
# ─────────────────────────────────────────────────────────────────────────────
# Covers the full Uzbek Cyrillic alphabet (pre-1995 reform) plus modern
# Uzbek-specific letters.  Multi-character Latin outputs (ш→sh) are handled
# by joining the list after per-character lookup.
_CYR_TO_LAT: dict[str, str] = {
    "а": "a",  "б": "b",  "в": "v",  "г": "g",  "д": "d",
    "е": "e",  "ё": "yo", "ж": "j",  "з": "z",  "и": "i",
    "й": "y",  "к": "k",  "л": "l",  "м": "m",  "н": "n",
    "о": "o",  "п": "p",  "р": "r",  "с": "s",  "т": "t",
    "у": "u",  "ф": "f",  "х": "x",  "ц": "ts", "ч": "ch",
    "ш": "sh", "ъ": "",   "ь": "",   "э": "e",  "ю": "yu",
    "я": "ya",
    # Uzbek-specific Cyrillic letters:
    "ў": "o'", "қ": "q",  "ғ": "g'", "ҳ": "h",  "ң": "ng",
}


def _transliterate_cyr(text: str) -> str:
    """Convert any Cyrillic Uzbek letters in *text* to Latin; pass others through."""
    parts: list[str] = []
    for ch in text:
        lat = _CYR_TO_LAT.get(ch.lower())
        if lat is not None:
            # Preserve original case for uppercase letters where it matters
            parts.append(lat.upper() if ch.isupper() and lat else lat)
        else:
            parts.append(ch)
    return "".join(parts)


# All Unicode apostrophe / modifier-letter variants used for o' and g'
_APOS_RE = re.compile(r"[''ʻʼ`ʹ′ʼʹ]")


def _normalize_for_matching(text: str) -> str:
    """
    Prepare input text for pattern matching:
      1. Cyrillic → Latin transliteration (catches Cyrillic-script Uzbek).
      2. Normalise all apostrophe variants to ASCII "'".
      3. Lowercase.
    """
    text = _transliterate_cyr(text)
    text = _APOS_RE.sub("'", text)
    return text.lower()


# Public alias used by context_scorer.py and any other module that needs
# to normalise text through the same Uzbek pipeline as Layer 1.
def normalize_for_matching(text: str) -> str:
    """Normalise *text* for pattern matching (Cyrillic→Latin, apostrophe, lowercase)."""
    return _normalize_for_matching(text)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Uzbek digraph / special-character patterns
# ─────────────────────────────────────────────────────────────────────────────
# These are 1- or 2-character KEYWORD sequences that must be treated as a
# single unit so the pattern builder does not split them.

# Two-character digraphs whose regex covers Cyrillic equivalents.
# Note: after _normalize_for_matching the INPUT is already in Latin, so
# Cyrillic equivalents here only help when the KEYWORD FILE itself uses Cyrillic.
_DIGRAPHS: dict[str, str] = {
    "sh": r"(?:sh|ş)",      # ш already transliterated → sh before matching
    "ch": r"(?:ch|č)",      # ч → ch
    "ng": r"(?:ng|ŋ)",
    "ts": r"(?:ts|tz)",
    "gh": r"(?:gh|ğ)",
    "yo": r"(?:yo|io)",
    "yu": r"(?:yu|iu)",
    "ya": r"(?:ya|ia)",
    "zh": r"(?:zh|j)",
}

# Uzbek letters-with-apostrophe — apostrophe made optional so "sokin" also
# matches the keyword "so'kin".  Includes common leet-speak substitutions for
# the base vowel.
_SPECIALS: dict[str, str] = {
    "o'": r"(?:[o0][']?)",  # o' or o (apostrophe optional), leet-o = 0
    "g'": r"(?:[g9q][']?)", # g' or g (apostrophe optional), leet-g = 9/q
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Extended leet-speak map (Latin + mixed-Cyrillic lookalikes)
# ─────────────────────────────────────────────────────────────────────────────
# Cyrillic lookalikes (а≈a, о≈o, с≈s …) are included so that "sokиn" (Latin
# with one embedded Cyrillic) is still caught.  Pure-Cyrillic input is handled
# upstream by _normalize_for_matching() before these patterns are applied.
LEET_MAP: dict[str, str] = {
    "a": r"[a4@а]",    "b": r"[b8б]",     "c": r"[c(\[с]",
    "d": r"[dд]",      "e": r"[e3€е]",    "f": r"[fф]",
    "g": r"[g69q]",    "h": r"[h#]",      "i": r"[i1!|и]",
    "j": r"[jй]",      "k": r"[kк]",      "l": r"[l1!|л]",
    "m": r"[mм]",      "n": r"[nн]",      "o": r"[o0о]",
    "p": r"[pр]",      "q": r"[q9kк]",    "r": r"[rр]",
    "s": r"[s5$с]",    "t": r"[t7+т]",    "u": r"[uüу]",
    "v": r"[vвb]",     "w": r"[w]",       "x": r"[x×*х]",
    "y": r"[yу]",      "z": r"[z2з]",
}

# Allows spaces / punctuation BETWEEN letters (leet-speak spacing evasion).
# Apostrophe is intentionally NOT excluded — it is consumed greedily by the
# _SPECIALS pattern for o' / g', so it won't block separator matching.
_SEP = r"[^\w]*"

# O'zbek tilidagi so'z qo'shimchalari (agglyutinativ til).
# "ahmoq" → "ahmoqsan", "ahmoqlarning", "ahmoqchilik" kabi shakllarni topish uchun.
# Pattern: so'z oxiriga ixtiyoriy qo'shimcha qo'shiladi.
# 10 belgi: "-larningday" (11 — chetki holat), "-chiligidan" (10), "-chasiga" (7)
# Eng uzun real qo'shimcha zanjirlari: -lar+ning+day = 10 belgi
_UZ_SUFFIX = r"\w{0,10}"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Tokeniser + pattern builder
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(keyword: str) -> list[str]:
    """
    Split *keyword* (already apostrophe-normalised and lowercased) into tokens:
      • "o'" and "g'"          → single 2-char token  (_SPECIALS)
      • Known digraphs (sh …) → single 2-char token  (_DIGRAPHS)
      • Everything else        → single character
    Handling these as units prevents the builder from splitting "o'" into
    separate "o" + "'" or "sh" into "s" + "h".
    """
    tokens: list[str] = []
    i = 0
    while i < len(keyword):
        pair = keyword[i : i + 2]
        if pair in _SPECIALS or pair in _DIGRAPHS:
            tokens.append(pair)
            i += 2
        else:
            tokens.append(keyword[i])
            i += 1
    return tokens


def _token_to_regex(token: str) -> str:
    if token in _SPECIALS:
        return _SPECIALS[token]
    if token in _DIGRAPHS:
        return _DIGRAPHS[token]
    if token in LEET_MAP:
        return LEET_MAP[token]
    return re.escape(token)


def _build_pattern(keyword: str) -> re.Pattern[str]:
    """
    Compile a regex for *keyword* that tolerates:
      • Apostrophe omission in o'/g'
      • Leet-speak substitutions
      • Spaces/punctuation between characters
      • Cyrillic keywords (transliterated to Latin first)
    """
    # Normalise the KEYWORD the same way we normalise the INPUT:
    # transliterate any Cyrillic, canonicalise apostrophes, lowercase.
    normalised = _transliterate_cyr(keyword.strip().lower())
    normalised = _APOS_RE.sub("'", normalised)

    # Very short words: exact boundary match only (avoids false positives)
    if len(normalised) <= 2:
        return re.compile(
            rf"(?<!\w){re.escape(normalised)}(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )

    tokens = _tokenize(normalised)
    parts  = [_token_to_regex(t) for t in tokens]
    # O'zbek qo'shimchalari — ildiz uzunligiga qarab suffix limit:
    #   1-3 harf (am, it, lox)   → suffix YO'Q (false positive xavfi yuqori)
    #   4-5 harf (ahmoq, jinni)  → \w{0,10} (to'liq qo'shimcha qo'llaniladi)
    #   6+ harf  (hayvon, fohisha)→ \w{0,10}
    # Tokenlar soni = haqiqiy harf/digraph soni (o' = 1 token)
    token_count = len(tokens)
    if token_count <= 3:
        # Qisqa so'zlar: aniq chegara, suffix yo'q
        return re.compile(
            rf"(?<!\w){_SEP.join(parts)}(?!\w)",
            re.IGNORECASE | re.UNICODE,
        )
    # 4+ token: O'zbek qo'shimchalarini qo'llab-quvvatlash
    return re.compile(
        rf"(?<!\w){_SEP.join(parts)}{_UZ_SUFFIX}(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Public interface
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_PATTERNS: list[re.Pattern[str]] = []


def _load_words(file_path: str | None = None) -> list[str]:
    if file_path is None:
        file_path = _DEFAULT_WORD_FILE
    try:
        if not os.path.exists(file_path):
            logger.warning("Negativ so'zlar fayli topilmadi: %s", file_path)
            return []
        with open(file_path, "r", encoding="utf-8") as fh:
            # Skip comment lines (starting with ---) and blank lines
            words = [
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("---")
            ]
        logger.info("Loaded %d negative keywords from %s", len(words), file_path)
        return words
    except Exception:
        logger.exception("Negative word file could not be read: %s", file_path)
        return []


def initialize_patterns(file_path: str | None = None) -> None:
    """Build KEYWORD_PATTERNS from the word list.  Call once at startup."""
    global KEYWORD_PATTERNS
    words = _load_words(file_path)
    KEYWORD_PATTERNS = [_build_pattern(w) for w in words]
    logger.info("%d keyword patterns compiled (keywords.py)", len(KEYWORD_PATTERNS))


_EMPTY_WARNING_LOGGED = False


def is_toxic_by_keywords(text: str) -> bool:
    """
    Return True if *text* contains any negative keyword.

    The text is normalised before matching (Cyrillic→Latin, apostrophe
    variants→ASCII, lowercased) so the same keyword list works for both
    Latin-script and Cyrillic-script Uzbek input.

    Agar KEYWORD_PATTERNS bo'sh bo'lsa — avtomatik initialize qilishga
    urinadi (run_bot.py orqali ishga tushganda himoya).
    """
    global _EMPTY_WARNING_LOGGED
    if not KEYWORD_PATTERNS:
        if not _EMPTY_WARNING_LOGGED:
            logger.warning(
                "KEYWORD_PATTERNS bo'sh — initialize_patterns() chaqirilmagan. "
                "Avtomatik yuklash urinilmoqda..."
            )
            _EMPTY_WARNING_LOGGED = True
        initialize_patterns()
        if not KEYWORD_PATTERNS:
            return False
    normalised = _normalize_for_matching(text)
    return any(p.search(normalised) for p in KEYWORD_PATTERNS)
