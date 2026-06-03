"""
Layer 2: Contextual Weight Scoring — Hybrid Moderation Pipeline.

Sits between the fast Regex/keyword pass (Layer 1) and the expensive AI
inference (Layer 3).  Returns a numeric toxicity SCORE (0.0 – 1.0) based on:

  A. Direct threat signals   — explicit violence vocabulary
  B. Scam combination rules  — money + action + credential patterns together
  C. Spam combination rules  — link + CTA or prize patterns together

Design principles
─────────────────
• Every pattern operates on NORMALISED text (Uzbek Cyrillic → Latin,
  apostrophe variants → ASCII "'", lowercased) so the same word-list covers
  both script variants without duplication.
• Individual signals produce ADDITIVE partial scores capped at 1.0.
• Combinations (B, C) are ONLY activated when two or more co-occurring
  signals are present in the same message — prevents single-word false positives
  for words like "pul" (money) or "uraman" (hit, but also "I play <instrument>").
• The caller controls thresholds; this module is purely a scorer.

Usage:
    from context_scorer import score_text, ScoredResult

    result = score_text("karta raqamingni hoziroq yubor")
    # ScoredResult(score=0.85, signals=['scam:credential_request'], reason='context_weight')
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Reuse the Uzbek-aware normaliser from Layer 1 so transliteration and
# apostrophe canonicalisation are identical across all pipeline stages.
from keywords import normalize_for_matching

# ─────────────────────────────────────────────────────────────────────────────
# A. Threat patterns
# Each tuple: (compiled-regex, score-contribution)
# Patterns are applied to the NORMALISED (Latin, lowercase) text.
#
# IMPORTANT: o' va g' — apostrof ixtiyoriy qilingan (o'? / g'?)
# Sabab: foydalanuvchi "oldiraman" deb apostrofsiz yozishi mumkin,
# Kirill yozuvdan kelganda esa normalize "o'ldiraman" qiladi.
# Ikkala holatda ham topilishi kerak.
# ─────────────────────────────────────────────────────────────────────────────

# Helper: o' va g' apostrof ixtiyoriy
_OQ = r"o'?"   # o' yoki o
_GQ = r"g'?"   # g' yoki g

_THREAT_HIGH: list[tuple[re.Pattern[str], float]] = [
    # Unambiguous violence threats (score ≥ 0.70)
    (re.compile(rf"\bqonarsan\b"),                   0.80),  # you will bleed
    (re.compile(rf"\b{_OQ}ldiraman\b"),              0.78),  # I will kill
    (re.compile(rf"\burib\s+{_OQ}ldiraman\b"),       0.85),  # I will beat to death
    (re.compile(rf"\bjonini\s+olaman\b"),             0.82),  # I will take his/her life
    (re.compile(rf"\bjoningni\s+olaman\b"),           0.85),  # I will take your life
    (re.compile(rf"\bkesib\s+tashlaman\b"),           0.80),  # I will cut off
    (re.compile(rf"\bpichoqlayman\b"),                0.82),  # I will stab
    (re.compile(rf"\b{_OQ}tib\s+{_OQ}ldiraman\b"),  0.88),  # I will shoot dead
    (re.compile(rf"\bqatl\s+qilaman\b"),              0.85),  # I will execute
    (re.compile(rf"\b{_OQ}limga\s+mahkum\b"),        0.80),  # condemned to death
]

_THREAT_MED: list[tuple[re.Pattern[str], float]] = [
    # Ambiguous — only meaningful in combos or with high AI backup (score 0.35–0.65)
    (re.compile(r"\buraman\b"),                       0.40),  # I will hit
    (re.compile(r"\bkesaman\b"),                      0.45),  # I will cut
    (re.compile(r"\bosaman\b"),                       0.50),  # I will hang
    (re.compile(r"\bopiraman\b"),                     0.50),  # I will chop
    (re.compile(r"\bkuyasan\b"),                      0.38),  # you will suffer
    (re.compile(r"\bqocharsan\b"),                    0.30),  # you will run (mild)
    (re.compile(r"\bjoningni\b"),                     0.40),  # your life
    (re.compile(r"\bterakt\b"),                       0.78),  # terrorist act
    (re.compile(rf"\bq{_OQ}z{_GQ}olon\b"),          0.55),  # uprising/riot
    (re.compile(r"\bportlataman\b"),                  0.82),  # I will explode
    # English violence
    (re.compile(r"\bkill\b"),                         0.55),
    (re.compile(r"\bmurder\b"),                       0.60),
    (re.compile(r"\bstab\b"),                         0.55),
]

# Contextual intensifiers: raise other signals when present
_THREAT_INTENSIFIER: list[re.Pattern[str]] = [
    re.compile(r"\bseni\b"),    # you (direct address)
    re.compile(r"\bsenga\b"),   # to you
    re.compile(r"\buni\b"),     # him/her
    re.compile(r"\bunga\b"),    # to him/her
    re.compile(r"\bbarchangni\b"), # all of you
    re.compile(r"\bhammangni\b"),  # all of you
]


# ─────────────────────────────────────────────────────────────────────────────
# B. Scam signal groups  (individual words — ONLY scored in combinations)
# ─────────────────────────────────────────────────────────────────────────────

_SCAM_MONEY: list[re.Pattern[str]] = [
    re.compile(r"\bpul\b"),         # money
    re.compile(r"\bdollar\b"),
    re.compile(rf"\bs{_OQ}m\b"),    # so'm — Uzbek soum
    re.compile(r"\bmillion\b"),
    re.compile(r"\bkarta\b"),       # bank card
    re.compile(r"\bcard\b"),
    re.compile(r"\bhisob\b"),       # bank account
    re.compile(r"\bbalans\b"),      # balance
    re.compile(r"\bsumma\b"),       # amount
]

_SCAM_ACTION: list[re.Pattern[str]] = [
    re.compile(r"\byuboring?\b"),          # send (imperative)
    re.compile(r"\byubor\b"),              # send (short form)
    re.compile(rf"\b{_OQ}tkazing?\b"),    # o'tkazing — transfer
    re.compile(rf"\bt{_OQ}lang?\b"),      # to'lang — pay
    re.compile(r"\bberging?\b"),           # give (imperative)
    re.compile(r"\btortib\b"),             # withdraw
    re.compile(r"\byuklang?\b"),           # load/upload
    re.compile(rf"\bt{_OQ}ldiring?\b"),   # to'ldiring — top up
]

_SCAM_CREDENTIAL: list[re.Pattern[str]] = [
    re.compile(r"\braqam(?:ingni|ni|ini)?\b"),   # (your) number
    re.compile(r"\bkarta(?:ngni|ni|ini)?\b"),    # (your) card
    re.compile(r"\bparol(?:ingni|ni|ini)?\b"),   # (your) password
    re.compile(r"\bkod(?:ingni|ni|ini)?\b"),     # (your) code
    re.compile(r"\bhisob(?:ingni|ni)?\b"),       # (your) account
    re.compile(r"\bpin\b"),                       # PIN
    re.compile(r"\bcvv\b"),                       # CVV
    re.compile(r"\bpincode\b"),
    re.compile(r"\bpassword\b"),
]

_SCAM_URGENCY: list[re.Pattern[str]] = [
    re.compile(r"\bhoziroq\b"),     # right now
    re.compile(r"\btezroq\b"),      # quickly
    re.compile(r"\bdarrov\b"),      # immediately
    re.compile(r"\bshoshiling\b"),  # hurry
    re.compile(r"\bbugun\b"),       # today
    re.compile(r"\btez\b"),         # fast/quick
    re.compile(r"\bkechiktirma\b"), # don't delay
    re.compile(r"\burgent\b"),
]


# ─────────────────────────────────────────────────────────────────────────────
# C. Spam signal groups
# ─────────────────────────────────────────────────────────────────────────────

_SPAM_LINK: list[re.Pattern[str]] = [
    re.compile(r"https?://"),
    re.compile(r"t\.me/"),
    re.compile(r"telegram\.me/"),
    re.compile(r"bit\.ly/"),
    re.compile(r"tinyurl\.com/"),
    re.compile(r"\bwww\.\w"),
]

_SPAM_CTA: list[re.Pattern[str]] = [
    re.compile(r"\bobuna\b"),              # subscribe
    re.compile(r"\bkanalga\b"),            # to channel
    re.compile(r"\bguruhga\b"),            # to group
    re.compile(rf"\bq{_OQ}shiling\b"),    # qo'shiling — join
    re.compile(r"\bkiring\b"),             # enter/join
    re.compile(r"\bbosin?g?\b"),           # press/click
    re.compile(r"\bclick\b"),
    re.compile(r"\bfollow\b"),
    re.compile(r"\bsubscribe\b"),
]

_SPAM_PRIZE: list[re.Pattern[str]] = [
    re.compile(r"\byutib\b"),       # won
    re.compile(r"\byutdingiz\b"),   # you won
    re.compile(rf"\bsov{_GQ}a\b"),  # sovg'a — gift
    re.compile(r"\bbepul\b"),       # free
    re.compile(r"\btekin\b"),       # free (colloquial)
    re.compile(r"\bgratis\b"),
    re.compile(r"\bbonus\b"),
    re.compile(r"\bprize\b"),
    re.compile(r"\baward\b"),
    re.compile(r"\biphone\b"),      # common bait
    re.compile(r"\bsamsung\b"),     # common bait
    re.compile(r"\byutqazdi\b"),    # drew/won
    re.compile(r"\blotereya\b"),    # lottery
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredResult:
    score:   float = 0.0
    signals: list[str] = field(default_factory=list)
    reason:  str = "context_weight"   # DB reason field value

    def _cap(self) -> None:
        self.score = min(1.0, max(0.0, self.score))


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_text(text: str) -> ScoredResult:
    """
    Score *text* for toxicity using the contextual weight model.

    The text is normalised via the same Uzbek pipeline as Layer 1
    (Cyrillic → Latin, apostrophe canonicalisation, lowercase) so results
    are consistent across the hybrid pipeline.

    Returns a ScoredResult with:
      .score   — 0.0 (harmless) … 1.0 (certainly toxic)
      .signals — list of matched signal names (for logging / audit)
    """
    normalised = normalize_for_matching(text)
    result     = ScoredResult()
    add        = result.signals.append

    # ── A. Threat signals ─────────────────────────────────────────────────────

    # Check if there is a direct address (you / him) — boosts threat weight
    has_target = any(p.search(normalised) for p in _THREAT_INTENSIFIER)

    high_threat_score = 0.0
    for pattern, weight in _THREAT_HIGH:
        if pattern.search(normalised):
            boost = 1.15 if has_target else 1.0
            high_threat_score += weight * boost
            add(f"threat_high:{pattern.pattern}")

    med_threat_score = 0.0
    for pattern, weight in _THREAT_MED:
        if pattern.search(normalised):
            boost = 1.20 if has_target else 1.0   # ambiguous words only matter with target
            med_threat_score += weight * boost
            add(f"threat_med:{pattern.pattern}")

    result.score += high_threat_score
    # Medium threat alone is not decisive — only add if direct target or multiple signals
    if has_target or len([s for s in result.signals if "threat_med" in s]) >= 2:
        result.score += med_threat_score * 0.70
    result._cap()

    # ── B. Scam combination rules ──────────────────────────────────────────────

    has_money  = any(p.search(normalised) for p in _SCAM_MONEY)
    has_action = any(p.search(normalised) for p in _SCAM_ACTION)
    has_cred   = any(p.search(normalised) for p in _SCAM_CREDENTIAL)
    has_urgent = any(p.search(normalised) for p in _SCAM_URGENCY)

    if has_cred and has_urgent:
        # Credential + urgency — eng xavfli scam signali
        result.score += 0.75
        add("scam:credential+urgency")
    elif has_cred:
        # Asking for credentials is always a top scam signal
        result.score += 0.60
        add("scam:credential_request")
    elif has_money and has_action and has_urgent:
        result.score += 0.50
        add("scam:money+action+urgency")
    elif has_money and has_action:
        result.score += 0.35
        add("scam:money+action")
    elif has_money and has_urgent:
        result.score += 0.22
        add("scam:money+urgency")

    result._cap()

    # ── C. Spam combination rules ──────────────────────────────────────────────

    has_link  = any(p.search(normalised) for p in _SPAM_LINK)
    has_cta   = any(p.search(normalised) for p in _SPAM_CTA)
    has_prize = any(p.search(normalised) for p in _SPAM_PRIZE)

    if has_link and has_prize:
        result.score += 0.55
        add("spam:link+prize")
    elif has_link and has_cta:
        result.score += 0.45
        add("spam:link+cta")
    elif has_prize and has_cta:
        result.score += 0.40
        add("spam:prize+cta")
    elif has_link and (has_cta or has_prize):
        result.score += 0.35
        add("spam:link+signal")

    result._cap()

    # ── D. Yashirin tahdid va tahqir (uzun kontekstli xabarlar) ───────────────
    # Qo'pol so'z ishlatmasdan, lekin kontekst toksik bo'lgan xabarlar.
    # Faqat kombinatsiyalarda ishlaydi — yolg'iz signal toksik emas.

    _INSULT_INDIRECT: list[re.Pattern[str]] = [
        re.compile(r"\bhech\s+narsa\s+uddalay\b"),        # hech narsa uddalay olmaysan
        re.compile(r"\bhech\s+kim\s+senga\b"),             # hech kim senga ishonmaydi
        re.compile(r"\bsenga\s+o'?rin\s+yo'?q\b"),        # senga o'rin yo'q
        re.compile(r"\bboshqa\s+joy\s+qara\b"),            # boshqa joy qara
        re.compile(r"\bhech\s+narsaga\s+yaramaysan\b"),    # hech narsaga yaramaysan
        re.compile(r"\bsending?\s+kabi\b"),                # sening kabi (kamchilik)
        re.compile(r"\bseni\s+tarbiya\s+qilgan\b"),       # seni tarbiya qilgan (oilaga hujum)
        re.compile(r"\bseng?a\s+aql\s+bergan\b"),         # senga aql bergan odam
        re.compile(r"\bnodonsan\b"),                       # nodonsan
        re.compile(r"\baybing?\s+ko'?p\b"),               # aybing ko'p
    ]

    _THREAT_HIDDEN: list[re.Pattern[str]] = [
        re.compile(r"nima\s+bo'?lishini\s+o'?zing?\s+bilasan"),      # nima bo'lishini o'zing bilasan
        re.compile(r"ko'?chada\s+ko'?rsam"),                         # ko'chada ko'rsam
        re.compile(r"gaplashadigan\s+gapim\s+bor"),                  # gaplashadigan gapim bor
        re.compile(r"bilib\s+qo'?y"),                                # bilib qo'y
        re.compile(r"kuning\s+qoldi"),                               # kuning qoldi
        re.compile(r"pushaymon\s+bo'?lasan"),                        # pushaymon bo'lasan
        re.compile(r"yig'?lab\s+qolasan"),                           # yig'lab qolasan
        re.compile(r"ko'?rasan\s+hali"),                             # ko'rasan hali
        re.compile(r"qo'?lingdan\s+keladi"),                         # qo'lingdan keladi
    ]

    # Direct address (seni/senga/sening) kerak — aks holda false positive yuqori
    has_direct = any(p.search(normalised) for p in [
        re.compile(r"\bseni\b"), re.compile(r"\bsenga\b"),
        re.compile(r"\bsening\b"), re.compile(r"\bsen\b"),
        re.compile(r"\bo'?zing"),  # o'zing, o'zingni, ozingni
    ])

    indirect_count = sum(1 for p in _INSULT_INDIRECT if p.search(normalised))
    hidden_count   = sum(1 for p in _THREAT_HIDDEN if p.search(normalised))

    if has_direct and hidden_count >= 1:
        result.score += 0.55
        add("hidden_threat:direct_address+veiled_threat")
    elif has_direct and indirect_count >= 1:
        result.score += 0.45
        add("hidden_insult:direct_address+indirect_attack")
    elif indirect_count >= 2:
        # Ikki yoki undan ko'p yashirin haqorat signali — target bo'lmasa ham
        result.score += 0.40
        add("hidden_insult:multiple_indirect")

    result._cap()

    return result
