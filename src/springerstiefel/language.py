"""Prompt language detection (stopword-based, no dependencies).

Hey_ answers in German by default. When the user writes English, the proxy
appends a reply-in-English directive so answers (including code comments)
match the prompt language. German is the default (native behavior).
"""

import re

EN_STOPWORDS = frozenset(
    """
    the and is are was were you your with for from write writes writing
    create creates created file files please explain explain show tell use
    using used this that what when where which there their have has had can
    will would should could there here how why not but all any some more
    than then make made into out about calculator program code function
    """.split()
)

DE_STOPWORDS = frozenset(
    """
    der die das den dem eines einer einem einen und ist sind war waren du
    dein mit für von vom zum zur bitte erstelle erstelle erstellen schreibe
    zeige sage datei dateien rechner programm code funktion dies dieser
    diese dieses was wann welche welcher welches nicht aber alle mehr als
    dann mache gemacht
    """.split()
)

LANGUAGE_DIRECTIVE = (
    "[Language]\nReply in English, including code comments."
)


def detect_language(text: str) -> str:
    """Return "en" or "de" by stopword majority (ties and empties → "de")."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    if not words:
        return "de"
    english = sum(1 for word in words if word in EN_STOPWORDS)
    german = sum(1 for word in words if word in DE_STOPWORDS)
    return "en" if english > german else "de"
