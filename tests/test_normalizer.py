from amanuensis.text.normalizer import apply_lexicon_casing, normalize

# SYNTHETIC fixtures. Real variants must come from docs/romanization.md (owner-written).
VARIANTS = {"hai": "hai", "he": "hai", "kya": "kya", "kyaa": "kya"}


def test_case_and_punctuation():
    assert normalize("Kya, kar raha hai?") == "kya kar raha hai"


def test_whitespace_collapsed():
    assert normalize("  a   b \n c ") == "a b c"


def test_variants_mapped():
    assert normalize("kyaa kar raha he", VARIANTS) == "kya kar raha hai"


def test_no_variants_leaves_words():
    assert normalize("kyaa") == "kyaa"


def test_empty():
    assert normalize("") == ""
    assert normalize("?!") == ""


def test_german_umlauts_kept():
    assert normalize("Schöne Grüße!") == "schöne grüße"


def test_idempotent():
    once = normalize("Kyaa, KAR raha he!", VARIANTS)
    assert normalize(once, VARIANTS) == once


def test_lexicon_casing():
    assert apply_lexicon_casing("i study at pccoe", ["PCCOE"]) == "i study at PCCOE"
