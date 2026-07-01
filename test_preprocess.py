# test_preprocess.py
"""
Tests for MoodAnalyzer.preprocess.

Run with:  pytest test_preprocess.py
"""

import pytest

from mood_analyzer import MoodAnalyzer


@pytest.fixture
def analyzer() -> MoodAnalyzer:
    # Word lists don't matter for preprocessing; use empty lists.
    return MoodAnalyzer([], [])


def test_lowercases_and_splits(analyzer):
    assert analyzer.preprocess("Hello World") == ["hello", "world"]


def test_strips_surrounding_whitespace(analyzer):
    assert analyzer.preprocess("   spaced   out   ") == ["spaced", "out"]


def test_removes_punctuation(analyzer):
    assert analyzer.preprocess("great, awesome! wow.") == ["great", "awesome", "wow"]


def test_normalizes_repeated_characters(analyzer):
    # 3+ repeats collapse to two; intentional doubles are preserved.
    assert analyzer.preprocess("soooo cool") == ["soo", "cool"]


def test_keeps_emoticons_as_tokens(analyzer):
    tokens = analyzer.preprocess("this is bad :-(")
    assert tokens == ["this", "is", "bad", ":-("]


def test_keeps_emoji_as_tokens(analyzer):
    tokens = analyzer.preprocess("so funny \U0001F602")
    assert "\U0001F602" in tokens
    assert tokens == ["so", "funny", "\U0001F602"]


def test_emoticon_survives_when_punctuation_removed(analyzer):
    # ":)" must be preserved even though ")" is punctuation.
    tokens = analyzer.preprocess("happy day :)")
    assert ":)" in tokens


def test_combined_features(analyzer):
    tokens = analyzer.preprocess("I am soooo happy!!! :)")
    assert tokens == ["i", "am", "soo", "happy", ":)"]


def test_empty_string_returns_empty_list(analyzer):
    assert analyzer.preprocess("") == []


def test_whitespace_only_returns_empty_list(analyzer):
    assert analyzer.preprocess("    \t\n  ") == []


# ---------------------------------------------------------------------
# score_text
# ---------------------------------------------------------------------

@pytest.fixture
def scorer() -> MoodAnalyzer:
    # Use the real default word lists so scoring reflects the shipped model.
    return MoodAnalyzer()


def test_positive_word_scores_up(scorer):
    assert scorer.score_text("I love this class so much") == 1


def test_negative_word_scores_down(scorer):
    assert scorer.score_text("Today was a terrible day") == -1


def test_neutral_text_scores_zero(scorer):
    assert scorer.score_text("It is Tuesday") == 0


def test_multiple_sentiment_words_accumulate(scorer):
    # "happy" (+1) and "awesome" (+1) both count.
    assert scorer.score_text("happy and awesome") == 2


def test_negation_flips_positive_to_negative(scorer):
    assert scorer.score_text("I am not happy about this") == -1


def test_negation_flips_negative_to_positive(scorer):
    assert scorer.score_text("not bad at all") == 1


def test_negation_reaches_across_filler(scorer):
    # "really" is filler and is skipped, so "not" still negates "happy".
    assert scorer.score_text("not really happy") == -1


def test_negation_uses_up_after_one_word(scorer):
    # "never" flips "fun" (-1); the later "great" stays positive (+1) -> 0.
    assert scorer.score_text("never fun but great") == 0


def test_stopwords_do_not_affect_score(scorer):
    assert scorer.score_text("the is a of to and but") == 0
