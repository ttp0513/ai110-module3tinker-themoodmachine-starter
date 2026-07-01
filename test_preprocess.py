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
