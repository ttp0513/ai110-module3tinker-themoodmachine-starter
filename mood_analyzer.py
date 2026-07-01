# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re

from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        # Common text emoticons we want to keep as their own tokens.
        # (Order matters: match longer patterns like ":-)" before ":)".)
        EMOTICON_PATTERN = r":-?\)|:-?\(|:-?D|:-?P|;-?\)|<3|:'\(|:\||:/"

        cleaned = text.strip().lower()

        # 1. Pull out emoticons and emoji BEFORE stripping punctuation, so
        #    signals like ":)" aren't destroyed when we remove punctuation.
        symbol_tokens: List[str] = re.findall(EMOTICON_PATTERN, cleaned)
        symbol_tokens += re.findall(r"[\U0001F000-\U0001FAFF☀-➿]", cleaned)

        # 2. Remove the emoticons/emoji from the text now that they're saved,
        #    then drop any remaining punctuation (keep letters, digits, spaces).
        cleaned = re.sub(EMOTICON_PATTERN, " ", cleaned)
        cleaned = re.sub(r"[^\w\s]", " ", cleaned, flags=re.UNICODE)

        # 3. Normalize runs of 3+ identical characters ("soooo" -> "soo"),
        #    keeping two so intentional doubles ("cool") survive.
        cleaned = re.sub(r"(.)\1{2,}", r"\1\1", cleaned)

        # 4. Split the leftover text into word tokens and combine with symbols.
        word_tokens = cleaned.split()
        tokens = word_tokens + symbol_tokens

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        # Words that invert the sentiment of the next meaningful word:
        #   "not happy" -> negative, "not bad" -> positive.
        NEGATION_WORDS = {"not", "no", "never", "n't", "cannot", "cant", "without"}

        # Meaningless filler that carries no sentiment. Skipping these keeps the
        # analysis efficient and lets negation reach the word it modifies even
        # when filler sits in between ("not really happy" -> negate "happy").
        # NOTE: never put negation or sentiment words in here.
        STOPWORDS = {
            "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
            "this", "that", "these", "those", "it", "its", "i", "you", "he",
            "she", "we", "they", "of", "to", "in", "on", "for", "and", "but",
            "so", "very", "really", "just", "kind", "about", "my", "your",
        }

        tokens = self.preprocess(text)

        score = 0
        negate_next = False  # does the next meaningful word get its sign flipped?

        for token in tokens:
            if token in STOPWORDS:
                continue  # ignore filler; leaves negation armed for the real word

            if token in NEGATION_WORDS:
                negate_next = True
                continue

            # +1 for positive, -1 for negative, 0 otherwise.
            value = 0
            if token in self.positive_words:
                value = 1
            elif token in self.negative_words:
                value = -1

            if value != 0 and negate_next:
                value = -value

            score += value
            negate_next = False  # window is one meaningful word wide

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        # TODO: Implement this method.
        #   1. Call self.score_text(text) to get the numeric score.
        #   2. Return "positive" if the score is above 0.
        #   3. Return "negative" if the score is below 0.
        #   4. Return "neutral" otherwise.
        pass

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
