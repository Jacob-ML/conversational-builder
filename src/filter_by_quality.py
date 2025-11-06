"""
This script filters conversations based on a quality score. It tries to rate
the accuracy of the "Leichte Sprache" responses of the conversations.
"""

import json
import re
import argparse

SCORE_THRESHOLD = 8.5

LineType = dict[str, list[dict[str, str]]]


class Rules:
    MAX_WORDS_PER_SENTENCE = 15
    PENALTY_PER_EXTRA_WORD = 0.2

    MAX_CHARS_PER_WORD = 15
    PENALTY_PER_EXTRA_CHAR = 0.1

    LONG_WORD_SPLIT_BONUS = 0.5

    COMMA_PENALTY = 0.2
    COMPLEX_WORD_PENALTY = 0.1


def load_complex_words(path: str) -> set[str]:
    """
    Loads a set of complex words from a given file.
    """

    complex_words = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            complex_words.add(line.strip().lower())

    return complex_words


def count_syllables(word: str) -> int:
    """
    Counts the number of syllables in a given word.
    """

    word = word.lower()

    word = re.sub(r"ie|au|ei|eu|äu", "V", word)
    word = re.sub(r"[aeiouyäöü]", "V", word)
    word = re.sub(r"V+", "V", word)

    count = word.count("V")

    if word.endswith(("e", "es", "en", "em", "er")):
        count += 1

    return max(1, count)


def rate_conversation(
    conversation: LineType,
    complex_words: set[str],
) -> float:
    """
    Rates the quality of a conversation based on predefined criteria.
    """

    combined_assistant_message = "\n\n".join(
        message["content"]
        for message in conversation["messages"]
        if message["role"] == "assistant"
    )

    return rate_text(combined_assistant_message, complex_words)


def rate_text(
    text: str,
    complex_words: set[str],
) -> float:
    """
    Rates the quality of a given text based on predefined criteria.
    """

    score = 10.0

    sentences = re.split(r"[.!?:]+", text)
    total_words = 0
    total_syllables = 0

    for sentence in sentences:
        words = re.split(r"[\s\-\"]+", sentence.strip())

        if len(words) > Rules.MAX_WORDS_PER_SENTENCE:
            extra_words = len(words) - Rules.MAX_WORDS_PER_SENTENCE
            score -= extra_words * Rules.PENALTY_PER_EXTRA_WORD

        for word in words:
            if len(word) > Rules.MAX_CHARS_PER_WORD:
                extra_chars = len(word) - Rules.MAX_CHARS_PER_WORD
                score -= extra_chars * Rules.PENALTY_PER_EXTRA_CHAR
                score += Rules.LONG_WORD_SPLIT_BONUS

            if "," in word:
                score -= Rules.COMMA_PENALTY

            syllables = count_syllables(word)
            total_syllables += syllables
            total_words += 1

            if word.lower() in complex_words or syllables > 3:
                score -= Rules.COMPLEX_WORD_PENALTY

    avg_syllables_per_word = (
        total_syllables / total_words if total_words > 0 else 0
    )

    score += -((0.5 * avg_syllables_per_word - 1) ** 3)

    return score


def read_jsonl(
    file_path: str,
) -> "list[LineType]":
    """
    Reads a JSONL file and returns a list of conversations.
    """

    conversations = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            conversations.append(json.loads(line))

    return conversations


def main():
    """
    Read conversations from a given JSONL file, rate them, filter based on
    quality score, and write back the highest quality conversations to a new
    JSONL file.
    """

    parser = argparse.ArgumentParser(
        description="Filter conversations by quality score."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file containing conversations.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output JSONL file for filtered conversations.",
    )
    args = parser.parse_args()

    conversations = read_jsonl(args.input_file)
    filtered_conversations = []

    complex_words = load_complex_words("./src/resources/complex_words.txt")

    for conversation in conversations:
        score = rate_conversation(conversation, complex_words)

        if score >= SCORE_THRESHOLD:
            filtered_conversations.append(conversation)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for conversation in filtered_conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
