"""
This script filters conversations based on a quality score. It tries to rate
the accuracy of the "Leichte Sprache" responses of the conversations.
"""

import json
import re
import argparse

SCORE_THRESHOLD = 8.5


class Rules:
    MAX_WORDS_PER_SENTENCE = 20
    PENALTY_PER_EXTRA_WORD = 0.2

    MAX_CHARS_PER_WORD = 15
    PENALTY_PER_EXTRA_CHAR = 0.1

    LONG_WORD_SPLIT_BONUS = 0.5

    COMMA_PENALTY = 0.2
    COMPLEX_WORD_PENALTY = 0.1


def rate_conversation(
    conversation: "dict[str, list[dict[str, str]]]",
) -> float:
    """
    Rates the quality of a conversation based on predefined criteria.
    """

    combined_assistant_message = "\n\n".join(
        message["content"]
        for message in conversation["messages"]
        if message["role"] == "assistant"
    )

    score = 10.0

    sentences = re.split(r"[.!?:]+", combined_assistant_message)

    for sentence in sentences:
        words = re.split(r"[\s-\"]+", sentence.strip())

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

            if len(word) > 10:
                score -= Rules.COMPLEX_WORD_PENALTY

    return score


def read_jsonl(file_path: str) -> "list[list[dict[str, str]]]":
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

    for conversation in conversations:
        score = rate_conversation(conversation)

        if score >= SCORE_THRESHOLD:
            filtered_conversations.append(conversation)

    with open(args.output_file, "w", encoding="utf-8") as f:
        for conversation in filtered_conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
