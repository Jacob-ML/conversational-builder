"""
Reads categories from a given JSONL conversational dataset file and returns
them as a summary.
"""

from argparse import ArgumentParser
import json


WORDS_PER_TOKEN = 3 / 4


def read_jsonl(file_path: str) -> list[dict]:
    """
    Reads a JSONL file and returns its content as a list of dictionaries.
    """

    data = []
    max_tokens_found = 0

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            words = line.count(" ") + 1
            tokens = int(words / WORDS_PER_TOKEN)

            if tokens > max_tokens_found:
                max_tokens_found = tokens

            data.append(json.loads(line))

    print(f"Max tokens found in a single line: {max_tokens_found}")

    return data


def extract_categories(data: list[dict]) -> dict[str, int]:
    """
    Extracts unique categories from the dataset.
    """

    categories = {}

    for entry in data:
        category = entry.get("category", "other")

        if category not in categories:
            categories[category] = 0

        categories[category] += 1

    categories = dict(
        sorted(categories.items(), key=lambda item: item[1], reverse=True)
    )

    return categories


def print_summary(categories: dict[str, int]) -> None:
    """
    Prints a summary of categories and their counts.
    """

    total_categories = sum(categories.values())
    unique_categories = len(categories)
    percentages = {
        category: (count / total_categories) * 100
        for category, count in categories.items()
    }

    print(
        f"""Total entries: {total_categories}
Unique categories: {unique_categories}
Category breakdown:"""
    )

    for category, count in categories.items():
        percentage = percentages[category]
        print(f" - {category}: {count} ({percentage:.2f}%)")


def main():
    """
    The main entry point of the script.
    """

    parser = ArgumentParser(
        description="Read categories from a JSONL dataset."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the JSONL dataset file.",
    )

    args = parser.parse_args()

    data = read_jsonl(args.file_path)
    categories = extract_categories(data)
    print_summary(categories)


if __name__ == "__main__":
    main()
