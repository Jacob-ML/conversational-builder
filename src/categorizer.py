"""
Categorizes conversations based on predefined topics. Queries an LLM for each
categorization task.
"""

import json
from os import getenv
from argparse import ArgumentParser

from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

TOPICS = [
    "maths",
    "science",
    "history",
    "literature",
    "art",
    "sports",
    "music",
    "travel",
    "baking and cooking",
    "geography",
    "translation",
    "casual conversation",
    "technology",
    "education",
    "finance",
    "movies",
    "politics",
    "health",
    "other",
]

PROMPT = """You are an expert query categorizer. You are given a query and \
shall categorize it. Allowed topics: {topics}. The query will be in German.""".format(
    topics=", ".join(TOPICS)
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "CategorySchema",
        "schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "The category of the query.",
                    "enum": TOPICS,
                },
            },
            "required": ["category"],
        },
    },
}


def categorize_conversation(conversation: dict, client: OpenAI) -> dict:
    """
    Takes in a conversation dictionary and returns it, adding a "category" key
    with the determined category as value.
    """

    user_message = ""

    for message in conversation["conversations"]:
        if message["role"] == "user":
            user_message = message["content"]
            break
    else:
        raise ValueError("No user message found in conversation.")

    category = "none"
    counter = 0

    while (category not in TOPICS) and (counter < 4):
        counter += 1
        response = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct",
            messages=[
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=RESPONSE_FORMAT,  # type: ignore
            reasoning_effort="low",
        )

        if response.choices[0].message.content is None:
            continue

        category = json.loads(response.choices[0].message.content)["category"]

    if category not in TOPICS:
        category = "other"

    conversation["category"] = category

    return conversation


def read_jsonl(filename: str) -> list[dict]:
    """
    Reads conversations from a JSONL file and returns them as a list of dicts.
    """

    conversations = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            conversations.append(json.loads(line))

    return conversations


def main():
    """
    Main function to demonstrate categorization. Writes the results to the
    output file on-the-fly during execution to avoid data loss in case of
    interruptions.
    """

    parser = ArgumentParser(description="Categorize conversations")
    parser.add_argument(
        "input_file",
        type=str,
        default="conversations.jsonl",
        help="Path to the input JSONL file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        default="conversations_categorized.jsonl",
        help="Path to the output JSONL file",
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=getenv("OPENAI_API_BASE_URL"),
        api_key=getenv("OPENAI_API_KEY"),
    )
    conversations = read_jsonl(args.input_file)

    for conversation in tqdm(conversations, desc="Categorizing"):
        try:
            categorized_conversation = categorize_conversation(
                conversation, client
            )
        except ValueError:
            print("Skipping conversation with no user message.")
            continue

        with open(args.output_file, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(categorized_conversation, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
