"""
Changes all tool call IDs in the dataset to Mistral-compatible IDs.
"""

from random import choices, choice
import string
import json
import argparse

import tqdm


def generate_mistral_id() -> str:
    """
    Rules for the ID: must be a-z, A-Z, 0-9, with a length of 9, and start with
    a letter.
    """

    return "".join(
        [choice(string.ascii_letters)]
        + choices(string.ascii_letters + string.digits, k=8)
    )


def convert_ids_in_conversation(conversation: list[dict]) -> list[dict]:
    """
    Converts all tool call IDs in the given conversation to Mistral-compatible IDs.
    """

    id_mapping = {}

    for message in conversation:
        if message["role"] in ["system", "user"]:
            continue

        if message["role"] == "assistant" and "tool_calls" in message:
            for tool_call in message["tool_calls"]:
                old_id = tool_call["id"]

                if old_id not in id_mapping:
                    id_mapping[old_id] = generate_mistral_id()

                tool_call["id"] = id_mapping[old_id]

        if message["role"] == "tool" and "tool_call_id" in message:
            old_id = message["tool_call_id"]

            if old_id not in id_mapping:
                id_mapping[old_id] = generate_mistral_id()

            message["tool_call_id"] = id_mapping[old_id]

    return conversation


def read_jsonl(file_path: str) -> list[dict]:
    """
    Reads a JSONL file and returns a list of dictionaries.
    """

    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def convert_line(entry: dict) -> dict:
    """
    Converts tool call IDs in a single dataset entry.
    """

    entry["messages"] = convert_ids_in_conversation(entry["messages"])

    return entry


def main():
    """
    Main entry point for the script.
    """

    parser = argparse.ArgumentParser(
        description="Convert tool call IDs in dataset to Mistral-compatible IDs."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSONL dataset file."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output JSONL dataset file."
    )
    args = parser.parse_args()

    data = read_jsonl(args.input_file)

    with open(args.output_file, "a", encoding="utf-8") as f:
        for entry in tqdm.tqdm(data):
            f.write(json.dumps(convert_line(entry)) + "\n")


if __name__ == "__main__":
    main()
