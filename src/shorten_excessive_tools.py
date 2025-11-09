"""
Shortens excessively long tool responses in the dataset, which are only caused
by the web search tool; the tool may return thousands of tokens, in which case
we truncate the response to a more reasonable length.
"""

import json
from argparse import ArgumentParser

from tools import POTENTIAL_TOOLS

POSSIBLE_WEB_SEARCH_TOOL_NAMES = POTENTIAL_TOOLS["web"]["names"]


def read_jsonl(file_path: str) -> list[dict]:
    """
    Reads a JSONL file and returns its content as a list of dictionaries.
    """

    data = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    return data


def shorten_excessive_tool_responses(
    data: list[dict], max_tool_response_tokens: int = 8192
) -> list[dict]:
    """
    Shortens excessively long tool responses in the dataset.
    """

    for entry in data:
        for message in entry.get("messages", []):
            if message.get("role") == "tool":
                response = message.get("content", "")
                words = response.count(" ") + 1
                tokens = int(words / (3 / 4))

                if tokens > max_tool_response_tokens:
                    json_resp = json.loads(response)
                    if not "results" in json_resp:
                        # not a web search tool response
                        continue

                    tool_response = json_resp["results"]

                    new_results = []

                    for result in tool_response:
                        result["content"] = (
                            result["content"][
                                : int(
                                    (
                                        max_tool_response_tokens
                                        / len(tool_response)
                                    )
                                    * (3 / 4)
                                )
                            ]
                            + "... [truncated]"
                        )
                        new_results.append(result)
                        print("TRUNCATED")

                    message["content"] = json.dumps({"results": new_results})

    return data


def write_jsonl(file_path: str, data: list[dict]) -> None:
    """
    Writes a list of dictionaries to a JSONL file.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")


def main():
    """
    The main function to execute the shortening process.
    """

    parser = ArgumentParser(
        description="Shorten excessive tool responses in a JSONL dataset."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to the input JSONL file."
    )
    parser.add_argument(
        "output_file", type=str, help="Path to the output JSONL file."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens allowed for tool responses.",
    )

    args = parser.parse_args()

    data = read_jsonl(args.input_file)
    shortened_data = shorten_excessive_tool_responses(data, args.max_tokens)
    write_jsonl(args.output_file, shortened_data)


if __name__ == "__main__":
    main()
