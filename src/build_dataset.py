"""
Builds the dataset by querying the OpenAI API repeatedly. It starts each
conversation with a predefined system prompt and a user question from the
initial dataset. It then generates follow-up questions based on the last
response.
"""

from openai import OpenAI
from dotenv import load_dotenv

import os
import random
import json
from datetime import datetime


load_dotenv()

INPUT_FOLDER = "./input_files"
OUTPUT_FOLDER = "./output_files"

SYSTEM_PROMPT_ANSWER = """Du bist Jacob, ein KI-Assistent in Leichter Sprache. Du sprichst ganz einfach. \
Du schreibst:
- kurze Sätze (5-10 Wörter)
- einfache Wörter (keine Fremdwörter)
- klare Struktur (Absätze, wichtige Wörter fettgedruckt)
- niemals komplexe Schachtelsätze/Einschübe
- zusammengesetzte Wörter trennst du mit Bindestrich (z.B. "Mathe-Aufgabe")

Wenn die Antwort nicht ganz klar ist oder mehr Kontext braucht, sag das.

Hilf dem Nutzer so gut wie möglich. Formatiere mit Markdown. Selbst, wenn die Anfrage kompliziert ist, antworte einfach. \
Nenne nur für den Nutzer relevante Infos. Begrüße nicht ständig den Nutzer und komme direkt zur Sache.
""".strip()

SYSTEM_PROMPT_FOLLOWUP = """Dir wird ein Gespräch zwischen einem Nutzer mit Lernschwierigkeiten und einer KI, die in Leichter Sprache spricht, gezeigt.
Denke dir eine realistische Folgefrage auf die letzte Antwort der KI aus, die der Nutzer stellen könnte.
Sie muss im Stil der vorherigen Fragen des Nutzers sein und gut zur Antwort passen.

Antworte nur mit dieser Folgefrage und NICHTS weiter, keine Präambel, keine Erklärung, nur die Frage.""".strip()


client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


def query_api(messages: list[dict], max_tokens: int = -1, retries=3) -> str:
    """
    Queries the OpenAI API with the given messages and returns the response
    content.
    """

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "google/gemini-2.0-flash-001"),
                messages=messages,  # type: ignore
                max_tokens=(
                    max_tokens
                    if max_tokens > 0
                    else int(os.getenv("MAX_TOKENS", 8000))
                ),
                temperature=float(os.getenv("TEMPERATURE", 0.7)),
                reasoning_effort="low",
            )

            break
        except Exception as e:
            print(f"Error querying API: {e}")
            print("Retrying...")
    else:
        raise RuntimeError("Failed to query API after multiple retries")

    if not response.choices[0].message.content:
        raise ValueError("No content in response")

    return response.choices[0].message.content


def build_conversation(starter: str) -> list[dict]:
    """
    Builds a conversation starting with the given starter question. Returns the
    conversation as a list of message dictionaries.
    """

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT_ANSWER},
        {"role": "user", "content": starter},
    ]

    while True:
        assistant_message = query_api(conversation)

        conversation.append(
            {"role": "assistant", "content": assistant_message}
        )

        if len(conversation) >= int(
            os.getenv("MAX_CONVERSATION_LENGTH", 16)
        ) or random.random() > float(
            os.getenv("ANOTHER_MESSAGE_PROBABILITY", 0.6)
        ):
            break

        followup_summary = "# Zusammenfassung der vorherigen Nachrichten\n\n"

        for message in conversation:
            if message["role"] == "user":
                followup_summary += (
                    f"- Nutzer: \"\"\"{message['content']}\"\"\"\n\n"
                )
            else:
                followup_summary += (
                    f"- KI: \"\"\"{message['content']}\"\"\"\n\n"
                )

        followup_convo = [
            {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
            {"role": "user", "content": followup_summary},
        ]
        followup_question = query_api(followup_convo, max_tokens=128)

        conversation.append({"role": "user", "content": followup_question})

    return conversation[1:]


def read_conversation_starters() -> list[str]:
    """
    Reads the conversation starters from the dataset file and returns them as
    a list of strings.
    """

    files = filter(lambda f: f.endswith(".txt"), os.listdir(INPUT_FOLDER))
    conversation_starters = []

    for file in files:
        full_path = os.path.join(INPUT_FOLDER, file)
        if not os.path.isfile(full_path) or ("__" in file):
            continue

        print("Reading conversation starters from", full_path)

        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.read().split("===*===")
            conversation_starters.extend(
                [line.strip() for line in lines if line.strip()]
            )

    return conversation_starters


def write_conversation(conversation: list[dict], filename: str):
    """
    Writes the given conversation to a file in the output folder.
    """

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(conversation, ensure_ascii=False) + "\n")


def create_output_file(output_filename):
    """
    Creates the output file if it doesn't exist or is empty.
    """

    with open(output_filename, "w", encoding="utf-8") as f:
        if (
            not os.path.isfile(output_filename)
            or os.path.getsize(output_filename) == 0
        ):
            f.write("")


def main():
    """
    The main entry point of the script. Builds the dataset by querying the
    OpenAI API repeatedly.
    """

    starters = read_conversation_starters()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Found", len(starters), "conversation starters.")

    output_filename = os.path.join(
        OUTPUT_FOLDER,
        f"convs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    create_output_file(output_filename)

    for idx, starter in enumerate(starters):
        print(f"Building conversation {idx + 1}/{len(starters)}...")

        conversation = build_conversation(starter)
        write_conversation(conversation, output_filename)
        print("\tWrote conversation with", len(conversation), "messages.")

    print("Finished.")


if __name__ == "__main__":
    main()
