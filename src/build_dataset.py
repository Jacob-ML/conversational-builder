"""
Builds the dataset by querying the OpenAI API repeatedly. It starts each
conversation with a predefined system prompt and a user question from the
initial dataset. It then generates follow-up questions based on the last
response.
"""

from typing import Union
from uuid import uuid4
from openai import OpenAI
from dotenv import load_dotenv

import os
import random
import json
import argparse
from datetime import datetime

from tools import POTENTIAL_TOOLS, get_tool_response


load_dotenv()

INPUT_FOLDER = "./input_files"
OUTPUT_FOLDER = "./output_files"

TOOLS_AVAILABLE = os.getenv("ENABLE_TOOLS", "false").lower() == "true"

SYSTEM_PROMPT_ANSWER = """Du bist **Jacob**, ein KI-Assistent in **Leichter Sprache**. Du sprichst ganz einfach. Hier sind die Regeln, die Du befolgen sollst:

## 1. Wörter

- **Verwende einfache und bekannte Wörter:** Du sollst Fachwörter und Fremdwörter vermeiden. Erkläre schwere Wörter und kündige sie an.
    - **Beispiel:** Schlecht: genehmigen / Gut: erlauben
- **Halte die Wörter kurz:** Bei langen Wörtern trennst Du sie mit einem Bindestrich, damit man sie besser lesen kann.
    - **Beispiel:** Schlecht: Bundesgleichstellungsgesetz / Gut: Bundes-Gleichstellungs-Gesetz
- **Benutze Verben (Tu-Wörter) und aktive Wörter:** Vermeide Nominalstil (Haupt-Wörter) und Passiv.
    - **Beispiel:** Schlecht: Morgen wird der Heim-Beirat gewählt. / Gut: Morgen wählen wir den Heim-Beirat.
- **Vermeide Genitiv und Konjunktiv:** Das macht die Sprache einfacher und konkreter.
    - **Beispiel (Genitiv):** Schlecht: Das Haus des Lehrers. / Gut: Das Haus von dem Lehrer.
    - **Beispiel (Konjunktiv):** Schlecht: Morgen könnte es regnen. / Gut: Morgen regnet es vielleicht.
- **Vermeide Redewendungen und bildliche Sprache:** Viele verstehen diese wörtlich.
    - **Beispiel:** Raben-Eltern sind nicht die Eltern von Raben-Küken. Mit Raben-Eltern meint man: schlechte Eltern.
- **Benutze positive Sprache:** Wenn Du negative Sprache (Wörter wie "nicht") brauchst, schreibe das Wort **fett**.

---

## 2. Zahlen und Zeichen

- **Schreibe Zahlen allgemein bekannt:** Benutze arabische Zahlen (z.B. 9) und vermeide römische Zahlen (z.B. IX).
- **Vermeide hohe Zahlen und Prozent-Zahlen:** Nutze stattdessen Vergleiche oder ungenaue Angaben.
    - **Beispiel:** Schlecht: 14.795 Menschen / Gut: Viele Menschen
- **Schreibe Telefon-Nummern mit Leerzeichen:** Das macht sie besser lesbar.
    - **Beispiel:** Schlecht: Tel.: (05544) 332211 / Gut: Telefon: 0 55 44 33 22 11
- **Vermeide Sonder-Zeichen:** Wenn Du eines benutzen musst, erkläre es eindeutig.
    - **Beispiel:** Schlecht: „ “ Anführungs-Striche / Gut: Das Zeichen für Paragraf ist: §

---

## 3. Sätze

- **Benutze kurze Sätze:** Mache in jedem Satz nur **eine** Aussage.
    - **Beispiel:** Schlecht: Das Buch, das auf dem Tisch liegt, habe ich schon gelesen. / Gut: Auf dem Tisch liegt ein Buch. Ich habe das Buch schon gelesen.
- **Benutze einen einfachen Satzbau:** Wer macht etwas? Was macht die Person?
    - **Beispiel:** Schlecht: Die Rechnung bezahlt Frau Weber. / Gut: Frau Weber bezahlt die Rechnung.
- **Vermeide Neben-Sätze:** Benutze besser mehrere Haupt-Sätze. Trenne die Sätze.
    - **Beispiel (weil-Satz):** Schlecht: Die Suppe schmeckt schlecht, weil in der Suppe zu viel Salz ist. / Gut: In der Suppe ist zu viel Salz. Deshalb schmeckt die Suppe schlecht.
- **Verkürzte Sätze sind erlaubt:** Du darfst Sätze mit Wörtern wie **Oder, Und, Aber** beginnen.
    - **Beispiel (Aber):** Schlecht: Ali war müde vom Sport, aber auch glücklich. / Gut: Ali war müde vom Sport. Aber glücklich.

---

## 4. Texte

- **Spreche die Nutzer*innen persönlich an:** Nutze die Anrede **Du**, wenn nicht anders gefragt.
- **Schreibe alles zusammen, was zusammen gehört:** Vermeide Verweise auf andere Stellen oder Texte.
    - **Beispiel:** Schlecht: (siehe: Heft 3) / Gut: In Heft 3 steht mehr dazu.

---

## 5. Gestaltung und Bilder

- **Formatierung:** Nutze, wo sinnvoll, Markdown-Formatierungen wie **Listen, Fettschrift und Trennlinien**. Nutze Teilüberschriften nur, wenn dies zur besseren Strukturierung langer Antworten beiträgt.
- **Absatz und Zeilenumbruch:** Schreibe **jeden neuen Satz in eine neue Zeile**. Trenne keine Wörter am Ende einer Zeile und mache viele Absätze und Überschriften.
    - **Beispiel (Trennung):** Schlecht: Der letzte Urlaub auf Mallorca war ein Er lebnis. / Gut: Der letzte Urlaub auf Mallorca war ein Erlebnis.
- **Hervorhebungen:** Hebe Wichtiges **fett** hervor, aber vermeide NUR GROßE BUCHSTABEN und Kursiv.
- **Nutze keine Tabellen oder sonstige komplizierte Formatierungen.**

---

Wenn die Antwort **nicht ganz klar** ist oder mehr Kontext braucht, **sag das**.
**Nur**, wenn du aktuelle oder sehr spezifische Infos brauchst, nutze ein passendes Tool, um sie zu finden. Tue das **nicht ständig**, sondern nur, wenn es zwingend nötig ist. Denke dir in solchen Fällen **keine** Antwort aus.

Hilf dem Nutzer so gut wie möglich. Selbst, wenn die Anfrage kompliziert ist, **antworte einfach**.
Nenne nur für den Nutzer **relevante Infos**. Begrüße nicht unnötig oft den Nutzer. Komme direkt zur Sache.
""".strip().replace(
    "\n\n\n", "\n\n"
)

SYSTEM_PROMPT_FOLLOWUP = """Dir wird ein Gespräch zwischen einem Nutzer mit Lernschwierigkeiten und einer KI, die in Leichter Sprache spricht, gezeigt.
Denke dir eine realistische Folge-Anfrage auf die letzte Antwort der KI aus, die der Nutzer senden könnte.
Es kann (aber muss nicht) eine Frage sein - vielleicht auch eine Aufforderung, oder der Nutzer möchte mehr Informationen/eine Erklärung, eine Umformulierung, Übersetzung, etc...
Sie muss im Stil der vorherigen Fragen des Nutzers sein und gut zur Antwort passen. Sie darf auch mehrzeilig und länger sein.

Antworte **nur** mit dieser Folge-Anfrage und **nichts** weiter, keine Präambel, keine Erklärung, **nur** die Frage.""".strip()


client = OpenAI(
    base_url=os.getenv("OPENAI_API_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


def query_api(
    messages: list[dict],
    tools: list[dict],
    max_tokens: int = -1,
    temperature: float = -1,
    model: str = "",
    retries=3,
) -> tuple[str, list]:
    """
    Queries the OpenAI API with the given messages and returns the response
    content.
    """

    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model=(
                    os.getenv("MODEL_NAME", "mistralai/mistral-medium-3.1")
                    if not model
                    else model
                ),
                messages=messages,  # type: ignore
                max_tokens=(
                    max_tokens
                    if max_tokens > 0
                    else int(os.getenv("MAX_TOKENS", 8000))
                ),
                tools=tools if len(tools) > 0 else None,  # type: ignore
                tool_choice="auto" if len(tools) > 0 else "none",
                temperature=(
                    float(os.getenv("TEMPERATURE", 0.7))
                    if temperature < 0
                    else temperature
                ),
                reasoning_effort="low",
            )

            break
        except Exception as e:
            print(f"Error querying API: {e}")
            print("Retrying...")
    else:
        raise RuntimeError("Failed to query API after multiple retries")

    content = response.choices[0].message.content
    tool_calls = response.choices[0].message.tool_calls

    if (not content) and (len(tool_calls) == 0):  # type: ignore
        raise ValueError("No content in response")

    return (
        content if content else "",
        tool_calls if tool_calls else [],
    )


def build_tools_list(forced_types: list = []) -> list[dict]:
    """
    Builds the list of tools to be used in the conversation randomly for good
    coverage of different tools.
    """

    tools = []

    for _tool_type, tool in POTENTIAL_TOOLS.items():
        if random.random() < 0.75 or _tool_type in forced_types:
            tool_name = random.choice(tool["names"])
            tool_description = random.choice(tool["descriptions"])
            tool_parameters = (
                random.choice(tool["parameters"]) if tool["parameters"] else {}
            )

            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_description,
                        "parameters": {
                            "type": "object",
                            "properties": tool_parameters.get(
                                "properties", {}
                            ),
                            "required": tool_parameters.get("required", []),
                        },
                    },
                }
            )

    return tools


def build_conversation(
    starter: str, original_file: str, system_prompt: str = ""
) -> dict[str, Union[str, list[dict]]]:
    """
    Builds a conversation starting with the given starter question. Returns the
    conversation as a list of message dictionaries.
    """

    tools = []

    if TOOLS_AVAILABLE:
        tools = build_tools_list(
            forced_types=[
                (
                    "weather"
                    if any(
                        x in starter.casefold()
                        for x in ["wetter", "regen", "sonne", "temperatur"]
                    )
                    else ""
                )
            ]
        )

    tool_info = (
        "Nur, wenn du aktuelle oder sehr spezifische Infos brauchst, nutze ein passendes Tool, um sie zu finden. Tue das nicht ständig, sondern nur, wenn es echt nötig ist. Denke dir keine Antwort aus."
        if len(tools) > 0
        else "Du hast keinen Zugriff auf das Internet und die Nachrichten. Wenn der Nutzer das glauben sollte, erkläre ihm das."
    )

    conversation: list[dict[str, Union[str, list[dict]]]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_ANSWER.format(tool_info=tool_info),
        },
        {"role": "user", "content": starter},
    ]

    while True:
        tool_calls = []
        first = True

        while first or (tool_calls and len(tool_calls) > 0):
            first = False
            assistant_message, tool_calls = query_api(
                conversation, tools if len(conversation) < 14 else []
            )

            conversation.append(
                {
                    "role": "assistant",
                    "content": assistant_message,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(
                        tool_call.function.arguments or "{}"
                    )

                    tool_response = get_tool_response(tool_name, tool_args)

                    conversation.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(tool_response),
                        }
                    )

        if len(conversation) >= int(
            os.getenv("MAX_CONVERSATION_LENGTH", 32)
        ) or random.random() > float(
            os.getenv("ANOTHER_MESSAGE_PROBABILITY", 0.65)
        ):
            break

        followup_summary = "# Zusammenfassung der vorherigen Nachrichten\n\n"

        for message in conversation:
            if message["role"] == "user":
                followup_summary += (
                    f"- Nutzer: \"\"\"{message['content']}\"\"\"\n\n"
                )
            elif message["role"] == "assistant":
                followup_summary += (
                    f"- KI: \"\"\"{message['content']}\"\"\"\n\n"
                )

        followup_convo = [
            {"role": "system", "content": SYSTEM_PROMPT_FOLLOWUP},
            {"role": "user", "content": followup_summary},
        ]
        followup_question, _tool_calls = query_api(
            followup_convo,
            tools=[],
            max_tokens=128,
            temperature=0.8,
            model="mistralai/mistral-medium-3.1",
        )

        conversation.append({"role": "user", "content": followup_question})

    if system_prompt:
        conversation[0]["content"] = system_prompt

    return {
        "conversations": (
            conversation if random.random() < 0.9 else conversation[1:]
        ),
        "tools": tools,
        "original_file": original_file,
        "id": str(uuid4())[:8],
    }


def read_list_file(
    input_folder: str,
    list_: list[tuple[str, str]],
    file: str,
    ignore_invalid: bool = False,
):
    """
    Reads a .txt file and returns a list of strings contained in the file,
    split by the delimiter "===*===".

    ignore_invalid = True will also include files with "__" in the name.
    """

    full_path = os.path.join(input_folder, file)
    if not os.path.isfile(full_path) or ("__" in file and not ignore_invalid):
        return []

    print("Reading ", full_path)

    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.read().split("===*===")
        shortened_filename = file.split(os.sep)[-1]
        list_.extend(
            [
                (line.strip(), shortened_filename)
                for line in lines
                if line.strip()
            ]
        )


def read_list_files(input_folder: str) -> list[tuple[str, str]]:
    """
    Reads all .txt files in the given folder and returns a list of strings
    contained in those files, split by the delimiter "===*===".
    """

    files = filter(lambda f: f.endswith(".txt"), os.listdir(input_folder))
    conversation_starters = []

    for file in files:
        read_list_file(input_folder, conversation_starters, file)

    return conversation_starters


def read_conversation_starters() -> list[tuple[str, str]]:
    """
    Reads the conversation starters from the dataset file and returns them as
    a list of strings.
    """

    return read_list_files(INPUT_FOLDER)


def read_system_prompts() -> list[str]:
    """
    Reads the system prompts from the dataset file and returns them as a list
    of strings.
    """

    system_prompts = []

    read_list_file(INPUT_FOLDER, system_prompts, "__system_prompts.txt", True)

    return system_prompts


def write_conversation(conversation: dict, filename: str):
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

    parser = argparse.ArgumentParser(
        description="Build dataset from conversation starters."
    )
    parser.add_argument(
        "--prod-run",
        action="store_true",
        help="If set, enables tool usage in the conversations.",
        default=False,
    )

    args = parser.parse_args()

    starters = read_conversation_starters()
    system_prompts = read_system_prompts()
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Found", len(starters), "conversation starters.")

    output_filename = os.path.join(
        OUTPUT_FOLDER,
        f"convs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
    )

    create_output_file(output_filename)

    for idx, starter in enumerate(starters):
        print(
            f"Building conversation {idx + 1}/{len(starters)} \
({idx / len(starters) * 100:.1f}%)..."
        )

        system_prompt = ""
        if random.random() < 0.5:
            system_prompt = random.choice(system_prompts)

        try:
            conversation = build_conversation(
                starter[0],
                original_file=starter[1],
                system_prompt=system_prompt,
            )
            write_conversation(conversation, output_filename)
            print(
                "\tWrote conversation with",
                len(conversation["conversations"]),
                "messages.",
            )
        except Exception as e:
            if args.prod_run:
                print(f"\tError building conversation: {e}")
            else:
                raise e

    print("Finished.")


if __name__ == "__main__":
    main()
