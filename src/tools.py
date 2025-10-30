"""
This module defines potential tools that can be used by the AI assistant
and provides mock implementations for these tools. In a real-world scenario,
these would interface with actual APIs or services to perform their tasks.
"""

import datetime
import os
import random
import time
from typing import Union
import requests

# necessary imports; do not remove! used by the interpreter mock tool
import math
from math import sqrt, pow, log, exp, sin, cos, tan, pi

from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

POTENTIAL_TOOLS = {
    "web": {
        "names": [
            "web_search",
            "websearch",
            "web-search",
            "search",
            "internet_search",
            "internetsearch",
            "internet-search",
            "search_tool",
            "searchtool",
            "search-tool",
            "web_lookup",
            "weblookup",
            "web-lookup",
        ],
        "descriptions": [
            "Ein Tool, um im Internet nach aktuellen Informationen zu suchen.",
            "Sucht im Internet nach aktuellen Informationen.",
            "Findet aktuelle Informationen im Internet.",
            "Führt eine Websuche durch, um aktuelle Informationen zu finden.",
            "Sucht online nach dem Suchbegriff.",
            "Tool, um eine Websuche durchzuführen.",
            "Searches the web for current information.",
            "Finds information on the internet.",
            "Performs a web search to find information.",
            "Looks up information online.",
        ],
        "parameters": [
            {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Die Suchanfrage",
                    }
                },
                "required": ["query"],
            },
            {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Anfrage für die Suchmaschine",
                    }
                },
                "required": ["query"],
            },
            {
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to use",
                    }
                },
                "required": ["query"],
            },
        ],
    },
    "web_fetch": {
        "names": [
            "fetch_website",
            "open_website",
            "website-open",
            "open_url",
            "fetch_url",
            "url_fetch",
        ],
        "descriptions": [
            "Ein Tool, um den Inhalt einer Website abzurufen.",
            "Ruft den Inhalt einer Website über ihre URL ab.",
            "Findet und gibt den Inhalt einer Website zurück.",
            "Tool, um eine Website zu öffnen und deren Inhalt zu extrahieren.",
            "Fetches the content of a website given its URL.",
            "Retrieves and returns the content of a webpage.",
            "Opens a website and extracts its content via the URL.",
        ],
        "parameters": [
            {
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Vollständige URL zum Abrufen der Website",
                    }
                },
                "required": ["url"],
            },
            {
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Die URL der Website, die abgerufen werden soll",
                    }
                },
                "required": ["url"],
            },
            {
                "properties": {
                    "link": {
                        "type": "string",
                        "description": "Full URL to fetch content from",
                    }
                },
                "required": ["link"],
            },
        ],
    },
    "weather": {
        "names": [
            "weather",
            "get_weather",
            "weather_tool",
            "weather-tool",
            "weatherinfo",
            "weather_info",
            "weather-info",
        ],
        "descriptions": [
            "Ein Tool, um das aktuelle Wetter an einem Ort zu bekommen.",
            "Sucht das aktuelle Wetter für einen Ort.",
            "Findet Wetterinformationen für einen Ort.",
            "Gibt das aktuelle Wetter für einen Ort zurück.",
            "Tool, um Wetterdaten abzurufen.",
            "Gets the current weather for a location.",
            "Finds weather information for a place.",
            "Returns the current weather for a location.",
            "Retrieves weather data for a given place.",
        ],
        "parameters": [
            {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Der Ort, für den das Wetter abgefragt werden soll",
                    },
                },
                "required": ["location"],
            },
            {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Ort der Wetterabfrage",
                    },
                },
                "required": ["location"],
            },
            {
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for",
                    },
                },
                "required": ["location"],
            },
        ],
    },
    "interpreter": {
        "names": [
            "code_interpreter",
            "codeinterpreter",
            "code-interpreter",
            "interpreter",
            "python_interpreter",
            "pythoninterpreter",
            "python-interpreter",
        ],
        "descriptions": [
            "Ein Tool, um Python-Code auszuführen. Gibt das Ergebnis zurück. Das math-Modul ist verfügbar. Bitte Anweisungen mit ';' trennen. Teuer, nur bei schwierigen Berechnungen nutzen.",
            "Führt Python-Code aus und gibt das Ergebnis zurück. Das math-Modul ist verfügbar. Tool ist teuer, nur bei schwierigen Berechnungen nutzen. Mehrere Anweisungen mit ';' trennen.",
            "Tool, um Python-Code zu interpretieren. Gibt das Ergebnis zurück. Das math-Modul ist verfügbar. Mehrere Anweisungen mit ';' trennen. Aufwendig, nur wenn nötig bei schwierigen Berechnungen.",
            "Interpretiert und führt Python-Code aus, gibt das Ergebnis zurück. Du kannst das Math-Modul verwenden. Trenne mehrere Anweisungen mit ';'. Nur bei sehr schweren oder genauen Berechnungen nutzen, da es teuer ist.",
            "Executes Python code and returns the result. math module is available. Tool is expensive, use only for difficult calculations. Split several statements with ';'.",
            "Interprets and runs Python code, returns result. math module is available. Split several statements with ';'. Use only for very hard or precise calculations, as it is expensive.",
            "Runs Python code snippets and returns the result. Split several statements with ';'. math module is available. Expensive, use only when necessary for complex calculations.",
            "Tool to execute Python code and get the result. Expensive, only use for hard or precise calculations.",
        ],
        "parameters": [
            {
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Der Python-Ausdruck, der ausgewertet werden soll",
                    }
                },
                "required": ["code"],
            },
            {
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Auszuwertender Python-Ausdruck",
                    }
                },
                "required": ["expression"],
            },
            {
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code snippet to execute",
                    }
                },
                "required": ["code"],
            },
            {
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The Python expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        ],
    },
}


def mock_weather_tool(location: str) -> dict:
    """
    A mock implementation of a weather tool. In a real implementation, this
    would call an actual weather API to get current weather data.
    """

    response: dict[
        str, Union[str, float, list[dict[str, Union[str, float]]]]
    ] = {"unit": "celsius"}

    response[random.choice(["city", "city_or_town", "location", "place"])] = (
        location
    )

    response[
        random.choice(
            [
                "current_temperature",
                "temperature_cur",
                "temp_now",
                "temp_current",
            ]
        )
    ] = round(random.uniform(-10, 35), 1)

    picked_future_name = random.choice(
        [
            "forecast",
            "weather_forecast",
            "future_weather",
            "upcoming_weather",
            "weather_prediction",
            "weather_future",
        ]
    )

    response[picked_future_name] = []

    for day in range(1, 6):
        day_formatted = (
            datetime.datetime.now() + datetime.timedelta(days=day)
        ).strftime("%Y-%m-%d")
        day_forecast = {
            "day": day_formatted,
            "temperature_high": (round(random.uniform(0, 35), 1)),
            "temperature_low": (round(random.uniform(-10, 25), 1)),
            "condition": random.choice(
                [
                    "sunny",
                    "partly cloudy",
                    "cloudy",
                    "rainy",
                    "stormy",
                    "snowy",
                    "windy",
                    "foggy",
                ]
            ),
        }
        response[picked_future_name].append(day_forecast)  # type: ignore

    return response


def get_tool_response(tool_name: str, tool_args: dict) -> dict:
    """
    Given a tool name and its arguments, returns a mock response for the tool.
    In a real implementation, this would call the actual tool's functionality.
    """

    if tool_name in POTENTIAL_TOOLS["web"]["names"]:
        query = tool_args.get("query", "")
        print("SEARCHING", query)

        try:
            # input("RUN WEB SEARCH? " + query + "\nPress Enter to continue...")
            time.sleep(20)  # delay to make rate limits less likely to be hit
            pass
        except (KeyboardInterrupt, EOFError):
            print("Cancelled.")

            return {
                "error": "search failed",
                "details": "cannot search right now",
            }

        resp = requests.post(
            "https://ollama.com/api/web_search",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
            },
            json={"query": query, "max_results": random.randint(1, 4)},
        )

        if resp.status_code != 200:
            return {
                "error": "search failed",
                "details": f"status code {resp.status_code}",
            }

        return resp.json()

    elif tool_name in POTENTIAL_TOOLS["web_fetch"]["names"]:
        url = tool_args.get("url") or tool_args.get("link", "")
        print("FETCHING", url)

        try:
            # input("RUN WEB FETCH? " + url + "\nPress Enter to continue...")
            pass
        except (KeyboardInterrupt, EOFError):
            print("Cancelled.")

            return {
                "error": "fetch failed",
                "details": "cannot fetch right now",
            }

        return requests.post(
            "https://ollama.com/api/web_fetch",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
            },
            json={"url": url},
        ).json()

    elif tool_name in POTENTIAL_TOOLS["weather"]["names"]:
        location = tool_args.get("location", "")
        print("GETTING WEATHER", location)

        return mock_weather_tool(location)

    elif tool_name in POTENTIAL_TOOLS["interpreter"]["names"]:
        code = tool_args.get("code") or tool_args.get("expression", "")

        try:
            # input(
            #     "RUN THIS EXPRESSION? "
            #     + code
            #     + "\nPress Enter to continue, CTRL+C to cancel..."
            # )

            forbidden_keywords = [
                "import",
                "__",
                "os.",
                "sys.",
                "requests",
                "open(",
                "eval(",
                "exec(",
                "getattr",
                "setattr",
                "delattr",
                "globals()",
                "locals()",
                "compile(",
                "pickle",
                "subprocess",
                "shlex",
                "socket",
                "thread",
                "multiprocessing",
                "ctypes",
            ]

            if any(keyword in code for keyword in forbidden_keywords):
                print("Execution refused: forbidden keywords detected.", code)
                raise KeyboardInterrupt()

            print("RUNNING", code)
        except (KeyboardInterrupt, EOFError):
            print("Cancelled.")

            return {
                "error": "execution refused",
                "details": "could not evaluate expression",
            }

        def _run(expression):
            parts = expression.split(";")

            for part in parts[:-1]:
                exec(part.strip())

            last_part: str = parts[-1].strip()

            if "print(" in last_part:
                last_part = last_part.lstrip("print(").rstrip(";").rstrip(")")

            return eval(last_part)

        try:
            result = _run(code)

            if isinstance(result, float):
                if math.isinf(result) or math.isnan(result):
                    result = str(result)
                else:
                    result = round(result, 6)

            return {"result": result}
        except Exception as e:
            return {"error": "execution threw exception", "details": str(e)}

    else:
        return {"error": f"unknown tool {tool_name}"}
