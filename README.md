# `conversational-builder`

Dieses Repo beinhaltet Skripte zur Erzeugung synthetischer Datensätze in Leichter Sprache.
Es wurde zum Generieren der Beispielkonversationen im Trainingsdatensatz von der KI "Jacob" genutzt.

Viele Szenarien werden bei der Erstellung der Datensätze abgedeckt:

- verschiedenste Gesprächsthemen
- Tool-Calling
- mentale Gesundheit und Sicherheit/Alignment der auf dem Datensatz trainierten LLMs
- Multi-Turn-Konversationen
- typische Nutzungsmerkmale wie Rechtschreibfehler von Seiten des Users

Der Code ist überall mit kurzen Docstrings versehen und an komplexen Stellen kommentiert.
Sollte dennoch irgendetwas unklar sein, gerne ein Issue eröffnen und ich kümmere mich so schnell wie möglich!

## Nutzung

Es geht ganz einfach:

1. Repository klonen mit `git clone https://github.com/Jacob-ML/conversational-builder.git`
2. Virtual Environment erstellen und die ganzen Abhängigkeiten installieren: `python3 -m venv .venv && . ./.venv/bin/activate && pip install -r ./requirements.txt`
3. Gegebenenfalls noch die Umgebungsvariablen setzen: `cp ./.env.example ./.env && nano ./.env`
4. Los geht's! Zum Beispiel: `python3 ./src/build_dataset.py`

## Lizenz

Die Software steht unter der MIT-Lizenz zur Verfügung! Siehe die [`LICENSE`](./LICENSE)-Datei.
