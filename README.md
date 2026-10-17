# Emotional Prompt

A Python script to mimic emotions in LLMs using a local [Ollama](https://ollama.com/) instance.

## Overview

The Emotion Prompt System interacts with Ollama to classify user input into emotional topics, map topics to underlying emotions (such as happiness, sadness, anger, fear, surprise, disgust, love, and indifference), dynamically update its emotion mappings with new keywords, and generate emotion-encoded responses.

## Requirements

- Python 3.10+
- `requests` (install via `pip install -r requirements.txt`)
- [Ollama](https://ollama.com/) running locally

## Installation

```bash
git clone https://github.com/Josh-Archer/emotional-prompt.git
cd emotional-prompt
pip install -r requirements.txt
```

## Usage

Ensure Ollama is running before starting the script:

```bash
ollama run llama3.2
```

Run the interactive prompt:

```bash
python emotion-prompt.py
```

Type your prompt when prompted with `> `, or type `quit` to exit.

### Command-Line Arguments

You can configure the Ollama model and server port via command-line arguments:

```
usage: emotion-prompt.py [-h] [--model MODEL] [--port PORT]

options:
  -h, --help            show this help message and exit
  --model, -m MODEL     Ollama model to use (default: llama3.2)
  --port, -p PORT       Ollama server port (default: 11434)
```

#### Examples

- Use the default model (`llama3.2`) and port (`11434`):
  ```bash
  python emotion-prompt.py
  ```

- Specify a different model:
  ```bash
  python emotion-prompt.py --model mistral
  # or using the short flag:
  python emotion-prompt.py -m mistral
  ```

- Specify a custom Ollama server port:
  ```bash
  python emotion-prompt.py --port 11435
  # or using the short flag:
  python emotion-prompt.py -p 11435
  ```

- Specify both a custom model and port:
  ```bash
  python emotion-prompt.py --model mistral --port 11435
  # or using short flags:
  python emotion-prompt.py -m mistral -p 11435
  ```

- Show help information:
  ```bash
  python emotion-prompt.py --help
  ```

## Running Tests

To run the test suite, ensure `pytest` is installed and run:

```bash
pytest
```
