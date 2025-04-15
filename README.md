# AI Script

This project provides a simple scripting language for interacting with Large Language Models (LLMs) like Google Gemini and local Ollama instances. You can define prompts, system instructions, and even basic configuration within a script file, making it easy to create and execute reusable AI tasks from the command line.

## Features

* **Script-based prompts:** Define your prompts in a structured format within a dedicated file.
* **Multiple backends:** Supports Google Gemini and local Ollama as LLM providers.
* **Command-line execution:** Execute scripts directly from your terminal.
* **Argument passing:** Provide input to your prompts via command-line arguments or standard input.
* **Configuration file:** Customize default settings like the default model and API keys.
* **Help and prompt printing:** Get script-specific help and print the final prompt before execution.

## Installation

1.  **Clone the repository (or download the script directly):**
    ```bash
    git clone https://github.com/IvanVolosyuk/ai_script
    cd ai_script
    ```

2.  **Install required Python packages:**
    ```bash
    python -m venv venv
    . venv/bin/activate
    pip install google-genai requests
    ```

3.  **Make the script executable (if you downloaded it directly):**
    ```bash
    chmod +x ai.py
    chmod +x en
    ```

## Configuration

You can configure the AI script interpreter by creating a configuration file at `~/.ai_script.cfg`. This file allows you to set your Google Gemini API key, the default LLM model, and the URL for your Ollama instance.

Here's an example of the `~/.ai_script.cfg` file:

```ini
[Credentials]
gemini_api_key = YOUR_GEMINI_API_KEY

[Defaults]
default_model = ollama:gemma3:1b  ; or gemini

[Ollama]
url = http://localhost:11434
```

**Note:** Replace ```YOUR_GEMINI_API_KEY``` with your actual Google Gemini API key if you plan to use Gemini. You can obtain one from the Google Cloud AI Platform.

The ```default_model``` specifies the LLM to use by default. Use ```gemini``` for Google's model or ```ollama:MODEL_NAME``` (e.g., ```ollama:gemma3:1b```) for a local Ollama model.

The url under the [Ollama] section specifies the address of your local Ollama server. The default is ```http://localhost:11434```.
If the configuration file is not found, the script will use default values.

## Usage

AI Script Structure
An AI script file is structured using the following blocks:

* **DESCRIPTION (Required)**: A human-readable description of what the script does.
* **SETTINGS (Optional)**: Configuration settings specific to this script, such as model parameters.
* **SYSTEM (Optional)**: Instructions for the LLM that set the context or guide its behavior.
* **PROMPT (Required)**: The main prompt template that will be sent to the LLM. You can use {ARGS} as a placeholder for input provided via command-line arguments or standard input.

Here's an example of an AI script (spell_check):

```
#!/usr/bin/env ai.py

DESCRIPTION
Fix spelling errors

SYSTEM
Repeat following text exactly, fix spelling errors:

PROMPT
{ARGS}
```

```bash
echo Hi | ./ru
./ru Hi
```

## Command-Line Options

Each script supports the following command-line options:
```
-h, --help: Show this help message, including the script's DESCRIPTION, and exit.
-d, -v, --debug: Print extra debug logging to standard error.
-l OLLAMA_MODEL_NAME, --ollama OLLAMA_MODEL_NAME: Override the default model and use a local Ollama instance with the specified model name (e.g., gemma3:1b).
-g, --gemini: Override the default model and use Google Gemini.
-p, --print-prompt: Print the final prompt to standard output instead of executing it.
```

## Examples

Get help for a specific script:

```bash
./en -h
```

Use a specific Ollama model:
```bash
./ru -l gemma3:1b "Translate this sentence."
```

Force the use of Gemini:
```bash
./ru -g Hello World
```

Print the prompt before execution:

```bash
./impl -p
```

## Advanced Usage

### Script-Specific Settings

You can include a SETTINGS block in your AI script to configure parameters specific to that script. Currently, the script supports setting the temperature for the LLM within the [Model] section.

Example (summarize):

```
#!/usr/bin/env ai.py
DESCRIPTION
Summarize a given text.

SETTINGS
[Model]
temperature = 0.5

SYSTEM
You are an expert summarizer. Provide a concise summary of the following text.

PROMPT
Summarize this:
{ARGS}
```

This will set the temperature for the LLM to 0.5 when this script is executed.

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests to suggest improvements or report bugs.

## License

MIT License
