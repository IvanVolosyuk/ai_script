#!/usr/bin/env python3

# Need following packages:
# pip install google-genai

"""
AI script interpreter. Can execute scripts like:
```
#!/usr/bin/env ai.py
DESCRIPTION
Translate text into english.

SYSTEM
Translate following text into English, don't add any additional text, just provide translation:

PROMPT
{ARGS}
```

Each script becomes an executable with input to LLM '{ARGS}' provided as stdin or command parameters.
Each script supports command line parameters:
    -h - help
    -p - print prompt
etc...
"""

import sys
import os
import argparse
import configparser
import requests
import json
import logging

from google import genai
from google.genai import types


# --- Configuration ---
CONFIG_FILE = os.path.expanduser("~/.ai_script.cfg")

# --- Default Values ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Default if 'gemini' is chosen but no specific model
DEFAULT_TEMPERATURE = 0.7 # Default temperature value

# --- Helper Functions ---

def load_config():
    """Loads configuration from ~/.ai_script.cfg."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"Warning: Configuration file not found at {CONFIG_FILE}", file=sys.stderr)
        return {
                "gemini_api_key": None,
                "default_model": "ollama:gemma3:1b",
                "ollama_url": "http://localhost:11434",
                }

    try:
        config.read(CONFIG_FILE)
        settings = {
            "gemini_api_key": config.get('Credentials', 'gemini_api_key', fallback=None),
            "default_model": config.get('Defaults', 'default_model', fallback="ollama:gemma3:1b"),
            "ollama_url": config.get('Ollama', 'url', fallback=DEFAULT_OLLAMA_URL),
        }
        # Basic validation for default_model format
        if not (settings["default_model"].startswith("ollama:") or settings["default_model"] == "gemini"):
             print(f"Warning: Invalid default_model '{settings['default_model']}' in config. Using 'ollama:gemma3:1b'.", file=sys.stderr)
             settings["default_model"] = "ollama:gemma3:1b"

        return settings
    except Exception as e:
        print(f"Error reading config file {CONFIG_FILE}: {e}", file=sys.stderr)
        # Return defaults but signal error by missing API key if Gemini was intended
        api_key = None
        try:
            api_key = config.get('Credentials', 'gemini_api_key', fallback=None)
        except:
            pass
        return {"gemini_api_key": api_key, "default_model": "ollama:gemma3:1b", "ollama_url": DEFAULT_OLLAMA_URL}


def parse_ai_script(script_path):
    """Parses the .ai script file."""
    description = ""
    prompt_template = ""
    system_instruction = None
    mode = None  # Can be 'DESCRIPTION', 'PROMPT', 'SYSTEM', 'CONFIG' or None

    try:
        with open(script_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            raise ValueError("Script file is empty.")

        # Skip shebang if present
        if lines[0].startswith("#!"):
            lines = lines[1:]

        description_lines = []
        prompt_lines = []
        system_lines = []
        settings_lines = []

        for line in lines:
            stripped_line = line.strip()
            if stripped_line == "DESCRIPTION":
                mode = "DESCRIPTION"
                continue
            elif stripped_line == "PROMPT":
                mode = "PROMPT"
                continue
            elif stripped_line == "SYSTEM":
                mode = "SYSTEM"
                continue
            elif stripped_line == "SETTINGS":
                mode = "SETTINGS"
                continue

            if mode == "DESCRIPTION":
                description_lines.append(line)
            elif mode == "PROMPT":
                prompt_lines.append(line)
            elif mode == "SYSTEM":
                system_lines.append(line)
            elif mode == "SETTINGS": # Capture config lines
                settings_lines.append(line)

        if not description_lines:
             print(f"Warning: No DESCRIPTION block found in {script_path}", file=sys.stderr)

        if not prompt_lines:
            raise ValueError(f"No PROMPT block found in {script_path}")

        description = "".join(description_lines).strip()
        prompt_template = "".join(prompt_lines).strip()
        system_instruction = "".join(system_lines).strip() if system_lines else None

        # Parse config lines for temperature
        settings_text = "".join(settings_lines)
        settings = None
        if settings_text:
            try:
                settings = configparser.ConfigParser()
                settings.read_string(settings_text)
            except Exception as e:
                logging.warn(f"Error parsing CONFIG block: {e}.")


        return description, prompt_template, system_instruction, settings

    except FileNotFoundError:
        print(f"Error: Script file not found at {script_path}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing script file {script_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while reading {script_path}: {e}", file=sys.stderr)
        sys.exit(1)

# --- Backend Invocation Functions ---

def invoke_ollama(prompt, model, ollama_url, system, settings):
    """Invokes the Ollama API."""
    logging.debug(f"--- Invoking Ollama ({model} at {ollama_url}) ---")
    if system: # Prepend system instruction to prompt for Ollama
        prompt = f"{system}\n{prompt}"
    payload = {
        "model": model,
        "prompt": prompt,
    }

    if settings:
        options = {}
        temperature = settings.get("Model", "temperature")
        if temperature:
            options["temperature"] = float(temperature)
        payload["options"] = options

    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, headers=headers, timeout=120, stream=True)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        for line in response.iter_lines():
            if line: # filter out keep-alive new lines
                try:
                    json_data = json.loads(line)
                    if 'response' in json_data:
                        print(json_data["response"], end="", flush=True) # Print response chunk by chunk
                    elif 'error' in json_data:
                        print(f"Error from Ollama: {json_data['error']}", file=sys.stderr)
                        sys.exit(1)
                except json.JSONDecodeError:
                    logging.error(f"Error decoding JSON chunk: {line}")
                    continue # Skip to the next line in stream

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to Ollama server at {ollama_url}. Is it running?", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.Timeout:
        print(f"Error: Request to Ollama timed out.", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error invoking Ollama API: {e}", file=sys.stderr)
        # Try to print more details if available
        try:
            print(f"Ollama Response: {response.text}", file=sys.stderr)
        except: # Handle cases where response might not exist
             pass
        sys.exit(1)


def invoke_gemini_v2(system, prompt, api_key, settings):
    client = genai.Client(api_key=api_key)

    model = "gemini-2.0-flash-thinking-exp-01-21"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    if not system:
      system = "First line of user message is the instruction"

    system_instruction = [
            types.Part.from_text(text=system)
        ]
    extra_args = {}
    if settings:
        temperature = settings.get("Model", "temperature")
        if temperature:
            extra_args["temperature"] = float(temperature)

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=system_instruction,
        **extra_args
    )

    res = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="", flush=True)
    return "".join(res)

# --- Main Execution ---

def main():
    # Configure logging, default level is WARNING
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr, format='%(levelname)s: %(message)s')

    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="AI Script Interpreter.",
        add_help=False # Disable default help to customize it
    )
    parser.add_argument(
        'script_path',
        metavar='AI_SCRIPT_PATH',
        help='Path to the .ai script file.'
    )
    parser.add_argument(
        '-h', '--help',
        action='store_true',
        help='Show this help message, including the script description, and exit.'
    )
    parser.add_argument(
        '-d', '-v', '--debug',
        action='store_true',
        help='Print extra debug logging to stderr.'
    )
    parser.add_argument(
        '-l', '--ollama',
        metavar='OLLAMA_MODEL_NAME',
        type=str,
        help='Override default model: Use local Ollama with the specified model name (e.g., gemma3:1b).'
    )
    parser.add_argument(
        '-g', '--gemini',
        action='store_true',
        help='Override default model: Use Google Gemini model.'
    )
    parser.add_argument(
        '-p', '--print-prompt',
        action='store_true',
        help='Print the final prompt to stdout instead of executing it.'
    )
    # Capture all remaining arguments as optional text input
    parser.add_argument(
        'optional_text',
        metavar='[OPTIONAL TEXT]',
        nargs='*', # Zero or more arguments
        help='Optional text to substitute for {ARGS} in the prompt. If omitted, stdin is used.'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # 2. Load Configuration
    config = load_config()
    gemini_api_key = config["gemini_api_key"]
    default_model_setting = config["default_model"]
    ollama_url = config["ollama_url"]

    # 3. Parse the AI Script File
    script_description, prompt_template, system_instruction, settings = parse_ai_script(args.script_path)

    # 4. Handle Help Request
    if args.help:
        print(f"Usage: {os.path.basename(sys.argv[1])} [-h] [-l OLLAMA_MODEL_NAME] [-g] [-p] [ ...]\n")
        print(script_description if script_description else "  (No description provided in the script)")
        print()
        # Print parser's help, excluding the positional args for clarity here
        help_text = parser.format_help()
        # print(help_text)
        # Manually format the options part for better control
        options_part = help_text.split('options:')[1].split('options:')[0]
        print("options:")
        print(options_part)
        sys.exit(0)

    # 5. Handle Print Prompt Request
    if args.print_prompt:
        if system_instruction:
          print(f"SYSTEM\n{system_instruction}\nPROMPT")
        print(prompt_template)
        sys.exit(0)


    # 6. Determine Input for {ARGS}a
    input_args = ""
    if args.optional_text:
        input_args = " ".join(args.optional_text)
        logging.debug("--- Using command line arguments for {ARGS} ---")
    elif "{ARGS}" in prompt_template:
        if sys.stdin.isatty():
             logging.debug("--- Reading from stdin for {ARGS} (press Ctrl+D to end) ---")
        else:
             logging.debug("--- Reading piped stdin for {ARGS} ---")
        input_args = sys.stdin.read()
        if not input_args:
             print("Warning: No input provided via command line arguments or stdin for {ARGS}.", file=sys.stderr)

    # 7. Format the Prompt
    try:
        final_prompt = prompt_template.replace("{ARGS}", input_args)
    except Exception as e:
         print(f"Error formatting prompt template: {e}", file=sys.stderr)
         # This shouldn't usually happen with simple replace, but good practice
         sys.exit(1)

    # 8. Determine Backend and Model
    chosen_backend = None # 'ollama' or 'gemini'
    chosen_model = None

    if args.ollama and args.gemini:
        print("Error: Cannot specify both -l/--ollama and -g/--gemini.", file=sys.stderr)
        sys.exit(1)
    elif args.ollama:
        chosen_backend = "ollama"
        chosen_model = args.ollama
        logging.debug(f"--- Using specified Ollama model: {chosen_model} ---")
    elif args.gemini:
        chosen_backend = "gemini"
        chosen_model = None
        logging.debug(f"--- Using specified Gemini model ---")
    else:
        # Use default from config
        logging.debug(f"--- Using default model from config: '{default_model_setting}' ---")
        if default_model_setting.startswith("ollama:"):
            chosen_backend = "ollama"
            chosen_model = default_model_setting.split(":", 1)[1]
            if not chosen_model:
                 print(f"Error: Invalid default_model format in config: '{default_model_setting}'. Expected 'ollama:model_name'.", file=sys.stderr)
                 sys.exit(1)
        elif default_model_setting == "gemini":
            chosen_backend = "gemini"
            chosen_model = None
        else:
            # This case should have been caught by config loading, but double-check
            print(f"Error: Unknown default_model type in config: '{default_model_setting}'. Must be 'gemini' or start with 'ollama:'.", file=sys.stderr)
            sys.exit(1)

    # 9. Invoke Backend
    result = ""
    if chosen_backend == "ollama":
        invoke_ollama(final_prompt, chosen_model, ollama_url, system_instruction, settings)
    elif chosen_backend == "gemini":
        invoke_gemini_v2(system_instruction, final_prompt, gemini_api_key, settings)
    else:
        # Should not happen due to earlier checks
        print("Internal Error: No backend selected.", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
