#!/home/ivan/experiments/openai/telegram/venv/bin/python

"""
AI script interpreter. Can execute scripts like:
```
#!/usr/bin/env ai.py
DESCRIPTION
Translate text into english.
PROMPT
Translate following text into English, don't add any additional text, just provide translation:
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
import google.generativeai as genai_v1

from google import genai
from google.genai import types


# --- Configuration ---
CONFIG_FILE = os.path.expanduser("~/.ai_script.cfg")

# --- Default Values ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Default if 'gemini' is chosen but no specific model
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
#DEFAULT_GEMINI_MODEL = "gemini-2.5.pro-exp-03-25"

# --- Helper Functions ---

def load_config():
    """Loads configuration from ~/.ai_script.cfg."""
    config = configparser.ConfigParser()
    if not os.path.exists(CONFIG_FILE):
        print(f"Warning: Configuration file not found at {CONFIG_FILE}", file=sys.stderr)
        return {"gemini_api_key": None, "default_model": "ollama:gemma3:1b"} # Sensible default fallback

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
    mode = None  # Can be 'DESCRIPTION', 'PROMPT', or None

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

        for line in lines:
            stripped_line = line.strip()
            if stripped_line == "DESCRIPTION":
                mode = "DESCRIPTION"
                continue
            elif stripped_line == "PROMPT":
                mode = "PROMPT"
                continue

            if mode == "DESCRIPTION":
                description_lines.append(line)
            elif mode == "PROMPT":
                prompt_lines.append(line)

        if not description_lines:
             print(f"Warning: No DESCRIPTION block found in {script_path}", file=sys.stderr)

        if not prompt_lines:
            raise ValueError(f"No PROMPT block found in {script_path}")

        description = "".join(description_lines).strip()
        prompt_template = "".join(prompt_lines).strip()

        return description, prompt_template

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

def invoke_ollama(prompt, model, ollama_url):
    """Invokes the Ollama API."""
    debug(f"--- Invoking Ollama ({model} at {ollama_url}) ---", file=sys.stderr)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False, # Keep it simple for now
    }
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, headers=headers, timeout=120) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["response"]
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
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON response from Ollama.", file=sys.stderr)
        print(f"Ollama Raw Response: {response.text}", file=sys.stderr)
        sys.exit(1)
    except KeyError:
         print(f"Error: Unexpected response format from Ollama (missing 'response' key).", file=sys.stderr)
         print(f"Ollama Raw Response: {response.json()}", file=sys.stderr)
         sys.exit(1)


def invoke_gemini(prompt, api_key):
    """Invokes the Google Gemini API."""
    debug(f"--- Invoking Gemini ({DEFAULT_GEMINI_MODEL}) ---", file=sys.stderr)
    if not api_key:
        print("Error: Gemini API key not found in config file (~/.ai_script.cfg).", file=sys.stderr)
        print("Please add 'gemini_api_key = YOUR_API_KEY' under [Credentials].", file=sys.stderr)
        sys.exit(1)

    try:
        genai_v1.configure(api_key=api_key)

        # Basic config, adjust as needed
        generation_config = {
            "temperature": 0.7, # Adjusted for potentially more predictable output
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048, # Increased token limit
            "response_mime_type": "text/plain",
        }

        model = genai_v1.GenerativeModel(
            model_name=DEFAULT_GEMINI_MODEL,
            generation_config=generation_config,
            # safety_settings = Adjust safety settings if needed
        )

        chat_session = model.start_chat() # Simpler than chat for single turn
        response = chat_session.send_message(prompt)
        return response.text

    except Exception as e:
        print(f"Error invoking Gemini API: {e}", file=sys.stderr)
        # Attempt to access specific Gemini error details if available
        if hasattr(e, 'message'):
             print(f"Gemini Error Message: {e.message}", file=sys.stderr)
        sys.exit(1)

def invoke_gemini_v2(system, prompt, api_key):
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
    system_instruction = []
    if system:
        system_instruction = [
            types.Part.from_text(text=system)
        ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=system_instruction,
    )

    res = []
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        res += chunk.text
    return "".join(res)

# --- Main Execution ---

def debug(message, *args, **kwargs):
    pass

def main():
    global debug
    # 1. Load Configuration
    config = load_config()
    gemini_api_key = config["gemini_api_key"]
    default_model_setting = config["default_model"]
    ollama_url = config["ollama_url"]

    # 2. Setup Argument Parser
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
        debug = print

    # 3. Parse the AI Script File
    script_description, prompt_template = parse_ai_script(args.script_path)

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
        print(prompt_template)
        sys.exit(0)


    # 6. Determine Input for {ARGS}
    if args.optional_text:
        input_args = " ".join(args.optional_text)
        debug("--- Using command line arguments for {ARGS} ---", file=sys.stderr)
    else:
        if sys.stdin.isatty():
             debug("--- Reading from stdin for {ARGS} (press Ctrl+D to end) ---", file=sys.stderr)
        else:
             debug("--- Reading piped stdin for {ARGS} ---", file=sys.stderr)
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
        debug(f"--- Using specified Ollama model: {chosen_model} ---", file=sys.stderr)
    elif args.gemini:
        chosen_backend = "gemini"
        # Gemini model name is currently fixed in invoke_gemini, but could be made configurable
        chosen_model = DEFAULT_GEMINI_MODEL # Placeholder, actual model used in invoke_gemini
        debug(f"--- Using specified Gemini model ---", file=sys.stderr)
    else:
        # Use default from config
        debug(f"--- Using default model from config: '{default_model_setting}' ---", file=sys.stderr)
        if default_model_setting.startswith("ollama:"):
            chosen_backend = "ollama"
            chosen_model = default_model_setting.split(":", 1)[1]
            if not chosen_model:
                 print(f"Error: Invalid default_model format in config: '{default_model_setting}'. Expected 'ollama:model_name'.", file=sys.stderr)
                 sys.exit(1)
        elif default_model_setting == "gemini":
            chosen_backend = "gemini"
            chosen_model = DEFAULT_GEMINI_MODEL # Placeholder
        else:
            # This case should have been caught by config loading, but double-check
            print(f"Error: Unknown default_model type in config: '{default_model_setting}'. Must be 'gemini' or start with 'ollama:'.", file=sys.stderr)
            sys.exit(1)

    # 9. Invoke Backend
    result = ""
    if chosen_backend == "ollama":
        result = invoke_ollama(final_prompt, chosen_model, ollama_url)
    elif chosen_backend == "gemini":
        result = invoke_gemini_v2("First line of user message is the instruction", final_prompt, gemini_api_key)
    else:
        # Should not happen due to earlier checks
        print("Internal Error: No backend selected.", file=sys.stderr)
        sys.exit(1)

    # 10. Print Result
    print(result)
    sys.exit(0)


if __name__ == "__main__":
    main()
