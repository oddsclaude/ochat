#!/usr/bin/env python3
"""ochat — terminal chat client for ollama. no external dependencies."""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COMPACT_AT   = 50_000  # auto-compact context when total tokens exceed this

R  = "\033[0m"        # reset
B  = "\033[1m"        # bold
D  = "\033[2m"        # dim
I  = "\033[3m"        # italic
CY = "\033[36m"       # cyan
GN = "\033[32m"       # green
YL = "\033[33m"       # yellow
BG = "\033[48;5;236m" # dark bg for code blocks

CS = "\033[s"  # save cursor
CR = "\033[u"  # restore cursor
CE = "\033[J"  # clear to end of screen


def is_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def start_ollama() -> subprocess.Popen:
    proc = subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        time.sleep(0.5)
        if is_running():
            return proc
    proc.kill()
    raise RuntimeError("ollama failed to start")


def list_models() -> list[str]:
    with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags") as r:
        return [m["name"] for m in json.loads(r.read()).get("models", [])]


def render_md(text: str) -> str:
    lines = text.split("\n")
    out = []
    in_code = False

    for line in lines:
        if line.startswith("```"):
            if not in_code:
                in_code = True
                lang = line[3:].strip()
                out.append(f"{BG}{D}{'─' * 50}{R}")
                if lang:
                    out.append(f"{BG}{D} {lang}{R}")
            else:
                in_code = False
                out.append(f"{BG}{D}{'─' * 50}{R}")
            continue

        if in_code:
            out.append(f"{BG}{CY}{line}{R}")
            continue

        m = re.match(r"^(#{1,6}) (.*)", line)
        if m:
            level, content = len(m.group(1)), m.group(2)
            if level == 1:
                out.append(f"{B}{YL}{content}{R}")
                out.append(f"{B}{YL}{'═' * len(content)}{R}")
            elif level == 2:
                out.append(f"{B}{GN}{content}{R}")
                out.append(f"{D}{'─' * len(content)}{R}")
            else:
                out.append(f"{B}{content}{R}")
            continue

        line = re.sub(r"\*\*\*(.*?)\*\*\*", lambda m: f"{B}{I}{m.group(1)}{R}", line)
        line = re.sub(r"\*\*(.*?)\*\*",     lambda m: f"{B}{m.group(1)}{R}",    line)
        line = re.sub(r"\*(.*?)\*",         lambda m: f"{I}{m.group(1)}{R}",    line)
        line = re.sub(r"__(.*?)__",         lambda m: f"{B}{m.group(1)}{R}",    line)
        line = re.sub(r"_(.*?)_",           lambda m: f"{I}{m.group(1)}{R}",    line)
        line = re.sub(r"`(.*?)`",           lambda m: f"{CY}{m.group(1)}{R}",   line)
        line = re.sub(r"^(\s*)([-*+]) ",    r"\1• ",                       line)

        out.append(line)

    return "\n".join(out)


def chat(model: str, messages: list[dict]) -> tuple[str, int, int, float]:
    """Stream a chat response. Returns (reply, prompt_tokens, eval_tokens, tokens_per_sec)."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": True,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    full = ""
    prompt_tokens = 0
    eval_tokens = 0
    eval_ns = 0

    sys.stdout.write(f"\n{B}{GN}assistant{R}\n")
    sys.stdout.write(CS)  # save cursor before streaming
    sys.stdout.flush()

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            full += token
            sys.stdout.write(token)
            sys.stdout.flush()
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)
                eval_tokens   = chunk.get("eval_count", 0)
                eval_ns       = chunk.get("eval_duration", 0)
                break

    # replace raw stream with markdown-rendered version
    sys.stdout.write(CR + CE)
    sys.stdout.write(render_md(full))
    sys.stdout.write("\n")
    sys.stdout.flush()

    tps = eval_tokens / (eval_ns / 1e9) if eval_ns > 0 else 0.0
    return full, prompt_tokens, eval_tokens, tps


def compact_history(history: list[dict]) -> list[dict]:
    """Drop the oldest half of message pairs to reduce context."""
    if len(history) <= 2:
        return history
    pairs = len(history) // 2
    drop = max(1, pairs // 2) * 2  # must be even
    return history[drop:]


def pick_model(models: list[str]) -> str:
    if not models:
        sys.exit(f"{YL}no models found. run: ollama pull <model>{R}")
    if len(models) == 1:
        print(f"model: {B}{models[0]}{R}")
        return models[0]
    print(f"\n{B}models:{R}")
    for i, m in enumerate(models, 1):
        print(f"  {D}{i}.{R} {m}")
    while True:
        try:
            raw = input(f"\n{D}>{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        if raw in models:
            return raw
        print(f"{YL}?{R}")


def main():
    ollama_proc = None

    def cleanup(*_):
        if ollama_proc is not None:
            print(f"\n{D}stopping ollama...{R}", flush=True)
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"{B}ochat{R} — ollama terminal chat")

    if is_running():
        print(f"{D}ollama is already running{R}")
    else:
        print(f"{D}starting ollama...{R}", end="", flush=True)
        try:
            ollama_proc = start_ollama()
            print(f" {GN}ready{R}")
        except (RuntimeError, FileNotFoundError) as e:
            sys.exit(f"\n{YL}error: {e}{R}")

    models = list_models()
    model = pick_model(models)
    print(f"\n{D}chatting with {model}  /clear /model /quit{R}\n")

    history: list[dict] = []
    total_ctx = 0

    while True:
        try:
            user_in = input(f"{B}you{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            cleanup()

        if not user_in:
            continue
        if user_in in ("/quit", "/exit", "/q"):
            cleanup()
        if user_in == "/clear":
            history.clear()
            total_ctx = 0
            print(f"{D}cleared{R}")
            continue
        if user_in == "/model":
            model = pick_model(models)
            history.clear()
            total_ctx = 0
            print(f"{D}switched to {model}{R}")
            continue

        history.append({"role": "user", "content": user_in})
        try:
            reply, prompt_tok, eval_tok, tps = chat(model, history)
            total_ctx = prompt_tok + eval_tok
            history.append({"role": "assistant", "content": reply})

            tps_str = f" | {tps:.1f} tok/s" if tps > 0 else ""
            print(f"{D}[+{eval_tok} tokens | {total_ctx:,} in context{tps_str}]{R}\n")

            if total_ctx > COMPACT_AT:
                history = compact_history(history)
                print(f"{YL}[context exceeded {COMPACT_AT:,} tokens — history compacted to {len(history)} messages]{R}\n")
                total_ctx = 0

        except urllib.error.URLError as e:
            history.pop()
            print(f"{YL}connection error: {e}{R}")
        except Exception as e:
            history.pop()
            print(f"{YL}error: {e}{R}")


if __name__ == "__main__":
    main()
