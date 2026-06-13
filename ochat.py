#!/usr/bin/env python3
"""ochat — terminal chat client for ollama. no external dependencies."""

import argparse
import curses
import json
import os
import queue
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
COMPACT_AT  = 50_000
SIDEBAR_W   = 22

# ── ollama helpers ────────────────────────────────────────────────────────────────────────

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


def stream_request(model: str, messages: list[dict], tok_q: queue.Queue) -> None:
    """Runs in a thread. Puts str tokens, a dict stats object, or an Exception."""
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            for line in resp:
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok = chunk.get("message", {}).get("content", "")
                if tok:
                    tok_q.put(tok)
                if chunk.get("done"):
                    tok_q.put({
                        "prompt_eval_count": chunk.get("prompt_eval_count", 0),
                        "eval_count":        chunk.get("eval_count", 0),
                        "eval_duration":     chunk.get("eval_duration", 0),
                    })
                    break
    except Exception as e:
        tok_q.put(e)


def compact_history(history: list[dict]) -> list[dict]:
    if len(history) <= 2:
        return history
    drop = max(1, (len(history) // 2) // 2) * 2
    return history[drop:]


# ── ANSI simple mode ────────────────────────────────────────────────────────────────────

R  = "\033[0m"; B  = "\033[1m"; D  = "\033[2m"; I  = "\033[3m"
CY = "\033[36m"; GN = "\033[32m"; YL = "\033[33m"
BG = "\033[48;5;236m"; CS = "\033[s"; CR = "\033[u"; CE = "\033[J"


def render_md_ansi(text: str) -> str:
    lines, out, in_code = text.split("\n"), [], False
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
            lv, c = len(m.group(1)), m.group(2)
            if lv == 1:
                out += [f"{B}{YL}{c}{R}", f"{B}{YL}{'═'*len(c)}{R}"]
            elif lv == 2:
                out += [f"{B}{GN}{c}{R}", f"{D}{'─'*len(c)}{R}"]
            else:
                out.append(f"{B}{c}{R}")
            continue
        line = re.sub(r"\*\*\*(.*?)\*\*\*", lambda m: f"{B}{I}{m.group(1)}{R}", line)
        line = re.sub(r"\*\*(.*?)\*\*",     lambda m: f"{B}{m.group(1)}{R}",    line)
        line = re.sub(r"\*(.*?)\*",         lambda m: f"{I}{m.group(1)}{R}",    line)
        line = re.sub(r"`(.*?)`",           lambda m: f"{CY}{m.group(1)}{R}",   line)
        line = re.sub(r"^(\s*)([-*+]) ",    r"\1• ",                            line)
        out.append(line)
    return "\n".join(out)


def simple_mode(models: list[str], ollama_proc) -> None:
    def cleanup(*_):
        if ollama_proc:
            print(f"\n{D}stopping ollama...{R}", flush=True)
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if len(models) == 1:
        model = models[0]
        print(f"model: {B}{model}{R}")
    else:
        print(f"\n{B}models:{R}")
        for i, m in enumerate(models, 1):
            print(f"  {D}{i}.{R} {m}")
        while True:
            try:
                raw = input(f"\n{D}>{R} ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if raw.isdigit() and 1 <= int(raw) <= len(models):
                model = models[int(raw) - 1]; break
            if raw in models:
                model = raw; break
            print(f"{YL}?{R}")

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
            history.clear(); total_ctx = 0
            print(f"{D}cleared{R}"); continue

        history.append({"role": "user", "content": user_in})
        tok_q: queue.Queue = queue.Queue()
        threading.Thread(target=stream_request, args=(model, history, tok_q), daemon=True).start()

        full = ""
        sys.stdout.write(f"\n{B}{GN}assistant{R}\n{CS}")
        sys.stdout.flush()
        prompt_tok = eval_tok = 0; tps = 0.0

        while True:
            item = tok_q.get()
            if isinstance(item, str):
                full += item
                sys.stdout.write(item); sys.stdout.flush()
            elif isinstance(item, dict):
                prompt_tok = item["prompt_eval_count"]
                eval_tok   = item["eval_count"]
                ns         = item["eval_duration"]
                tps        = eval_tok / (ns / 1e9) if ns > 0 else 0.0
                break
            elif isinstance(item, Exception):
                print(f"\n{YL}error: {item}{R}"); history.pop(); full = ""; break

        if full:
            sys.stdout.write(CR + CE + render_md_ansi(full) + "\n")
            sys.stdout.flush()
            total_ctx = prompt_tok + eval_tok
            history.append({"role": "assistant", "content": full})
            tps_s = f" | {tps:.1f} tok/s" if tps > 0 else ""
            print(f"{D}[+{eval_tok} tokens | {total_ctx:,} in context{tps_s}]{R}\n")
            if total_ctx > COMPACT_AT:
                history = compact_history(history)
                print(f"{YL}[compacted to {len(history)} messages]{R}\n")
                total_ctx = 0


# ── curses TUI ───────────────────────────────────────────────────────────────────────────

CP_TITLE = 1; CP_USER = 2; CP_ASST = 3; CP_CODE = 4; CP_BORDER = 5


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(CP_TITLE,  curses.COLOR_WHITE,  curses.COLOR_BLUE)
    curses.init_pair(CP_USER,   curses.COLOR_YELLOW, -1)
    curses.init_pair(CP_ASST,   curses.COLOR_GREEN,  -1)
    curses.init_pair(CP_CODE,   curses.COLOR_CYAN,   -1)
    curses.init_pair(CP_BORDER, curses.COLOR_WHITE,  -1)


def md_to_curses_lines(text: str, width: int) -> list[tuple[str, int]]:
    """Convert markdown text to list of (line, curses_attr) pairs."""
    result = []
    in_code = False
    w = width - 4

    for raw in text.split("\n"):
        if raw.startswith("```"):
            if not in_code:
                in_code = True
                lang = raw[3:].strip()
                result.append(("  " + "─" * min(w, 40), curses.color_pair(CP_CODE) | curses.A_DIM))
                if lang:
                    result.append((f"  {lang}", curses.color_pair(CP_CODE) | curses.A_DIM))
            else:
                in_code = False
                result.append(("  " + "─" * min(w, 40), curses.color_pair(CP_CODE) | curses.A_DIM))
            continue

        if in_code:
            for seg in textwrap.wrap(raw, w) or [""]:
                result.append((f"  {seg}", curses.color_pair(CP_CODE)))
            continue

        hm = re.match(r"^(#{1,6}) (.*)", raw)
        if hm:
            content = hm.group(2)
            for seg in textwrap.wrap(content, w) or [content]:
                result.append((f"  {seg}", curses.color_pair(CP_USER) | curses.A_BOLD))
            continue

        clean = re.sub(r"\*\*\*(.*?)\*\*\*", r"\1", raw)
        clean = re.sub(r"\*\*(.*?)\*\*",     r"\1", clean)
        clean = re.sub(r"\*(.*?)\*",         r"\1", clean)
        clean = re.sub(r"`(.*?)`",           r"\1", clean)
        clean = re.sub(r"^(\s*)([-*+]) ",    r"\1• ", clean)

        for seg in textwrap.wrap(clean, w) or [""]:
            result.append((f"  {seg}", 0))

    return result


class OchatTUI:
    def __init__(self, stdscr, models: list[str], ollama_proc):
        self.scr      = stdscr
        self.models   = models
        self.ollama   = ollama_proc
        self.model    = None
        self.history: list[dict]             = []
        self.messages: list[tuple[str, str]] = []
        self.input_buf = ""
        self.scroll    = 0
        self.streaming = False
        self.stream_buf = ""
        self.tok_q: queue.Queue = queue.Queue()
        self.status = ""
        self.stats = {"ctx": 0, "last_tok": 0, "tps": 0.0, "msgs": 0}
        self.h = self.w = 0

    def resize(self):
        self.h, self.w = self.scr.getmaxyx()
        self.chat_w = max(20, self.w - SIDEBAR_W)
        self.chat_h = max(5,  self.h - 3)

    # ── drawing ───────────────────────────────────────────────────────────────────────────

    def _put(self, y, x, text, attr=0):
        if y < 0 or y >= self.h or x < 0:
            return
        text = text[:max(0, self.w - x - 1)]
        try:
            self.scr.addstr(y, x, text, attr)
        except curses.error:
            pass

    def draw_title(self):
        model_str = self.model or "no model selected"
        line = f" ochat  {model_str}"
        self._put(0, 0, line.ljust(self.w - 1), curses.color_pair(CP_TITLE) | curses.A_BOLD)

    def draw_sidebar(self):
        x  = self.chat_w
        sw = self.w - x

        def sb(y, text, attr=0):
            if y < 1 or y >= self.h - 2:
                return
            self._put(y, x, "│", curses.color_pair(CP_BORDER))
            self._put(y, x + 1, (" " + text)[:sw - 1], attr)

        sb(1,  "STATS", curses.A_BOLD | curses.A_UNDERLINE)
        sb(2,  "─" * (sw - 2), curses.A_DIM)
        sb(3,  "model", curses.A_DIM)
        sb(4,  (self.model or "—")[:sw - 3], curses.color_pair(CP_ASST))
        sb(5,  "─" * (sw - 2), curses.A_DIM)
        sb(6,  "context", curses.A_DIM)
        sb(7,  f"{self.stats['ctx']:,} tok")
        sb(8,  "─" * (sw - 2), curses.A_DIM)
        sb(9,  "last reply", curses.A_DIM)
        sb(10, f"+{self.stats['last_tok']} tok")
        tps = self.stats["tps"]
        r = 11
        if tps > 0:
            sb(r, f"{tps:.1f} tok/s", curses.color_pair(CP_ASST)); r += 1
        sb(r, "─" * (sw - 2), curses.A_DIM); r += 1
        sb(r, "history", curses.A_DIM); r += 1
        sb(r, f"{self.stats['msgs']} messages")

        for y in range(1, self.h - 2):
            try:
                self.scr.addch(y, x, ord("│"), curses.color_pair(CP_BORDER))
            except curses.error:
                pass

    def draw_chat(self):
        rendered: list[tuple[str, int]] = []
        for role, text in self.messages:
            if role == "user":
                rendered.append((f" you", curses.color_pair(CP_USER) | curses.A_BOLD))
                for seg in textwrap.wrap(text, self.chat_w - 4) or [""]:
                    rendered.append((f"  {seg}", 0))
                rendered.append(("", 0))
            else:
                rendered.append((f" assistant", curses.color_pair(CP_ASST) | curses.A_BOLD))
                rendered.extend(md_to_curses_lines(text, self.chat_w))
                rendered.append(("", 0))

        if self.streaming and self.stream_buf:
            rendered.append((" assistant", curses.color_pair(CP_ASST) | curses.A_BOLD))
            for line in self.stream_buf.split("\n"):
                for seg in textwrap.wrap(line, self.chat_w - 4) or [""]:
                    rendered.append((f"  {seg}", curses.A_DIM))

        total   = len(rendered)
        visible = self.chat_h
        start   = max(0, total - visible - self.scroll)

        for row in range(visible):
            i = start + row
            text, attr = rendered[i] if i < total else ("", 0)
            self._put(row + 1, 0, text[:self.chat_w - 1].ljust(self.chat_w - 1), attr)

    def draw_input(self):
        prompt = "you: "
        avail  = self.w - len(prompt) - 3
        display = self.input_buf[-avail:] if len(self.input_buf) > avail else self.input_buf
        self._put(self.h - 2, 0, f" {prompt}{display}", curses.color_pair(CP_USER))
        try:
            self.scr.clrtoeol()
        except curses.error:
            pass

        hint = self.status or "/clear /model /quit  PgUp/PgDn scroll"
        self._put(self.h - 1, 0, f" {hint}"[:self.w - 1], curses.A_DIM)
        try:
            self.scr.clrtoeol()
        except curses.error:
            pass

        cx = min(1 + len(prompt) + len(self.input_buf), self.w - 2)
        try:
            self.scr.move(self.h - 2, cx)
        except curses.error:
            pass

    def draw(self):
        self.scr.erase()
        self.draw_title()
        self.draw_chat()
        self.draw_sidebar()
        self.draw_input()
        self.scr.refresh()

    # ── model selection overlay ─────────────────────────────────────────────────────────────────

    def model_select(self) -> str | None:
        idx = self.models.index(self.model) if self.model in self.models else 0
        ow  = min(54, self.w - 4)
        oh  = min(len(self.models) + 4, self.h - 4)
        oy  = (self.h - oh) // 2
        ox  = (self.w - ow) // 2

        # use blocking getch for the picker (disable the 50ms timeout)
        self.scr.nodelay(False)
        try:
            while True:
                self.scr.erase()
                self.draw_title()
                try:
                    win = curses.newwin(oh, ow, oy, ox)
                    win.box()
                    win.addstr(0, 2, " select model ", curses.A_BOLD)
                    for i, m in enumerate(self.models):
                        row = i + 2
                        if row >= oh - 1:
                            break
                        label = m[:ow - 6]
                        if i == idx:
                            # selected: green + bold + arrow
                            win.addstr(row, 2, f"▶ {label}", curses.color_pair(CP_ASST) | curses.A_BOLD)
                        else:
                            # unselected: normal text (not dim — dim is invisible on dark terminals)
                            win.addstr(row, 2, f"  {label}")
                    footer = " ↑↓ select  enter confirm  esc cancel "
                    win.addstr(oh - 1, max(0, (ow - len(footer)) // 2), footer[:ow - 2])
                    win.refresh()
                except curses.error:
                    pass

                key = self.scr.getch()
                if   key == curses.KEY_UP   and idx > 0:                    idx -= 1
                elif key == curses.KEY_DOWN and idx < len(self.models) - 1: idx += 1
                elif key in (10, 13):   return self.models[idx]
                elif key == 27:         return None
        finally:
            # restore non-blocking mode for the main loop
            self.scr.nodelay(True)

    # ── stream processing ─────────────────────────────────────────────────────────────────────

    def process_stream(self):
        try:
            while True:
                item = self.tok_q.get_nowait()
                if isinstance(item, str):
                    self.stream_buf += item
                elif isinstance(item, dict):
                    self.streaming  = False
                    reply           = self.stream_buf
                    self.stream_buf = ""
                    self.messages.append(("assistant", reply))
                    self.history.append({"role": "assistant", "content": reply})
                    pt  = item["prompt_eval_count"]
                    et  = item["eval_count"]
                    ns  = item["eval_duration"]
                    tps = et / (ns / 1e9) if ns > 0 else 0.0
                    ctx = pt + et
                    self.stats.update({"ctx": ctx, "last_tok": et, "tps": tps, "msgs": len(self.history)})
                    self.status = ""
                    self.scroll = 0
                    if ctx > COMPACT_AT:
                        self.history = compact_history(self.history)
                        self.stats.update({"ctx": 0, "msgs": len(self.history)})
                        self.status = f"compacted to {len(self.history)} messages"
                elif isinstance(item, Exception):
                    self.streaming  = False
                    self.stream_buf = ""
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                    if self.messages and self.messages[-1][0] == "user":
                        self.messages.pop()
                    self.status = f"error: {item}"
        except queue.Empty:
            pass

    # ── input handling ────────────────────────────────────────────────────────────────────────

    def send(self):
        text = self.input_buf.strip()
        if not text:
            return
        self.input_buf = ""

        if text in ("/quit", "/exit", "/q"):
            self.quit()
        if text == "/clear":
            self.history.clear(); self.messages.clear()
            self.stats  = {"ctx": 0, "last_tok": 0, "tps": 0.0, "msgs": 0}
            self.status = "cleared"; return
        if text == "/model":
            selected = self.model_select()
            if selected and selected != self.model:
                self.model = selected
                self.history.clear(); self.messages.clear()
                self.stats  = {"ctx": 0, "last_tok": 0, "tps": 0.0, "msgs": 0}
                self.status = f"switched to {selected}"
            return

        self.messages.append(("user", text))
        self.history.append({"role": "user", "content": text})
        self.stats["msgs"] = len(self.history)
        self.streaming  = True
        self.stream_buf = ""
        self.scroll     = 0
        self.status     = "streaming…"

        threading.Thread(
            target=stream_request, args=(self.model, self.history, self.tok_q), daemon=True
        ).start()

    def quit(self):
        if self.ollama:
            self.ollama.terminate()
            try:
                self.ollama.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ollama.kill()
        sys.exit(0)

    # ── main loop ─────────────────────────────────────────────────────────────────────────────

    def run(self):
        init_colors()
        curses.curs_set(1)
        self.scr.timeout(50)
        self.resize()

        self.model = self.model_select() or self.models[0]
        self.status = f"chatting with {self.model}"

        while True:
            self.process_stream()
            self.draw()

            key = self.scr.getch()
            if   key == curses.KEY_RESIZE:                              self.resize()
            elif key == curses.KEY_PPAGE:                               self.scroll = min(self.scroll + self.chat_h // 2, 9999)
            elif key == curses.KEY_NPAGE:                               self.scroll = max(0, self.scroll - self.chat_h // 2)
            elif key in (curses.KEY_BACKSPACE, 127, 8):                 self.input_buf = self.input_buf[:-1]
            elif key in (10, 13) and not self.streaming:                self.send()
            elif key in (3, 27):                                        self.quit()
            elif 32 <= key <= 126:                                      self.input_buf += chr(key)


# ── entry point ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(prog="ochat", description="ollama terminal chat — no external dependencies")
    ap.add_argument("--simple", "-s", action="store_true", help="plain text mode, no TUI")
    args = ap.parse_args()

    print("ochat — connecting to ollama...")
    ollama_proc = None
    if is_running():
        print("ollama already running")
    else:
        print("starting ollama...", end="", flush=True)
        try:
            ollama_proc = start_ollama()
            print(" ready")
        except (RuntimeError, FileNotFoundError) as e:
            sys.exit(f"error: {e}")

    models = list_models()
    if not models:
        sys.exit("no models found — run: ollama pull <model>")

    if args.simple or not sys.stdout.isatty():
        simple_mode(models, ollama_proc)
    else:
        try:
            curses.wrapper(lambda s: OchatTUI(s, models, ollama_proc).run())
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            if ollama_proc:
                ollama_proc.terminate()
                try:
                    ollama_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ollama_proc.kill()


if __name__ == "__main__":
    main()
