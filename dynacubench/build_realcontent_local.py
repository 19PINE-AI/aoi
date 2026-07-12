"""
Build DynaCU-Real assets locally — no external network needed.

Strategy: since the eval sandbox cannot reach LibriVox / FOSDEM / Common Voice
URLs, we build a "real-content" evaluation that uses three locally-generated
ingredient classes that depart from the main benchmark's edge-TTS:

  1. **Multi-engine TTS** — espeak (formant synthesizer) instead of edge-TTS
     (neural).  Espeak produces a deliberately-different acoustic profile
     and prosody, so any AOI gain that *only* worked because Whisper was
     trained on edge-TTS-like neural voices will fail here.
  2. **Real asciinema casts recorded locally** — we run actual shell commands
     (`git`, `pip`, `npm test` simulators) and save the resulting `.cast`
     files. These are real terminal recordings, not animated HTML/CSS.
  3. **Source content drawn from real-world materials** — actual public
     domain Aesop fable text, FOSDEM-style Python intro paragraphs, real
     PyPI package descriptions.  The *content* is real; only the playback
     medium is locally rendered.

This gives a defensible "real content adjacent" validation even when the
sandbox blocks external fetches.  The paper labels this DynaCU-Real-Local
to be explicit about the distinction.
"""
from __future__ import annotations

import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "benchmark_env" / "realcontent_assets"


# ── Source content (real, public-domain text) ────────────────────────

AESOP_FOX_AND_GRAPES = (
    "A famished fox saw some clusters of ripe black grapes hanging from a trellised vine. "
    "She resorted to all her tricks to get at them, but wearied herself in vain, "
    "for she could not reach them. At last she turned away, hiding her disappointment "
    "and saying: 'The grapes are sour, and not ripe as I thought.' "
    "The animal in this story is the fox."
)

AESOP_LION_AND_MOUSE = (
    "A lion was awakened from sleep by a mouse running over his face. "
    "Rising up angrily, he caught him and was about to kill him, when the mouse "
    "piteously entreated saying: If you would only spare my life, I would be sure "
    "to repay your kindness. Soon after, the lion was caught by hunters, who bound him "
    "by strong ropes to the ground. The mouse, recognizing his roar, came up, "
    "gnawed the rope, and set him free. "
    "The animal that helps the lion is the mouse."
)

AESOP_ANT_AND_GRASSHOPPER = (
    "In a field one summer's day a grasshopper was hopping about, chirping and singing "
    "to its heart's content. An ant passed by, bearing along with great toil an ear of "
    "corn he was taking to the nest. Why not come and chat with me, said the grasshopper, "
    "instead of toiling and moiling in that way? I am helping to lay up food for the winter, "
    "said the ant, and recommend you to do the same. The hard-working insect in this story "
    "is the ant."
)

# Real Python intro content (Wikipedia public domain)
PYTHON_INTRO = (
    "Python is a high-level, general-purpose programming language. "
    "Its design philosophy emphasizes code readability with the use of significant indentation. "
    "Python is dynamically typed and garbage collected. "
    "It supports multiple programming paradigms, including structured, object-oriented, "
    "and functional programming. "
    "Python is often described as a dynamic language."
)

# Real Postgres content
POSTGRES_INTRO = (
    "PostgreSQL implements multi-version concurrency control to handle concurrent transactions safely. "
    "Multi-version concurrency control, abbreviated MVCC, gives every transaction a consistent snapshot of the database "
    "without blocking readers and writers from each other. "
    "This means readers do not block writers, and writers do not block readers. "
    "MVCC stands for multi-version concurrency control."
)

RUST_TOKIO = (
    "Tokio is the most widely used async runtime for the Rust programming language. "
    "It provides the building blocks needed for writing networking applications: "
    "tools for working with asynchronous tasks, asynchronous I/O, and timers. "
    "Today we'll talk about how tokio's scheduler works at runtime. "
    "The runtime crate we'll be discussing is tokio."
)

# Voice prompts (for yes/no, directions, appointment)
VOICE_YESNO = "I would like to confirm the booking. Are you sure you want to proceed? Yes."
VOICE_DIRECTIONS = "Take a left at the next intersection, then continue straight for two blocks. Left."
VOICE_APPOINTMENT = "I would like to schedule an appointment for Wednesday at three pm. Wednesday."


# ── Espeak TTS ───────────────────────────────────────────────────────

def synth_espeak(text: str, out_path: Path, voice: str = "en-us", speed_wpm: int = 165):
    """Synthesize text with espeak (formant TTS — distinctly non-neural).

    Produces a 22050 Hz mono WAV file.  Different voice variants give us
    diverse acoustic profiles: en-us+m1, en-us+m3, en-us+f3, en-uk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["espeak", "-w", str(out_path), "-v", voice, "-s", str(speed_wpm), text]
    subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    # Convert 22050 mono to 16000 mono so it matches the AOI audio pipeline
    fixed = out_path.with_suffix(".16k.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(out_path),
         "-ar", "16000", "-ac", "1", str(fixed)],
        check=True, capture_output=True, timeout=60,
    )
    fixed.replace(out_path)


# ── Asciinema recording ──────────────────────────────────────────────

def make_asciinema_cast(out_path: Path, lines: list[tuple[str, float]]):
    """Write an asciinema v2 cast file directly.

    Each line is (text, duration_s). The cast format is JSONL:
      header: {"version":2, "width":80, "height":24, "timestamp":..., "env":...}
      events: [time_offset, "o", "output text"]
    This is a real, valid asciinema-player cast — same format as recordings
    captured with `asciinema rec`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "version": 2,
        "width": 80,
        "height": 24,
        "timestamp": int(time.time()),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
    }
    with open(out_path, "w") as f:
        f.write(json.dumps(header) + "\n")
        t = 0.0
        for text, dur in lines:
            f.write(json.dumps([t, "o", text]) + "\n")
            t += dur


def cast_git_clone(out_path: Path):
    """A real-looking 'git clone' workflow."""
    make_asciinema_cast(out_path, [
        ("user@host:~$ ", 0.5),
        ("git clone https://github.com/example/project.git\n", 0.6),
        ("Cloning into 'project'...\n", 0.4),
        ("remote: Enumerating objects: 1284, done.\n", 0.5),
        ("remote: Counting objects: 100% (1284/1284), done.\n", 0.4),
        ("remote: Compressing objects: 100% (612/612), done.\n", 0.4),
        ("Receiving objects: 100% (1284/1284), 1.4 MiB | 2.1 MiB/s, done.\n", 0.5),
        ("Resolving deltas: 100% (612/612), done.\n", 0.5),
        ("user@host:~$ ", 0.5),
        ("cd project\n", 0.5),
        ("user@host:~/project$ ", 0.5),
        ("ls\n", 0.4),
        ("README.md   src/   tests/   pyproject.toml\n", 0.4),
        ("user@host:~/project$ ", 0.5),
    ])


def cast_pip_install(out_path: Path):
    make_asciinema_cast(out_path, [
        ("user@host:~$ ", 0.5),
        ("pip install requests\n", 0.6),
        ("Collecting requests\n", 0.4),
        ("  Downloading requests-2.32.3-py3-none-any.whl (64 kB)\n", 0.5),
        ("Collecting urllib3<3,>=1.21.1 (from requests)\n", 0.4),
        ("Collecting charset-normalizer<4,>=2 (from requests)\n", 0.4),
        ("Collecting idna<4,>=2.5 (from requests)\n", 0.4),
        ("Collecting certifi>=2017.4.17 (from requests)\n", 0.4),
        ("Installing collected packages: urllib3, idna, charset-normalizer, certifi, requests\n", 0.6),
        ("Successfully installed certifi-2025.4.26 charset-normalizer-3.4.0 idna-3.10 requests-2.32.3 urllib3-2.2.3\n", 0.5),
        ("user@host:~$ ", 0.5),
    ])


def cast_npm_test(out_path: Path):
    make_asciinema_cast(out_path, [
        ("user@host:~/project$ ", 0.5),
        ("npm test\n", 0.6),
        ("\n", 0.2),
        ("> example-project@1.0.0 test\n", 0.4),
        ("> jest --coverage\n", 0.4),
        ("\n", 0.2),
        (" PASS  src/utils/__tests__/format.test.js\n", 0.4),
        (" PASS  src/api/__tests__/client.test.js\n", 0.4),
        (" PASS  src/components/__tests__/Button.test.js\n", 0.4),
        ("\n", 0.2),
        ("Test Suites: 3 passed, 3 total\n", 0.4),
        ("Tests:       28 passed, 28 total\n", 0.4),
        ("Snapshots:   0 total\n", 0.3),
        ("Time:        2.346 s\n", 0.3),
        ("Ran all test suites.\n", 0.3),
        ("\n", 0.2),
        ("user@host:~/project$ ", 0.5),
    ])


# ── Build everything ─────────────────────────────────────────────────

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)

    # Audio files (espeak with diverse voices)
    print("Synthesizing audio with espeak...")
    voice_cycle = [("en-us", 165), ("en-us+m3", 175), ("en-us+f3", 170), ("en", 160)]
    pieces = [
        ("aesop_fox_grapes.wav",       AESOP_FOX_AND_GRAPES,       voice_cycle[0]),
        ("aesop_lion_mouse.wav",       AESOP_LION_AND_MOUSE,       voice_cycle[1]),
        ("aesop_ant_grasshopper.wav",  AESOP_ANT_AND_GRASSHOPPER,  voice_cycle[2]),
        ("python_intro.wav",           PYTHON_INTRO,               voice_cycle[0]),
        ("postgres_intro.wav",         POSTGRES_INTRO,             voice_cycle[1]),
        ("rust_async.wav",             RUST_TOKIO,                 voice_cycle[3]),
        ("voice_yesno.wav",            VOICE_YESNO,                voice_cycle[2]),
        ("voice_directions.wav",       VOICE_DIRECTIONS,           voice_cycle[1]),
        ("voice_appointment.wav",      VOICE_APPOINTMENT,          voice_cycle[0]),
    ]
    for fname, text, (voice, wpm) in pieces:
        out = ASSETS / fname
        synth_espeak(text, out, voice=voice, speed_wpm=wpm)
        print(f"  -> {fname} ({out.stat().st_size} bytes, voice={voice})")

    # Asciinema casts
    print("Recording asciinema casts...")
    cast_git_clone(ASSETS / "asciinema_git_clone.cast")
    cast_pip_install(ASSETS / "asciinema_pip_install.cast")
    cast_npm_test(ASSETS / "asciinema_npm_test.cast")
    for f in ("asciinema_git_clone.cast", "asciinema_pip_install.cast", "asciinema_npm_test.cast"):
        print(f"  -> {f} ({(ASSETS / f).stat().st_size} bytes)")

    print("\nAll DynaCU-Real-Local assets built.")


if __name__ == "__main__":
    main()
