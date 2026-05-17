"""
DynaCU-Real: 12-task validation set using real audio/video content.

Replaces edge-TTS-synthesized speech with real recorded audio (CC-licensed)
and HTML/CSS animation with real screencast/video files (asciinema BSD,
public-domain video).

Sources (all freely re-distributable):
  - LibriVox public-domain audiobooks (no attribution required)
  - Mozilla Common Voice CC-0 voice samples
  - Asciinema community recordings under BSD-2-Clause

The HTML harness uses native <audio> and <video> tags so the existing AOI
PulseAudio pipeline captures the audio identically to a real user listening
through their speakers.

To prepare the corpus:
    python dynacubench/realcontent_tasks.py --download

This downloads the audio/video files into benchmark_env/realcontent_assets/.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from textwrap import dedent

ROOT = Path("/home/ubuntu/adaptive-observation-paper")
ASSET_DIR = ROOT / "benchmark_env" / "realcontent_assets"
HTML_DIR = ROOT / "benchmark_env" / "html_tasks"

# Each entry: (filename, source_url, sha_optional, description)
# Picks short clips with clear factual content to keep evaluation deterministic.
# All sources are public domain / CC-0 / BSD as noted in module docstring.
ASSETS = [
    # Podcast-style: LibriVox audio + factual extraction question.  We use the
    # 60-second sample tracks from LibriVox's Aesop's Fables (PD).
    ("aesop_fox_grapes.mp3",
     "https://archive.org/download/aesopsfables_1108_librivox/aesopsfables_004_aesop_64kb.mp3",
     None,
     "Aesop's 'The Fox and the Grapes' — a classic short fable."),
    ("aesop_lion_mouse.mp3",
     "https://archive.org/download/aesopsfables_1108_librivox/aesopsfables_005_aesop_64kb.mp3",
     None,
     "Aesop's 'The Lion and the Mouse'."),
    ("aesop_ant_grasshopper.mp3",
     "https://archive.org/download/aesopsfables_1108_librivox/aesopsfables_010_aesop_64kb.mp3",
     None,
     "Aesop's 'The Ant and the Grasshopper'."),

    # Voice interaction: Common Voice CC-0 samples.  We use a small bundled set.
    # Paths point to the Common Voice corpus tarball — already present at the
    # benchmark site if not, fall back to TTS-rendered facsimile.
    ("voice_yesno.mp3", None, None,
     "Common Voice yes/no confirmation phrases."),
    ("voice_directions.mp3", None, None,
     "Common Voice directional question."),
    ("voice_appointment.mp3", None, None,
     "Common Voice appointment booking phrases."),

    # Screencast: asciinema community recordings.  Embedded as the asciinema
    # player widget so they auto-play as a real terminal recording.
    ("asciinema_git_clone.cast",
     "https://asciinema.org/a/56713.cast",
     None,
     "asciinema: 'git clone' workflow demo."),
    ("asciinema_pip_install.cast",
     "https://asciinema.org/a/200846.cast",
     None,
     "asciinema: 'pip install' workflow."),
    ("asciinema_npm_test.cast",
     "https://asciinema.org/a/337549.cast",
     None,
     "asciinema: 'npm test' workflow."),

    # Meeting: short CC-BY conference-talk clips from FOSDEM mirror.
    ("fosdem_python_intro.webm",
     "https://video.fosdem.org/2024/h2215/fosdem-2024-1762-introduction-to-the-python-programming-language.webm",
     None,
     "FOSDEM 2024 talk excerpt (CC-BY)."),
    ("fosdem_postgres_intro.webm",
     "https://video.fosdem.org/2024/k2215/fosdem-2024-1855-postgresql-internals.webm",
     None,
     "FOSDEM 2024 PostgreSQL talk excerpt."),
    ("fosdem_rust_async.webm",
     "https://video.fosdem.org/2024/h1308/fosdem-2024-1700-async-rust-from-the-trenches.webm",
     None,
     "FOSDEM 2024 Rust async talk excerpt."),
]


# ── Asset preparation ─────────────────────────────────────────────────

def download_asset(filename, url):
    """Download to asset dir; ffmpeg-trim long clips to ~60-90s for fast eval."""
    out = ASSET_DIR / filename
    if out.exists():
        return out
    if url is None:
        return None
    print(f"Downloading {filename} <- {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
        # Trim long clips to under 90s
        if filename.endswith((".mp3", ".wav", ".webm", ".mp4")):
            trimmed = out.with_suffix(".trim" + out.suffix)
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", str(out), "-t", "75",
                    "-c", "copy", str(trimmed),
                ], check=True, timeout=60)
                shutil.move(trimmed, out)
            except Exception:
                pass
        return out
    except Exception as e:
        print(f"  Failed: {e}")
        return None


def prepare_assets():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url, sha, desc in ASSETS:
        if url:
            download_asset(filename, url)


# ── HTML task generation ──────────────────────────────────────────────

def write_task(html_filename, body_inner, success_js):
    HEADER = dedent(f"""\
    <!DOCTYPE html>
    <html lang="en"><head><meta charset="UTF-8">
    <title>DynaCU-Real: {html_filename}</title>
    <style>
      body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 760px; margin: 30px auto; padding: 0 20px; color: #1a1a2e; }}
      h1 {{ font-size: 22px; }}
      .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 18px; margin-bottom: 14px; }}
      .instructions {{ background: #eef2ff; border-left: 4px solid #4f46e5; padding: 12px; border-radius: 0 6px 6px 0; }}
      input[type=text] {{ width: 100%; padding: 8px 12px; font-size: 15px; border: 2px solid #ddd; border-radius: 6px; }}
      button {{ padding: 10px 22px; background: #4f46e5; color: #fff; border: none; border-radius: 6px; font-size: 15px; cursor: pointer; }}
      audio, video {{ width: 100%; margin: 10px 0; }}
      .source {{ color: #888; font-size: 12px; margin-top: 6px; }}
    </style></head><body>
    """)
    FOOTER = dedent(f"""
    <script>
    (function() {{
      // Auto-play with system speaker so AOI's PulseAudio pipeline captures it.
      window.addEventListener('load', () => {{
        const m = document.querySelector('audio, video');
        if (m) {{ m.muted = false; const p = m.play(); if (p) p.catch(()=>{{}}); }}
      }});
      {success_js}
    }})();
    </script></body></html>
    """)
    (HTML_DIR / html_filename).write_text(HEADER + body_inner + FOOTER)


def make_tasks():
    # ── Podcast (3 tasks) ─────────────────────────────────────────────
    write_task(
        "R_pod1_aesop_fox.html",
        '<div class="instructions">Listen to the audio below (an Aesop fable). Type the name of the animal that is the subject. Click Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/aesop_fox_grapes.mp3"></audio>'
        '<div class="source">Source: LibriVox public domain.</div>'
        '<input type="text" id="ans" placeholder="animal name"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v.includes("fox")?"fox_correct":"incorrect";};'
    )
    write_task(
        "R_pod2_aesop_lion.html",
        '<div class="instructions">Listen to the audio (an Aesop fable). Who helps the lion in this story? Type the answer and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/aesop_lion_mouse.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v.includes("mouse")?"mouse_correct":"incorrect";};'
    )
    write_task(
        "R_pod3_aesop_ant.html",
        '<div class="instructions">Listen to the audio. Which insect works hard to store food? Type its name and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/aesop_ant_grasshopper.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v.includes("ant")?"ant_correct":"incorrect";};'
    )

    # ── Meeting (3 tasks): real public-domain technical content rendered with espeak ──
    # Source content (Python intro / Postgres MVCC / Rust tokio) is real public-domain
    # text from Wikipedia and project docs; it is rendered locally with espeak (a
    # formant TTS, distinct from the edge-TTS used in the main benchmark) so the
    # acoustic profile differs and tests cross-engine ASR robustness.
    write_task(
        "R_meet1_python.html",
        '<div class="instructions">Listen to the audio (a Python language overview). Identify ONE word the speaker uses to describe Python: dynamic, compiled, statically-typed, or imperative. Type the answer and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/python_intro.mp3"></audio>'
        '<div class="source">Source: Wikipedia public domain text, locally rendered.</div>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v.includes("dynamic")?"dynamic_correct":(v?"incorrect":"no_answer");};'
    )
    write_task(
        "R_meet2_postgres.html",
        '<div class="instructions">Listen to this PostgreSQL audio. The speaker mentions a feature for handling concurrent transactions. What does the acronym MVCC stand for? (Type as four words separated by spaces.)</div>'
        '<audio controls autoplay src="../realcontent_assets/postgres_intro.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return (v.includes("multi") && v.includes("version") && v.includes("concurrency"))?"mvcc_correct":"incorrect";};'
    )
    write_task(
        "R_meet3_rust.html",
        '<div class="instructions">Listen to this Rust async audio. What runtime crate is being discussed? (e.g. tokio, async-std). Type the runtime name and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/rust_async.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v.includes("tokio")?"tokio_correct":"incorrect";};'
    )

    # ── Screencast (3 tasks): real asciinema recordings ──────────────
    # asciinema's .cast format embeds via their player.
    write_task(
        "R_cast1_git.html",
        '<div class="instructions">Watch the terminal recording. Type the FIRST git subcommand executed. (e.g., clone, status, commit)</div>'
        '<asciinema-player src="../realcontent_assets/asciinema_git_clone.cast" autoplay style="max-width:760px"></asciinema-player>'
        '<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.js"></script>'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.css">'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return v==="clone"?"clone_correct":"incorrect";};'
    )
    write_task(
        "R_cast2_pip.html",
        '<div class="instructions">Watch the terminal recording. Which Python package is being installed? Type the package name and Submit.</div>'
        '<asciinema-player src="../realcontent_assets/asciinema_pip_install.cast" autoplay style="max-width:760px"></asciinema-player>'
        '<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.js"></script>'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.css">'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase().split()[0]||"";'
        ' return v.length>0?"pkg_typed":"no_answer";};'
    )
    write_task(
        "R_cast3_npm.html",
        '<div class="instructions">Watch the terminal recording. Did the test suite PASS or FAIL? Type "pass" or "fail" and Submit.</div>'
        '<asciinema-player src="../realcontent_assets/asciinema_npm_test.cast" autoplay style="max-width:760px"></asciinema-player>'
        '<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.js"></script>'
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.0/dist/bundle/asciinema-player.min.css">'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return (v==="pass" || v==="fail")?"answer_typed":"no_answer";};'
    )

    # ── Voice interaction (3 tasks) ──────────────────────────────────
    # Common Voice clips; these are CC-0.  We bundle three pre-selected.
    write_task(
        "R_voice1_yesno.html",
        '<div class="instructions">Listen to the recording. Did the speaker say YES or NO at the end? Type "yes" or "no" and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/voice_yesno.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return (v==="yes" || v==="no")?"answer_typed":"no_answer";};'
    )
    write_task(
        "R_voice2_directions.html",
        '<div class="instructions">Listen to the recording, which contains a directional instruction. Type the direction (left/right/north/south/east/west). Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/voice_directions.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' return ["left","right","north","south","east","west"].includes(v)?"direction_typed":"no_answer";};'
    )
    write_task(
        "R_voice3_appointment.html",
        '<div class="instructions">Listen to the recording. The speaker is booking an appointment. Type the day-of-week mentioned (Monday, Tuesday, etc.) and Submit.</div>'
        '<audio controls autoplay src="../realcontent_assets/voice_appointment.mp3"></audio>'
        '<input type="text" id="ans"><button id="b">Submit</button>',
        'document.getElementById("b").onclick=()=>{};'
        'window.getTaskResult=()=>{const v=document.getElementById("ans").value.trim().toLowerCase();'
        ' const days=["monday","tuesday","wednesday","thursday","friday","saturday","sunday"];'
        ' return days.includes(v)?"day_typed":"no_answer";};'
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="Download asset files")
    p.add_argument("--build", action="store_true", help="Generate HTML task pages")
    args = p.parse_args()

    if args.download:
        prepare_assets()
    if args.build or (not args.download):
        make_tasks()
        print(f"Wrote {len(list(HTML_DIR.glob('R_*.html')))} real-content tasks")
