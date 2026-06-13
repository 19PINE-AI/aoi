#!/usr/bin/env python3
"""Replay recorded agent trajectories on the benchmark task pages and record
videos of them with Playwright.

For each selected (task, run) pair the script loads the task HTML, overlays a
HUD showing the agent's step / action / narration / audio captions, executes
the recorded actions with approximately the original step timing, and saves a
.webm video (converted to .mp4 + poster .jpg by ffmpeg afterwards).
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402

HTML_TASKS_DIR = ROOT / "benchmark_env" / "html_tasks"
VIDEO_DIR = ROOT / "website" / "public" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720

# (task_id, run_file, mode_label, max_steps or None)
RECORDINGS = [
    # AOI-full successes, one per category
    ("A-M3", "v9_full_100_claude_aoi.json", "aoi", None),
    ("B-M1", "v9_full_100_claude_aoi.json", "aoi", None),
    ("C-M2", "v9_full_100_claude_aoi.json", "aoi", None),
    ("D-E2", "v9_full_100_claude_aoi.json", "aoi", None),
    ("E-E2", "v9_full_100_claude_aoi.json", "aoi", None),
    ("F-M3", "v9_full_100_claude_aoi.json", "aoi", None),
    ("G-M2", "v9_full_100_claude_aoi.json", "aoi", None),
    ("H-E1", "v9_full_100_claude_aoi.json", "aoi", None),
    ("I-E1", "v9_full_100_claude_aoi.json", "aoi", None),
    ("J-E2", "v9_full_100_claude_aoi.json", "aoi", None),
    # Standard-mode failures on the same tasks, for side-by-side comparison
    ("A-M3", "v9_full_100_claude_standard.json", "standard", 4),
    ("C-M2", "v9_full_100_claude_standard.json", "standard", 5),
    ("F-M3", "v9_full_100_claude_standard.json", "standard", 3),
    ("I-E1", "v9_full_100_claude_standard.json", "standard", 4),
]

SPEECH_PATCH = """
window.__capturedUtterances = [];
const _origSpeak = speechSynthesis.speak.bind(speechSynthesis);
speechSynthesis.speak = function(utterance) {
    if (utterance && utterance.text) {
        window.__capturedUtterances.push({
            text: utterance.text,
            rate: utterance.rate || 1.0,
            t: performance.now(),
        });
    }
    if (utterance && utterance.onend) {
        const words = utterance.text ? utterance.text.split(/\\s+/).length : 0;
        const rate = utterance.rate || 1.0;
        const ms = Math.max(100, (words / (3.0 * rate)) * 1000);
        setTimeout(() => utterance.onend(new Event('end')), ms);
    }
};
"""

# The eval harness rendered page speechSynthesis text with edge-tts GuyNeural
# (browser_eval.py: page JS -> extract text -> edge-tts -> virtual_speaker), so
# the same voice here reproduces exactly the audio the agent heard.
PAGE_VOICE = "en-US-GuyNeural"
AGENT_VOICE = "en-US-AriaNeural"  # the agent's own speak() output, distinct voice

HUD_JS = """
() => {
  if (document.getElementById('__aoi_hud')) return;
  const css = `
    #__aoi_hud { position: fixed; left: 0; right: 0; bottom: 0; z-index: 2147483647;
      background: rgba(10,14,26,0.92); color: #e8ecf4; padding: 14px 22px 16px;
      font-family: 'Segoe UI', system-ui, sans-serif; font-size: 15px;
      border-top: 2px solid #5E81AC; backdrop-filter: blur(4px); }
    #__aoi_hud .row1 { display: flex; gap: 10px; align-items: center; margin-bottom: 7px; }
    #__aoi_hud .chip { font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em;
      padding: 3px 10px; border-radius: 999px; text-transform: uppercase; }
    #__aoi_hud .chip.step { background: #2e3650; color: #cfd8ee; }
    #__aoi_hud .chip.mode-aoi { background: #5E81AC; color: #fff; }
    #__aoi_hud .chip.mode-standard { background: #BF616A; color: #fff; }
    #__aoi_hud .chip.kf { background: #243524; color: #A3BE8C; }
    #__aoi_hud .phase { color: #8a94ad; font-size: 13px; font-style: italic; }
    #__aoi_hud .action { font-family: ui-monospace, Menlo, monospace; font-size: 14.5px;
      color: #EBCB8B; min-height: 20px; }
    #__aoi_hud .narration { color: #88C0D0; font-size: 13.5px; font-style: italic;
      margin-top: 5px; line-height: 1.45; max-height: 60px; overflow: hidden; }
    #__aoi_caption { position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
      z-index: 2147483647; max-width: 76%; background: rgba(10,14,26,0.88); color: #fff;
      padding: 9px 18px; border-radius: 10px; font-family: 'Segoe UI', sans-serif;
      font-size: 15px; line-height: 1.4; display: none; border-left: 3px solid #B48EAD; }
    #__aoi_end { position: fixed; inset: 0; z-index: 2147483647; display: none;
      align-items: center; justify-content: center; background: rgba(8,10,20,0.82); }
    #__aoi_end .card { text-align: center; font-family: 'Segoe UI', sans-serif; }
    #__aoi_end .mark { font-size: 84px; font-weight: 800; }
    #__aoi_end .msg { font-size: 26px; color: #e8ecf4; margin-top: 10px; font-weight: 600; }
    #__aoi_end .sub { font-size: 15px; color: #9aa3bd; margin-top: 8px; }`;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
  const hud = document.createElement('div');
  hud.id = '__aoi_hud';
  hud.innerHTML = `<div class="row1">
      <span class="chip step" id="__hud_step"></span>
      <span class="chip" id="__hud_mode"></span>
      <span class="chip kf" id="__hud_kf" style="display:none"></span>
      <span class="phase" id="__hud_phase"></span></div>
    <div class="action" id="__hud_action"></div>
    <div class="narration" id="__hud_narr" style="display:none"></div>`;
  document.body.appendChild(hud);
  const cap = document.createElement('div');
  cap.id = '__aoi_caption';
  document.body.appendChild(cap);
  const end = document.createElement('div');
  end.id = '__aoi_end';
  end.innerHTML = `<div class="card"><div class="mark" id="__end_mark"></div>
    <div class="msg" id="__end_msg"></div><div class="sub" id="__end_sub"></div></div>`;
  document.body.appendChild(end);
  window.__hud = (s) => {
    const el = (id) => document.getElementById(id);
    if (s.step) el('__hud_step').textContent = s.step;
    if (s.mode) { el('__hud_mode').textContent = s.modeLabel;
      el('__hud_mode').className = 'chip mode-' + s.mode; }
    if (s.phase !== undefined) el('__hud_phase').textContent = s.phase;
    if (s.action !== undefined) el('__hud_action').textContent = s.action;
    if (s.kf !== undefined) {
      el('__hud_kf').style.display = s.kf ? '' : 'none';
      el('__hud_kf').textContent = s.kf; }
    if (s.narration !== undefined) {
      el('__hud_narr').style.display = s.narration ? '' : 'none';
      el('__hud_narr').textContent = s.narration ? '\\ud83d\\udcdd narration: ' + s.narration : ''; }
    if (s.caption !== undefined) {
      const c = el('__aoi_caption');
      c.style.display = s.caption ? 'block' : 'none';
      c.innerHTML = s.caption; }
    if (s.end) {
      el('__end_mark').textContent = s.end.ok ? '\\u2713' : '\\u2717';
      el('__end_mark').style.color = s.end.ok ? '#A3BE8C' : '#BF616A';
      el('__end_msg').textContent = s.end.msg;
      el('__end_sub').textContent = s.end.sub || '';
      el('__aoi_end').style.display = 'flex'; }
  };
}
"""


def hud(page, **state):
    page.evaluate("(s) => window.__hud(s)", state)


def poll_caption(page, last_n):
    """Show the most recent captured speech utterance as a caption."""
    utt = page.evaluate("() => window.__capturedUtterances || []")
    if len(utt) > last_n:
        text = utt[-1]["text"]
        hud(page, caption="\U0001f50a " + (text[:220] + ("…" if len(text) > 220 else "")))
    return len(utt)


def execute(page, action):
    """Execute a recorded action string on the page (replay subset)."""
    a = action.strip()
    m = re.match(r'fill\(\s*([#\.\w\[\]="\'-]+)\s*,\s*"(.*)"\s*\)$', a, re.S)
    if m:
        sel, text = m.group(1), m.group(2)
        page.click(sel, timeout=3000)
        page.fill(sel, text, timeout=3000)
        return True
    m = re.match(r"click\((\d+),\s*(\d+)\)$", a)
    if m:
        page.mouse.click(int(m.group(1)), int(m.group(2)))
        return True
    m = re.match(r'type\("(.*)"\)$', a, re.S)
    if m:
        page.keyboard.type(m.group(1), delay=28)
        return True
    m = re.match(r"scroll\((\w+),?\s*(\d+)?\)$", a)
    if m:
        amount = int(m.group(2) or 400)
        dy = amount if m.group(1) == "down" else -amount
        page.mouse.wheel(0, dy)
        return True
    if a.startswith("speak(") or a in ("wait()", "done"):
        return True  # displayed in HUD only
    m = re.match(r'press\("?(\w+)"?\)$', a)
    if m:
        page.keyboard.press(m.group(1))
        return True
    print(f"    !! unhandled action: {a!r}")
    return False


def step_wait(page, seconds, utt_count):
    """Wait while polling speech captions twice a second."""
    endt = time.time() + seconds
    while time.time() < endt:
        utt_count = poll_caption(page, utt_count)
        time.sleep(0.4)
    return utt_count


def record(task_rec, mode, max_steps, html_file, out_name):
    steps = task_rec["steps"][:max_steps] if max_steps else task_rec["steps"]
    truncated = max_steps is not None and task_rec["steps_taken"] > max_steps
    mode_label = "AOI full" if mode == "aoi" else "Standard (screenshot-only)"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=[
            "--no-sandbox", "--disable-dev-shm-usage",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-web-security", "--allow-file-access-from-files",
            "--mute-audio",
        ])
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(VIDEO_DIR / "_tmp"),
            record_video_size={"width": W, "height": H},
        )
        page = ctx.new_page()
        page.add_init_script(SPEECH_PATCH)
        t0 = time.time()  # ≈ video start: recording begins when the page is created
        page.goto(f"file://{(HTML_TASKS_DIR / html_file).absolute()}")
        t_nav = time.time()  # ≈ performance.now() epoch of the page
        page.wait_for_timeout(400)
        page.evaluate(HUD_JS)
        hud(page, step="loading", mode=mode, modeLabel=mode_label,
            phase="task page loading…", action="")
        page.wait_for_timeout(1800)

        audio_events = []  # {offset_s, text, rate, voice} relative to video start
        utt = 0
        n = len(steps)
        for i, s in enumerate(steps, 1):
            obs_s = ((s.get("obs_overhead_ms") or 0) + (s.get("model_latency_ms") or 0)) / 1000.0
            obs_s = max(2.5, min(obs_s, 9.0))
            hud(page, step=f"step {i}/{task_rec['steps_taken']}", phase="observing…",
                action="", narration="",
                kf=(f"{s.get('n_keyframes', 0)} keyframes captured"
                    if mode == "aoi" and s.get("n_keyframes") else ""))
            utt = step_wait(page, obs_s, utt)

            action = s.get("action") or ""
            shown = action
            if action.startswith("speak("):
                shown = "\U0001f5e3 " + action
            hud(page, phase="acting", action=f"▶ {shown}",
                narration=(s.get("narration") or "") if mode == "aoi" else "")
            page.wait_for_timeout(1300)
            m = re.match(r'speak\("(.*)"\)$', action, re.S)
            if m:
                audio_events.append({"offset_s": time.time() - t0, "text": m.group(1),
                                     "rate": 1.0, "voice": AGENT_VOICE})
            try:
                execute(page, action)
            except Exception as e:
                print(f"    !! action failed: {action!r}: {e}")
            page.wait_for_timeout(1500)

        ok = task_rec["success"]
        sub = ""
        if truncated:
            sub = f"replay truncated — agent gave up after {task_rec['steps_taken']} steps"
        hud(page, caption="", end={
            "ok": ok,
            "msg": "Task passed" if ok else "Task failed",
            "sub": sub,
        })
        page.wait_for_timeout(2600)

        # All page speech, with in-page performance.now() timestamps
        for u in page.evaluate("() => window.__capturedUtterances || []"):
            audio_events.append({"offset_s": (t_nav - t0) + u["t"] / 1000.0,
                                 "text": u["text"], "rate": u.get("rate", 1.0),
                                 "voice": PAGE_VOICE})
        audio_events.sort(key=lambda e: e["offset_s"])

        video = page.video
        page.close()
        ctx.close()
        browser.close()
        tmp_path = Path(video.path())
        final = VIDEO_DIR / f"{out_name}.webm"
        tmp_path.rename(final)
        return final, audio_events


_TTS_CACHE = {}


def synth_tts(text, voice, rate, tmpdir):
    """Synthesize text with edge-tts, cached on (text, voice, rate)."""
    key = (text, voice, round(rate, 2))
    if key in _TTS_CACHE:
        return _TTS_CACHE[key]
    out = tmpdir / f"tts_{len(_TTS_CACHE):03d}.mp3"
    rate_pct = f"{round((rate - 1) * 100):+d}%"
    subprocess.run([sys.executable, "-m", "edge_tts", "--voice", voice,
                    f"--rate={rate_pct}", "--text", text,
                    "--write-media", str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _TTS_CACHE[key] = out
    return out


def mux_audio(mp4, events, tmpdir):
    """Mix the synthesized utterances into the mp4 at their recorded offsets."""
    clips = [(e["offset_s"], synth_tts(e["text"], e["voice"], e["rate"], tmpdir))
             for e in events]
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip())
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4)]
    for _, p in clips:
        cmd += ["-i", str(p)]
    filt, labels = [], []
    for i, (off, _) in enumerate(clips, 1):
        ms = max(0, int(off * 1000))
        filt.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
        labels.append(f"[a{i}]")
    filt.append("".join(labels) + f"amix=inputs={len(clips)}:normalize=0[aout]")
    tmp_out = mp4.with_suffix(".mux.mp4")
    cmd += ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-t", f"{dur:.3f}",
            "-movflags", "+faststart", str(tmp_out)]
    subprocess.run(cmd, check=True)
    tmp_out.replace(mp4)


def main():
    runs = {}
    catalog = json.loads((ROOT / "website" / "public" / "data" / "tasks.json").read_text())
    all_events = {}
    for task_id, run_file, mode, max_steps in RECORDINGS:
        if run_file not in runs:
            runs[run_file] = {r["task_id"]: r for r in
                              json.loads((ROOT / "results" / run_file).read_text())}
        rec = runs[run_file][task_id]
        html_file = next(t["html_file"] for t in catalog if t["task_id"] == task_id)
        out_name = f"{task_id}_{mode}"
        print(f"recording {out_name} ({html_file}, {len(rec['steps'])} steps)…")
        t0 = time.time()
        path, events = record(rec, mode, max_steps, html_file, out_name)
        all_events[out_name] = events
        print(f"  -> {path.name} in {time.time()-t0:.0f}s, {len(events)} audio events")

    # convert to mp4 + posters, then mux in the reconstructed audio track
    tts_dir = VIDEO_DIR / "_tmp"
    for webm in sorted(VIDEO_DIR.glob("*.webm")):
        mp4 = webm.with_suffix(".mp4")
        poster = webm.with_suffix(".jpg")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm),
                        "-c:v", "libx264", "-crf", "27", "-preset", "fast",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "3", "-i", str(webm),
                        "-frames:v", "1", "-q:v", "4", str(poster)], check=True)
        webm.unlink()
        events = all_events.get(webm.stem, [])
        if events:
            mux_audio(mp4, events, tts_dir)
        print(f"converted {mp4.name} ({len(events)} utterances muxed)")
    import shutil
    shutil.rmtree(VIDEO_DIR / "_tmp", ignore_errors=True)


if __name__ == "__main__":
    main()
