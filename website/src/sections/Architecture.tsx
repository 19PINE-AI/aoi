export function Architecture() {
  return (
    <section className="block" id="architecture">
      <div className="wrap">
        <div className="sec-kicker">System Design</div>
        <h2 className="sec-title">A perception layer between the environment and any CU model</h2>
        <p className="sec-sub">
          The AOI observes the <em>entire interval between agent steps</em> and converts continuous screen
          and audio streams into the sparse images and text every CU model already accepts. Each component
          sits behind a sub-millisecond gate: on static, silent content the AOI produces nothing and the
          loop is identical to the standard one.
        </p>
        <div className="card card-pad">
          <div className="arch">
            <div className="arch-col">
              <div className="arch-box env">
                <span className="tag">Environment</span>
                <b>🖥 Screen stream</b>
                Continuous capture at ~3 Hz between agent steps
              </div>
              <div className="arch-box env">
                <span className="tag">Environment</span>
                <b>🔊 Audio stream</b>
                System audio: speech, notifications, media
              </div>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-col">
              <div className="arch-box aoi">
                <span className="tag">AOI · gated</span>
                <b>1 · Inter-step keyframe capture</b>
                Pixel gate (&lt;1 ms) → CLIP-ViT-B/16 semantic distance (~7 ms). Emits 0–5 keyframes per
                step; loading spinners and blinking cursors are suppressed.
              </div>
              <div className="arch-box aoi">
                <span className="tag">AOI · gated</span>
                <b>2 · Volume-gated audio observer</b>
                RMS energy gate → Whisper large-v3 ASR. Produces a transcript only when audio is actually
                present.
              </div>
              <div className="arch-box aoi">
                <span className="tag">AOI · gated</span>
                <b>3 · Visual narration context</b>
                The CU model itself emits a one-line description of new visual information at each step;
                narrations persist as text after keyframe images are pruned from context.
              </div>
            </div>
            <div className="arch-arrow">→</div>
            <div className="arch-col">
              <div className="arch-box model">
                <span className="tag">Unchanged</span>
                <b>🤖 Any CU model</b>
                Claude, GPT, Gemini, Grok, EvoCUA, Fara, Qwen3-VL… standard image + text input, zero
                retraining.
              </div>
              <div className="arch-box model">
                <span className="tag">Output</span>
                <b>⚡ Action</b>
                click · type · fill · scroll · speak — the discrete action loop is untouched.
              </div>
            </div>
          </div>
          <p className="note">
            Observation record at step N = current screenshot + inter-step keyframes + audio transcript
            (two-layer) + accumulated narration history. The interface is a separable design axis: on
            Gemini 3 the keyframe stream flips negative while audio and scaffold stay positive, so
            component selection must become per-model rather than a fixed bundle.
          </p>
        </div>

        <div className="card card-pad" style={{ marginTop: 22 }}>
          <h3>Same task, same model — different observation interface</h3>
          <div className="sub">
            Task B-E1 (meeting): <em>“Read the meeting slides and listen to the discussion. What is the new
            launch date?”</em> · Claude Sonnet 4.6, logged trajectories from the headline evaluation run
          </div>
          <div className="tc-grid">
            <div className="tc-col std">
              <div className="tc-head">
                <span className="chip std">Standard · screenshot-only</span>
                <span className="chip fail">✗ failed</span>
                <span className="tc-meta">15 steps · 51.2 s</span>
              </div>
              <ol className="tc-steps">
                <li><code>wait()</code><span className="tc-note">sees the static “Meeting Agenda” slide</span></li>
                <li><code>wait()</code><span className="tc-note">same slide — nothing appears to change</span></li>
                <li><code>wait()</code><span className="tc-note">same slide</span></li>
                <li className="tc-ellipsis">⋮ &nbsp;wait() ×11 more</li>
                <li><code>wait()</code><span className="tc-note">“No visual change — the screen still shows Slide 3 of 3”</span></li>
              </ol>
              <p className="tc-verdict fail">
                The launch date was only ever <b>spoken</b>. The agent is deaf, sees only slide stills,
                waits out all 15 steps, and never fills the field.
              </p>
            </div>
            <div className="tc-col aoi">
              <div className="tc-head">
                <span className="chip aoi">AOI full</span>
                <span className="chip pass">✓ passed</span>
                <span className="tc-meta">2 steps · 26.2 s</span>
              </div>
              <ol className="tc-steps">
                <li>
                  <code>wait()</code>
                  <span className="tc-audio">🔊 “Here’s the team. As you can see we have strong cross-functional coverage.”</span>
                  <span className="tc-narr">📝 “The screen shows a Google Meet ‘Product Launch Planning’ meeting…”</span>
                </li>
                <li>
                  <code>fill(#launchDate, "April 28th")</code>
                  <span className="tc-audio">🔊 “And I want to mention, we’ve moved the launch date to April 28th. Please update your calendars.”</span>
                  <span className="tc-narr">📝 “The meeting slide shows the Project Team with 6 members…”</span>
                </li>
              </ol>
              <p className="tc-verdict pass">
                The ASR transcript delivers the spoken date the moment it is uttered; the agent fills the
                form on the very next step.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
