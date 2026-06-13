import { useMemo, useState } from 'react'
import { BASE, useJson, type TaskInfo, type Trajectory } from '../data'

// Paired recordings: same task, AOI success vs standard-mode failure
const PAIRED = ['A-M3', 'C-M2', 'F-M3', 'I-E1']
// AOI-only recordings (one per remaining category)
const AOI_ONLY = ['B-M1', 'D-E2', 'E-E2', 'G-M2', 'H-E1', 'J-E2']
// How many steps of each standard-mode failure the replay video shows
const STD_VIDEO_STEPS: Record<string, number> = { 'A-M3': 4, 'C-M2': 5, 'F-M3': 3, 'I-E1': 4 }

const BLURBS: Record<string, string> = {
  'A-M3': 'A news podcast reads out three headlines. The agent must transcribe all three — impossible without hearing the audio.',
  'B-M1': 'A meeting presenter speaks the value of a blank spreadsheet cell aloud; the agent fills it in.',
  'C-M2': 'A settings walkthrough video pages through three screens; the agent must name each page it saw.',
  'D-E2': 'A carousel auto-rotates through slides; the agent counts items shown on a slide that is no longer visible.',
  'E-E2': 'A live stock chart ticks in real time; the agent reports the peak price reached during the task.',
  'F-M3': 'Toast notifications appear and auto-dismiss; the agent counts how many fired while it watched.',
  'G-M2': 'An inbound voice call dictates a shipping address; the agent fills the four-field form from speech.',
  'H-E1': 'A voice interview asks a question out loud; the agent answers using the speak action.',
  'I-E1': 'A collaborator speaks an instruction in a shared editor; the agent types the dictated sentence.',
  'J-E2': 'A reaction game requires clicking targets that appear at unpredictable moments.',
}

function OpHistory({ traj, shownSteps }: { traj?: Trajectory; shownSteps?: number }) {
  if (!traj) return null
  // compress runs of consecutive identical actions (e.g. the standard agent's wait() loops)
  const groups: { action: string; from: number; to: number; kf: number; narration: string }[] = []
  for (const s of traj.steps) {
    const last = groups[groups.length - 1]
    // start a new group at the replay cutoff so the "not in replay" marker is exact
    const crossesCutoff = shownSteps !== undefined && s.step === shownSteps + 1
    if (last && last.action === s.action && !crossesCutoff) {
      last.to = s.step
      last.kf += s.n_keyframes
    } else {
      groups.push({ action: s.action, from: s.step, to: s.step, kf: s.n_keyframes, narration: s.narration })
    }
  }
  return (
    <div className="ops">
      <div className="ops-title">Operation sequence · {traj.steps_taken} steps</div>
      {groups.map((g) => {
        const offVideo = shownSteps !== undefined && g.from > shownSteps
        return (
          <div className={`op-row ${offVideo ? 'dim' : ''}`} key={g.from} title={g.narration || undefined}>
            <span className="op-num">{g.from === g.to ? g.from : `${g.from}–${g.to}`}</span>
            <span className="op-act">
              {g.action}
              {g.to > g.from && <span className="op-rep"> ×{g.to - g.from + 1}</span>}
            </span>
            {g.kf > 0 && <span className="op-kf">🎞{g.kf}</span>}
            {offVideo && <span className="op-off">not in replay</span>}
          </div>
        )
      })}
    </div>
  )
}

function Video({ name, mode, badge }: { name: string; mode: 'aoi' | 'standard'; badge: string }) {
  return (
    <div style={{ position: 'relative' }}>
      <video
        controls
        preload="none"
        poster={`${BASE}videos/${name}.jpg`}
        src={`${BASE}videos/${name}.mp4`}
      />
      <span
        className={`chip ${mode === 'aoi' ? 'aoi' : 'std'}`}
        style={{ position: 'absolute', top: 10, left: 10, boxShadow: '0 1px 6px rgba(0,0,0,0.25)' }}
      >
        {badge}
      </span>
    </div>
  )
}

export function Recordings({ tasks }: { tasks: TaskInfo[] | null }) {
  const [tab, setTab] = useState<'pairs' | 'gallery'>('pairs')
  const byId = new Map((tasks ?? []).map((t) => [t.task_id, t]))
  const aoiTraj = useJson<Trajectory[]>('data/traj_claude_aoi.json')
  const stdTraj = useJson<Trajectory[]>('data/traj_claude_standard.json')
  const aoiById = useMemo(() => new Map((aoiTraj ?? []).map((t) => [t.task_id, t])), [aoiTraj])
  const stdById = useMemo(() => new Map((stdTraj ?? []).map((t) => [t.task_id, t])), [stdTraj])

  return (
    <section className="block" id="recordings">
      <div className="wrap">
        <div className="sec-kicker">See it run</div>
        <h2 className="sec-title">Agent trajectory recordings</h2>
        <p className="sec-sub">
          Deterministic replays of logged Claude Sonnet 4.6 trajectories on the original DynaCU-Bench task
          pages, recorded in the real browser environment — <b>with sound</b>. The soundtrack is the page’s
          speech reconstructed through the same TTS pipeline the evaluation harness played to the agent; the
          overlay shows the agent’s observation phase, executed actions, AOI narration, and audio captions
          (🔊). Outcomes are the original evaluation verdicts.
        </p>

        <div className="seg" style={{ marginBottom: 26 }}>
          <button className={tab === 'pairs' ? 'on' : ''} onClick={() => setTab('pairs')}>
            ⚔ AOI vs. Standard — same task
          </button>
          <button className={tab === 'gallery' ? 'on' : ''} onClick={() => setTab('gallery')}>
            ▶ AOI across all 10 categories
          </button>
        </div>

        {tab === 'pairs' &&
          PAIRED.map((tid) => {
            const t = byId.get(tid)
            return (
              <div key={tid}>
                <div className="traj-head" style={{ marginBottom: 10 }}>
                  <h3>
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent-deep)' }}>{tid}</span>
                    {' · '}
                    {t ? t.category.replace('_', ' — ') : ''}
                  </h3>
                  {t && <span className={`diff ${t.difficulty}`}>{t.difficulty}</span>}
                </div>
                <p style={{ fontSize: 14, color: 'var(--ink-soft)', margin: '0 0 12px' }}>
                  {BLURBS[tid]} {t && <em>Instruction: “{t.instruction}”</em>}
                </p>
                <div className="pair-row">
                  <div className="card video-card">
                    <Video name={`${tid}_standard`} mode="standard" badge="Standard · screenshot-only" />
                    <div className="video-meta">
                      <div className="title">
                        Standard loop <span className="chip fail">✗ failed</span>
                      </div>
                      <div className="desc">
                        Deaf and blind between screenshots — the agent waits, sees nothing change, and gives up
                        or guesses. The audio you hear is exactly what the agent couldn’t.
                      </div>
                    </div>
                    <OpHistory traj={stdById.get(tid)} shownSteps={STD_VIDEO_STEPS[tid]} />
                  </div>
                  <div className="card video-card">
                    <Video name={`${tid}_aoi`} mode="aoi" badge="AOI full" />
                    <div className="video-meta">
                      <div className="title">
                        Same model + AOI <span className="chip pass">✓ passed</span>
                      </div>
                      <div className="desc">
                        Inter-step keyframes, ASR transcript, and narration give the model the missing
                        context — same model, zero retraining.
                      </div>
                    </div>
                    <OpHistory traj={aoiById.get(tid)} />
                  </div>
                </div>
              </div>
            )
          })}

        {tab === 'gallery' && (
          <div className="video-grid">
            {[...PAIRED, ...AOI_ONLY]
              .sort()
              .map((tid) => {
                const t = byId.get(tid)
                return (
                  <div className="card video-card" key={tid}>
                    <Video name={`${tid}_aoi`} mode="aoi" badge="AOI full" />
                    <div className="video-meta">
                      <div className="title">
                        <span style={{ fontFamily: 'var(--font-mono)' }}>{tid}</span>
                        {t && <span className="chip cat">{t.category.replace('_', ' ')}</span>}
                        <span className="chip pass">✓</span>
                      </div>
                      <div className="desc">{BLURBS[tid]}</div>
                    </div>
                    <OpHistory traj={aoiById.get(tid)} />
                  </div>
                )
              })}
          </div>
        )}

        <p className="note">
          Replays execute the exact logged action sequence with approximately the original step timing.
          The audio track reconstructs what the agent heard: the benchmark pages declare speech via
          speechSynthesis, which the evaluation harness rendered with edge-tts into a virtual speaker —
          the same engine and voice are used here, at the replayed utterance timestamps. The agent’s own
          speak() output is voiced distinctly. Captions duplicate the audio for accessibility.
        </p>
      </div>
    </section>
  )
}
