import { useMemo, useState } from 'react'
import { BASE, useJson, type RunInfo, type TaskInfo, type Trajectory } from '../data'

/** Reconstruct the observation record sent to the CU model at step `idx`
 *  (1-based), in the agent loop's ObservationRecord.to_prompt_text() format.
 *  Narrations are generated as a side-output of step N and therefore appear
 *  as context only from step N+1 onward. */
function buildLlmContext(traj: Trajectory, instruction: string, idx: number, standard: boolean): string {
  const lines: string[] = [`=== Step ${idx} Observation ===`, '']
  const prior = traj.steps.slice(0, idx - 1)
  const cur = traj.steps[idx - 1]
  if (standard) {
    if (prior.length) {
      lines.push('[CONTEXT — prior actions]')
      for (const p of prior) lines.push(`  Step ${p.step}: ${p.action}`)
      lines.push('')
    }
    lines.push('[CURRENT SCREENSHOT — what the screen looks like right now]')
    lines.push('')
    lines.push(`[TASK] ${instruction}`)
    return lines.join('\n')
  }
  if (prior.length) {
    lines.push('[CONTEXT — prior steps]')
    for (const p of prior) {
      lines.push(`  Step ${p.step}:`)
      if (p.audio_text) lines.push(`    AUDIO: ${p.audio_text}`)
      if (p.narration) lines.push(`    VISUAL: ${p.narration}`)
      lines.push(`    ACTION: ${p.action}`)
      lines.push('')
    }
  }
  lines.push('[NEW — current interval]')
  if (cur?.audio_text) lines.push(`  [AUDIO — recent] ${cur.audio_text}`)
  if (cur && cur.n_keyframes > 0) {
    lines.push(`  [KEYFRAMES — ${cur.n_keyframes} visual change(s) detected]`)
    for (let i = 1; i <= cur.n_keyframes; i++) lines.push(`    Image ${i}: (keyframe image)`)
  }
  lines.push('  [CURRENT SCREENSHOT — what the screen looks like right now]')
  lines.push('')
  lines.push(`[TASK] ${instruction}`)
  return lines.join('\n')
}

export function Trajectories({ runs, tasks }: { runs: RunInfo[] | null; tasks: TaskInfo[] | null }) {
  const [runId, setRunId] = useState('claude_aoi')
  const [taskId, setTaskId] = useState('A-M3')
  const [filter, setFilter] = useState<'all' | 'pass' | 'fail'>('all')
  const traj = useJson<Trajectory[]>(`data/traj_${runId}.json`)
  const byId = useMemo(() => new Map((tasks ?? []).map((t) => [t.task_id, t])), [tasks])

  const list = (traj ?? []).filter((t) =>
    filter === 'all' ? true : filter === 'pass' ? t.success : !t.success)
  const sel = (traj ?? []).find((t) => t.task_id === taskId) ?? list[0]
  const selTask = sel ? byId.get(sel.task_id) : undefined

  return (
    <section className="block" id="trajectories">
      <div className="wrap">
        <div className="sec-kicker">Step by step</div>
        <h2 className="sec-title">Trajectory explorer</h2>
        <p className="sec-sub">
          Browse every logged trajectory from the headline evaluation runs: each step shows the executed
          action, the AOI’s <span style={{ color: '#34607a' }}>visual narration</span>, the captured{' '}
          <span style={{ color: '#6d5878' }}>audio transcript</span>, and how many inter-step keyframes
          were attached.
        </p>

        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18, alignItems: 'center' }}>
          <div className="seg">
            {(runs ?? []).map((r) => (
              <button key={r.id} className={r.id === runId ? 'on' : ''} onClick={() => setRunId(r.id)}>
                {r.label} <span style={{ color: 'var(--ink-faint)', fontWeight: 500 }}>({r.pass}/{r.total})</span>
              </button>
            ))}
          </div>
        </div>

        <div className="traj-layout">
          <div className="card">
            <div style={{ padding: '12px 14px', borderBottom: '1px solid var(--line)', display: 'flex', gap: 8, alignItems: 'center' }}>
              <b style={{ fontSize: 14 }}>Tasks</b>
              <div className="seg" style={{ marginLeft: 'auto' }}>
                {(['all', 'pass', 'fail'] as const).map((f) => (
                  <button key={f} className={filter === f ? 'on' : ''} onClick={() => setFilter(f)}>
                    {f}
                  </button>
                ))}
              </div>
            </div>
            <div className="traj-list">
              {list.map((t) => (
                <button
                  key={t.task_id}
                  className={`traj-item ${sel?.task_id === t.task_id ? 'on' : ''}`}
                  onClick={() => setTaskId(t.task_id)}
                >
                  <span className={`dot ${t.success ? 'pass' : 'fail'}`} />
                  <span className="tid">{t.task_id}</span>
                  <span className="cat">{t.category.replace('_', ' ')}</span>
                  <span style={{ color: 'var(--ink-faint)', fontSize: 12 }}>{t.steps_taken} steps</span>
                </button>
              ))}
            </div>
          </div>

          <div className="card card-pad">
            {sel ? (
              <>
                <div className="traj-head">
                  <h3 style={{ fontFamily: 'var(--font-mono)' }}>{sel.task_id}</h3>
                  <span className={`chip ${sel.success ? 'pass' : 'fail'}`}>
                    {sel.success ? '✓ passed' : '✗ failed'}
                  </span>
                  <span className={`diff ${sel.difficulty}`}>{sel.difficulty}</span>
                  <span className="chip cat">{sel.category.replace('_', ' ')}</span>
                  <span style={{ color: 'var(--ink-faint)', fontSize: 13 }}>
                    {sel.steps_taken} steps · {sel.total_time_s}s
                  </span>
                  {selTask && (
                    <a
                      className="chip cat"
                      style={{ textDecoration: 'none', cursor: 'pointer' }}
                      href={`${BASE}tasks/html_tasks/${selTask.html_file}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      ↗ open task page
                    </a>
                  )}
                </div>
                {selTask && (
                  <p style={{ fontSize: 14, color: 'var(--ink-soft)', marginBottom: 16 }}>
                    <b>Instruction:</b> {selTask.instruction}
                  </p>
                )}
                <div style={{ maxHeight: 560, overflowY: 'auto', paddingRight: 4 }}>
                  {sel.steps.map((s) => (
                    <div className="step-card" key={s.step}>
                      <div className="step-head">
                        <span className="step-num">Step {s.step}</span>
                        {s.n_keyframes > 0 && (
                          <span className="chip kf">🎞 {s.n_keyframes} keyframe{s.n_keyframes > 1 ? 's' : ''}</span>
                        )}
                      </div>
                      <div className="action">{s.action || '(no action)'}</div>
                      {s.narration && <div className="narr">📝 {s.narration}</div>}
                      {s.audio_text && <div className="audio">🔊 {s.audio_text}</div>}
                      <details className="ctx">
                        <summary>LLM context at this step</summary>
                        <pre>{buildLlmContext(sel, selTask?.instruction ?? '', s.step,
                          runId.includes('standard'))}</pre>
                        <div className="ctx-note">
                          Reconstructed from the logged trajectory in the agent loop’s observation-record
                          format; images are shown as placeholders.
                        </div>
                      </details>
                    </div>
                  ))}
                  {sel.error && (
                    <div className="step-card" style={{ borderColor: '#e5c2c6' }}>
                      <div className="step-head"><span className="step-num" style={{ color: 'var(--red)' }}>error</span></div>
                      <div style={{ fontSize: 13.5, color: 'var(--red)' }}>{sel.error}</div>
                    </div>
                  )}
                </div>
              </>
            ) : (
              <p style={{ color: 'var(--ink-faint)' }}>Loading trajectories…</p>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
