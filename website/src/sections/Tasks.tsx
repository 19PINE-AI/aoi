import { useMemo, useState } from 'react'
import { BASE, type TaskInfo } from '../data'

const CAT_LABEL: Record<string, string> = {
  A_podcast: 'A · Podcast',
  B_meeting: 'B · Meeting',
  C_video: 'C · Video',
  D_carousel: 'D · Carousel',
  E_dashboard: 'E · Dashboard',
  F_transient: 'F · Transient',
  G_phone: 'G · Phone',
  H_interview: 'H · Interview',
  I_collab: 'I · Collab',
  J_game: 'J · Game',
  S_static: 'S · Static',
}

export function Tasks({ tasks }: { tasks: TaskInfo[] | null }) {
  const [cat, setCat] = useState<string>('A_podcast')
  const [open, setOpen] = useState<TaskInfo | null>(null)
  const cats = useMemo(
    () => [...new Set((tasks ?? []).map((t) => t.category))],
    [tasks],
  )
  const list = (tasks ?? []).filter((t) => t.category === cat)

  return (
    <section className="block" id="benchmark">
      <div className="wrap">
        <div className="sec-kicker">DynaCU-Bench</div>
        <h2 className="sec-title">Play the benchmark tasks yourself</h2>
        <p className="sec-sub">
          Every DynaCU-Bench task is a self-contained dynamic web page — podcasts that speak, meetings that
          present, carousels that rotate, notifications that vanish. Click any task to run it live in your
          browser (enable sound: most tasks use speech synthesis), and see whether <em>you</em> can beat
          the screenshot-only agent.
        </p>

        <div className="seg" style={{ marginBottom: 20 }}>
          {cats.map((c) => (
            <button key={c} className={c === cat ? 'on' : ''} onClick={() => setCat(c)}>
              {CAT_LABEL[c] ?? c}
            </button>
          ))}
        </div>

        <div className="task-grid">
          {list.map((t) => (
            <button className="task-card" key={t.task_id} onClick={() => setOpen(t)}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="tid">{t.task_id}</span>
                <span className={`diff ${t.difficulty}`}>{t.difficulty}</span>
              </div>
              <div className="inst">{t.instruction}</div>
              <div className="meta">
                {t.axes.map((a) => (
                  <span key={a} className="chip cat" style={{ fontSize: 10.5 }}>
                    {a === 'audio' ? '🔊 audio' : a === 'visual' ? '👁 visual-temporal' : '⚡ real-time'}
                  </span>
                ))}
                {t.requires_audio_out && <span className="chip kf" style={{ fontSize: 10.5 }}>🗣 speak</span>}
              </div>
            </button>
          ))}
        </div>

        {open && (
          <div className="modal-back" onClick={() => setOpen(null)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-head">
                <span className="tid" style={{ fontFamily: 'var(--font-mono)', fontWeight: 750, color: 'var(--accent-deep)' }}>
                  {open.task_id}
                </span>
                <span className={`diff ${open.difficulty}`}>{open.difficulty}</span>
                <span className="chip cat">{open.eval_type} eval · {open.duration_s}s budget</span>
                <a
                  className="chip cat"
                  style={{ textDecoration: 'none' }}
                  href={`${BASE}tasks/html_tasks/${open.html_file}`}
                  target="_blank"
                  rel="noreferrer"
                >
                  ↗ open full page
                </a>
                <button className="x" onClick={() => setOpen(null)}>✕</button>
              </div>
              <iframe
                title={open.task_id}
                src={`${BASE}tasks/html_tasks/${open.html_file}`}
                allow="autoplay"
              />
              <div className="modal-inst">
                <b>Agent instruction:</b> {open.instruction}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
