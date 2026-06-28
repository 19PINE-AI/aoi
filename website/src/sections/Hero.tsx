import type { ResultsData } from '../data'
import { BASE } from '../data'

function HeroArch() {
  return (
    <div className="hero-arch" aria-label="AOI architecture overview">
      <div className="ha-row streams">
        <div className="ha-box stream">🖥 Screen stream<small>continuous, ~3 Hz</small></div>
        <div className="ha-box stream">🔊 Audio stream<small>speech · alerts · media</small></div>
      </div>
      <div className="ha-arrow">▼ <span>sub-ms gates — nothing passes on static, silent content</span></div>
      <div className="ha-box aoi">
        <div className="ha-title">Agent-Computer Observation Interface</div>
        <div className="ha-comp"><b>① Keyframes</b> pixel gate → CLIP, 0–5 images/step</div>
        <div className="ha-comp"><b>② Audio</b> volume gate → Whisper ASR transcript</div>
        <div className="ha-comp"><b>③ Narration</b> model-written, persists as text</div>
      </div>
      <div className="ha-arrow">▼ <span>standard image + text input</span></div>
      <div className="ha-row">
        <div className="ha-box model">🤖 Any CU model<small>zero retraining</small></div>
        <div className="ha-box action">⚡ Action<small>click · fill · speak</small></div>
      </div>
    </div>
  )
}

export function Hero({ results }: { results: ResultsData | null }) {
  const claude = results?.main_results.find((m) => m.model === 'Claude Sonnet 4.6')
  const h = results?.headline
  const minD = h?.delta_min ?? 17
  const maxD = h?.delta_max ?? 48
  const nModels = h?.n_models ?? 9
  const stream = results?.streaming ?? []
  const aoiStream = stream.find((s) => s.system.includes('AOI full'))
  const rt2Alone = stream.find((s) => s.system.includes('gpt-realtime-2') && s.system.includes('alone'))

  return (
    <header className="hero">
      <div className="wrap">
        <div className="hero-grid">
          <div>
            <h1>Agent-Computer Observation Interfaces Enable Dynamic Computer Use</h1>
            <p className="byline">
              Bojie Li <span>· Pine AI</span> &nbsp;·&nbsp; Noah Shi <span>· University of Washington</span>
            </p>
            <p className="lede">
              Computer-use agents see the world through a screenshot every 3–5 seconds and are entirely deaf.
              The <b>Agent-Computer Observation Interface (AOI)</b> is a model-agnostic perception layer that
              decouples <b>observation</b> (continuous, adaptive) from <b>action</b> (discrete): gated keyframe
              capture, volume-gated audio understanding, and persistent visual narration — unlocking dynamic
              computer use from any existing CU model with <b>zero retraining</b>.
            </p>
            <div className="hero-actions">
              <a className="btn primary" href={BASE + 'paper/aoi-paper.pdf'} target="_blank" rel="noreferrer">
                📄 Read the paper
              </a>
              <a className="btn ghost" href="#recordings">▶ Watch agent recordings</a>
              <a className="btn ghost" href="#benchmark">🧪 Try the benchmark tasks</a>
            </div>
          </div>
          <HeroArch />
        </div>
        <div className="hero-stats">
          <div className="stat-card">
            <div className="num">
              {claude ? `${claude.standard.rate}% → ${claude.aoi_full.rate}%` : '38% → 82%'}
            </div>
            <div className="lab">Claude Sonnet 4.6 task success on DynaCU-Bench, standard → AOI full</div>
          </div>
          <div className="stat-card">
            <div className="num">
              <span className="up">+{minD}</span> to <span className="up">+{maxD}</span> pp
            </div>
            <div className="lab">Gain across {nModels} CU models, 7B to frontier scale (Gemini 3 Flash the lone exception)</div>
          </div>
          <div className="stat-card">
            <div className="num">
              {aoiStream ? `${aoiStream.pass}/${aoiStream.total}` : '12/12'}{' '}
              <span style={{ fontSize: 18, color: '#9aa8c4' }}>vs {rt2Alone ? `${rt2Alone.pass}/${rt2Alone.total}` : '2/12'}</span>
            </div>
            <div className="lab">Audio subset: AOI vs. native streaming gpt-realtime-2 — the gap is action grounding, not perception</div>
          </div>
          <div className="stat-card">
            <div className="num">≈½ <span style={{ fontSize: 18, color: '#9aa8c4' }}>steps · −15-50% tokens</span></div>
            <div className="lab">AOI ends tasks in roughly half the steps at lower cost — perception replaces blind retries</div>
          </div>
          <div className="stat-card">
            <div className="num">100 + 50</div>
            <div className="lab">DynaCU-Bench: 100 dynamic tasks across 10 categories, plus a 50-task static control</div>
          </div>
        </div>
      </div>
    </header>
  )
}
