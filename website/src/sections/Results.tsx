import { useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { COLORS, type ResultsData } from '../data'

const CAT_ORDER = ['A_podcast', 'B_meeting', 'C_video', 'D_carousel', 'E_dashboard',
  'F_transient', 'G_phone', 'H_interview', 'I_collab', 'J_game']
const CAT_SHORT: Record<string, string> = {
  A_podcast: 'A · Podcast', B_meeting: 'B · Meeting', C_video: 'C · Video',
  D_carousel: 'D · Carousel', E_dashboard: 'E · Dashboard', F_transient: 'F · Transient',
  G_phone: 'G · Phone', H_interview: 'H · Interview', I_collab: 'I · Collab', J_game: 'J · Game',
}

// Compact axis labels for the 9-model main chart.
const MODEL_SHORT: Record<string, string> = {
  'Claude Sonnet 4.6': 'Claude 4.6',
  'GPT-5.4': 'GPT-5.4',
  'Gemini 2.5 Flash': 'Gemini 2.5',
  'Gemini 3 Flash': 'Gemini 3',
  'Grok-4': 'Grok-4',
  'Grok-4.3': 'Grok-4.3',
  'Grok-4-fast-reasoning': 'Grok-4-fast',
  'EvoCUA-32B': 'EvoCUA-32B',
  'Fara-7B': 'Fara-7B',
}
const short = (m: string) => MODEL_SHORT[m] ?? m

function heatColor(rate: number) {
  // 0 → light red, 50 → pale yellow, 100 → green
  const t = Math.max(0, Math.min(1, rate / 100))
  const stops: [number, number[]][] = [
    [0, [246, 224, 226]],
    [0.5, [248, 240, 214]],
    [1, [212, 230, 200]],
  ]
  let lo = stops[0], hi = stops[stops.length - 1]
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) { lo = stops[i]; hi = stops[i + 1]; break }
  }
  const f = (t - lo[0]) / (hi[0] - lo[0] || 1)
  const c = lo[1].map((v, i) => Math.round(v + (hi[1][i] - v) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export function Results({ data }: { data: ResultsData | null }) {
  const [catModel, setCatModel] = useState(0)
  if (!data) return null

  const mainChart = data.main_results.map((m) => ({
    model: short(m.model) + (m.outlier ? ' ★' : ''),
    group: m.group,
    outlier: m.outlier,
    Standard: m.standard.rate,
    'AOI full': m.aoi_full.rate,
    delta: m.delta,
  }))
  const h = data.headline
  const nClosed = data.main_results.filter((m) => m.group === 'closed').length

  const selModel = data.main_results[catModel]
  const ablChart = data.ablation
    .filter((a) => ['standard', 'aoi_visual', 'aoi_visual_asr', 'aoi_full'].includes(a.mode))
    .map((a) => ({ name: a.label, rate: a.rate, mode: a.mode }))
  const ablColor: Record<string, string> = {
    standard: COLORS.standard, aoi_visual: COLORS.visual,
    aoi_visual_asr: COLORS.asr, aoi_full: COLORS.aoi,
  }

  const selectionChart = [
    ...data.ablation
      .filter((a) => ['uniform_1fps', 'uniform_3fps', 'random_keyframes', 'pixel_diff', 'aoi_visual'].includes(a.mode))
      .map((a) => ({ name: a.label.replace('+ CLIP keyframes (AOI visual)', 'CLIP (AOI visual)'), rate: a.rate, group: 'Claude Sonnet 4.6' })),
  ]

  const thetaChart = data.theta_sweep.map((t) => ({
    theta: t.theta, 'Success rate (%)': t.rate, 'Avg keyframes / step': t.avg_keyframes_per_step,
  }))

  const fourway = data.gemini3_fourway.map((f) => ({ name: f.label, rate: f.rate, mode: f.mode }))
  const fourColor: Record<string, string> = {
    standard: COLORS.standard, standard_audio: COLORS.asr,
    aoi_audio: COLORS.sage, aoi_full: COLORS.aoi,
  }

  const promptChart = data.prompt_decomposition.map((p) => ({ name: p.label, rate: p.rate, mode: p.mode }))
  const narrChart = data.narration_ablation.map((p) => ({ name: p.label, rate: p.rate, mode: p.mode }))
  const ossChart = data.oss_replication.map((m) => ({
    model: m.model.replace('Qwen3-VL-30B-A3B', 'Qwen3-VL 30B').replace('Qwen3-VL-235B-A22B', 'Qwen3-VL 235B'),
    Standard: m.standard.rate,
    'AOI full': m.aoi_full.rate,
    delta: m.delta,
  }))

  const kfContext = (data.keyframe_context ?? []).map((k) => ({
    model: k.model.replace(' Flash', ' Flash').replace('Claude Sonnet 4.6', 'Claude 4.6'),
    'AOI audio (no KF)': k.aoi_audio.rate,
    'AOI full (+ KF)': k.aoi_full.rate,
    delta: k.kf_delta,
  }))

  return (
    <section className="block" id="results">
      <div className="wrap">
        <div className="sec-kicker">Evaluation</div>
        <h2 className="sec-title">Results on DynaCU-Bench</h2>
        <p className="sec-sub">
          100 dynamic browser tasks across 10 categories (podcasts, meetings, video, carousels, dashboards,
          transient notifications, voice calls, interviews, collaborative editing, games). Every chart below
          is computed directly from the raw per-task evaluation records.
        </p>

        <div className="card card-pad" style={{ marginBottom: 22 }}>
          <h3>{h.n_models} CU models, standard loop vs. AOI full — zero retraining</h3>
          <div className="sub">
            Task success rate (%) on the 100 dynamic tasks · {h.n_closed} closed-source + {h.n_open} open-source ·
            paired McNemar p &lt; 10⁻³ for every model except the two starred
          </div>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={mainChart} margin={{ top: 24, right: 8, left: -16, bottom: 18 }} barGap={3}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" vertical={false} />
              <XAxis dataKey="model" interval={0} angle={-22} textAnchor="end" height={56}
                tick={{ fontSize: 12, fill: '#4c566a' }} tickLine={false} axisLine={{ stroke: '#d6dbe6' }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`]} />
              {/* divider between closed-source and open-source models */}
              <ReferenceLine x={mainChart[nClosed]?.model} stroke="#c2c9d6" strokeDasharray="4 4"
                label={{ value: 'open-source →', position: 'top', fontSize: 10.5, fill: '#9aa3b5' }} />
              <Legend wrapperStyle={{ fontSize: 13 }} />
              <Bar dataKey="Standard" fill={COLORS.standard} radius={[4, 4, 0, 0]}>
                <LabelList dataKey="Standard" position="top" style={{ fontSize: 11, fill: '#96434b', fontWeight: 700 }} formatter={(v) => `${v}`} />
              </Bar>
              <Bar dataKey="AOI full" fill={COLORS.aoi} radius={[4, 4, 0, 0]}>
                {mainChart.map((d) => (
                  <Cell key={d.model} fill={d.outlier ? '#9aa8c4' : COLORS.aoi} />
                ))}
                <LabelList dataKey="AOI full" position="top"
                  style={{ fontSize: 11, fill: '#3f5b80', fontWeight: 700 }} formatter={(v) => `${v}`} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="note">
            Across the {h.n_models} models, AOI lifts task success by <b>+{h.delta_min} to +{h.delta_max} pp</b> with
            zero retraining — {h.best_abs_model} reaches the highest absolute {h.best_abs_rate}% (3-seed mean{' '}
            {(data.seeds.reduce((s, x) => s + x.rate, 0) / data.seeds.length).toFixed(1)}%:
            {' '}{data.seeds.map((s) => `${s.rate}%`).join(', ')}). The lone exception is <b>Gemini 3 Flash</b> (★, +9 pp,
            p = 0.18): its default bundle is dragged down by keyframe-token dilution, so components must be selected
            per model — see the sign-flip analysis below.
          </p>
        </div>

        <div className="card card-pad" style={{ marginBottom: 22 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap', marginBottom: 4 }}>
            <div style={{ flex: 1, minWidth: 260 }}>
              <h3>Per-category breakdown</h3>
              <div className="sub" style={{ marginBottom: 0 }}>Tasks passed out of 10 per category, standard vs. AOI full</div>
            </div>
            <div className="seg">
              {data.main_results.map((m, i) => (
                <button key={m.model} className={i === catModel ? 'on' : ''} onClick={() => setCatModel(i)}>
                  {m.model}
                </button>
              ))}
            </div>
          </div>
          <div style={{ overflowX: 'auto', marginTop: 14 }}>
            <table className="res heat">
              <thead>
                <tr>
                  <th>Mode</th>
                  {CAT_ORDER.map((c) => (
                    <th key={c} style={{ textAlign: 'center' }} title={data.categories[c]?.[1]}>
                      {CAT_SHORT[c]}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(['standard', 'aoi_full'] as const).map((mode) => (
                  <tr key={mode}>
                    <td><span className={`chip ${mode === 'standard' ? 'std' : 'aoi'}`}>{mode === 'standard' ? 'Standard' : 'AOI full'}</span></td>
                    {CAT_ORDER.map((c) => {
                      const r = selModel.per_category[mode][c]
                      return (
                        <td key={c} className="cell" style={{ background: r ? heatColor(r.rate) : undefined }}>
                          {r ? `${r.pass}/${r.total}` : '—'}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="note">
            Audio-dependent categories (A, B, G, H) are where screenshot-only agents are blind; visual-temporal
            categories (C, D, E, F) require seeing what happens <em>between</em> steps.
          </p>
        </div>

        <div className="grid-2" style={{ marginBottom: 22 }}>
          <div className="card card-pad">
            <h3>Component ablation — what each piece adds</h3>
            <div className="sub">Claude Sonnet 4.6, 100 tasks · progressive AOI components</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={ablChart} layout="vertical" margin={{ top: 0, right: 44, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`, 'success']} />
                <Bar dataKey="rate" radius={[0, 5, 5, 0]} barSize={26}>
                  {ablChart.map((d) => <Cell key={d.mode} fill={ablColor[d.mode]} />)}
                  <LabelList dataKey="rate" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#1b2236' }} formatter={(v) => `${v}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              The first step (+20 pp) bundles inter-step keyframes with the structured prompt scaffold that
              bare <code>standard</code> lacks: the scaffold carries +19 pp, the keyframe images +1 pp as raw
              input (Prompt-format chart below). ASR +6 pp, narration +18 pp. A narration-discarded control
              splits the narration gain into +8 pp persistent text memory (p = 0.039) + 10 pp inference-time
              chain-of-thought (p = 0.12).
            </p>
          </div>

          <div className="card card-pad">
            <h3>Selection doesn’t matter — and keyframes pay off through narration</h3>
            <div className="sub">Four keyframe selection strategies are statistically indistinguishable (McNemar p &gt; 0.5)</div>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={selectionChart} layout="vertical" margin={{ top: 0, right: 44, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`, 'success']} />
                <Bar dataKey="rate" radius={[0, 5, 5, 0]} barSize={22} fill={COLORS.visual}>
                  <LabelList dataKey="rate" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#1b2236' }} formatter={(v) => `${v}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              Claude Sonnet 4.6, keyframes-only modes (no ASR / narration). Replicated on open-source
              Qwen3-VL-32B: {data.oss_selection.map((o) => `${o.label.toLowerCase()} ${o.rate}%`).join(', ')} on the
              50-task visual subset. How frames are picked doesn’t matter — and the raw images add little on their
              own (+1 pp). Their value is realized through narration: see the keyframe×narration chart below.
            </p>
          </div>
        </div>

        <div className="card card-pad" style={{ marginBottom: 22 }}>
          <h3>AOI vs. production streaming &amp; realtime systems</h3>
          <div className="sub">
            12-task spoken-content subset (3 each: podcast, meeting, phone, interview) · every system drives the
            same browser · ▮ AOI&nbsp;blue = AOI scaffold present
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={data.streaming} layout="vertical" margin={{ top: 0, right: 64, left: 10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
              <XAxis type="number" domain={[0, 12]} tickCount={7} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
              <YAxis type="category" dataKey="system" width={232} tick={{ fontSize: 12, fill: '#4c566a' }} tickLine={false} axisLine={false} />
              <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v, _n, p) => [`${v}/${(p.payload as { total: number }).total} tasks`, 'passed']} />
              <Bar dataKey="pass" radius={[0, 5, 5, 0]} barSize={24}>
                {data.streaming.map((s) => (
                  <Cell key={s.system}
                    fill={s.system.includes('AOI full') ? COLORS.aoi : s.highlight ? COLORS.sage : COLORS.gray} />
                ))}
                <LabelList dataKey="pass" position="right" style={{ fontSize: 12.5, fontWeight: 750, fill: '#1b2236' }} formatter={(v) => `${v}/12`} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="note">
            The decisive contrast: <b>gpt-realtime-2</b> — OpenAI's current GA realtime model — solves just
            2/12 when it drives the browser itself, but <b>11/12</b> once the AOI scaffold handles action
            grounding (and AOI full reaches <b>12/12</b>). The deficit is <em>action grounding, not perception</em>.
            Grok Voice (no vision) tops out at 1/12, and the older Gemini Live / OpenAI Realtime (gpt-4o) adapters
            manage 0–3/12; an adapter sanity check rules out infrastructural failure.
          </p>
        </div>

        <div className="card card-pad" style={{ marginBottom: 22 }}>
          <h3>CLIP threshold (θ) sensitivity</h3>
          <div className="sub">40 visual tasks (categories C–F), Claude Sonnet 4.6 · deployed default θ = 0.04</div>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={thetaChart} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" />
              <XAxis dataKey="theta" tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
              <Tooltip formatter={(v) => [`${v}`]} labelFormatter={(l) => `θ = ${l}`} />
              <Legend wrapperStyle={{ fontSize: 12.5 }} />
              <Line type="monotone" dataKey="Success rate (%)" stroke={COLORS.aoi} strokeWidth={2.5} dot={{ r: 4 }} />
              <Line type="monotone" dataKey="Avg keyframes / step" stroke={COLORS.visual} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="note">
            Success is flat across an order of magnitude of θ while keyframe volume drops — the capture
            mechanism, not precise tuning, carries the gain.
          </p>
        </div>

        <div className="grid-2" style={{ marginBottom: 22 }}>
          <div className="card card-pad">
            <h3>Gemini 3: the first component sign-flip</h3>
            <div className="sub">Four-way decomposition on Gemini 3 Flash, 100 tasks</div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={fourway} layout="vertical" margin={{ top: 0, right: 48, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={195} tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`, 'success']} />
                <Bar dataKey="rate" radius={[0, 5, 5, 0]} barSize={24}>
                  {fourway.map((d) => <Cell key={d.mode} fill={fourColor[d.mode]} />)}
                  <LabelList dataKey="rate" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#1b2236' }} formatter={(v) => `${v}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              Audio adds +12 pp and the structured scaffold +9 pp, but the keyframe-image stream —
              worth +10 pp on Claude via narration — flips to <b>−12 pp</b> on Gemini 3 (image-token
              dilution). Dropping keyframes (AOI audio) recovers +21 pp over standard. The keyframe channel
              is the most model-dependent component; selection must become per-model.
            </p>
          </div>

          <div className="card card-pad">
            <h3>Prompt format vs. perception</h3>
            <div className="sub">standard_structured control, Claude Sonnet 4.6, 100 tasks</div>
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={promptChart} layout="vertical" margin={{ top: 0, right: 48, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`, 'success']} />
                <Bar dataKey="rate" radius={[0, 5, 5, 0]} barSize={24}>
                  {promptChart.map((d, i) => (
                    <Cell key={d.mode} fill={i === promptChart.length - 1 ? COLORS.aoi : i === 0 ? COLORS.gray : COLORS.gold} />
                  ))}
                  <LabelList dataKey="rate" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#1b2236' }} formatter={(v) => `${v}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              The structured prompt scaffold alone is worth +19 pp on Claude; genuine perception adds
              another +25 pp on top. Both components matter — and the control separates them cleanly.
            </p>
          </div>
        </div>

        {kfContext.length > 0 && (
          <div className="card card-pad" style={{ marginBottom: 22 }}>
            <h3>Keyframes pay off only through narration</h3>
            <div className="sub">
              AOI&nbsp;audio (scaffold + ASR + narration, <b>no keyframes</b>) vs. AOI&nbsp;full (+ inter-step
              keyframes), 100 tasks each. The gap is the keyframe images’ marginal value <em>in the deployed
              context</em>.
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={kfContext} margin={{ top: 24, right: 8, left: -16, bottom: 0 }} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" vertical={false} />
                <XAxis dataKey="model" tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={{ stroke: '#d6dbe6' }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`]} />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <Bar dataKey="AOI audio (no KF)" fill={COLORS.sage} radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="AOI audio (no KF)" position="top" style={{ fontSize: 11.5, fill: '#5c7a44', fontWeight: 700 }} formatter={(v) => `${v}`} />
                </Bar>
                <Bar dataKey="AOI full (+ KF)" fill={COLORS.aoi} radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="delta" position="top" style={{ fontSize: 11.5, fill: '#3f5b80', fontWeight: 700 }} formatter={(v) => `${(v as number) > 0 ? '+' : ''}${v} pp`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              The same keyframe-image stream that adds just +1 pp as raw visual input (no narration) is worth
              <b> +10 pp on Claude</b> once the model narrates the captured frames into persistent text — a
              keyframe×narration synergy concentrated on transient-UI and carousel tasks (where content changes
              between steps). Across four models the contribution spans +10 (Claude) and +6 (Gemini 2.5),
              through a neutral −2 (GPT-5.4), to <b>−12 pp on Gemini 3 Flash</b> (image-token dilution): the
              keyframe channel is the most model-dependent AOI component, while audio and narration — both
              delivered as text — are robust everywhere. (GPT-5.4 measured via OpenRouter, both modes, same
              adapter.)
            </p>
          </div>
        )}

        <div className="grid-2" style={{ marginBottom: 22 }}>
          <div className="card card-pad">
            <h3>Narration ablation — why does narration help?</h3>
            <div className="sub">Claude Sonnet 4.6, 100 tasks</div>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={narrChart} layout="vertical" margin={{ top: 0, right: 48, left: 10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" width={195} tick={{ fontSize: 12.5, fill: '#4c566a' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`, 'success']} />
                <Bar dataKey="rate" radius={[0, 5, 5, 0]} barSize={26}>
                  {narrChart.map((d, i) => (
                    <Cell key={d.mode} fill={[COLORS.asr, COLORS.gold, COLORS.aoi][i]} />
                  ))}
                  <LabelList dataKey="rate" position="right" style={{ fontSize: 12, fontWeight: 700, fill: '#1b2236' }} formatter={(v) => `${v}%`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              Generating narration but discarding it keeps +10 pp (inference-time chain-of-thought);
              letting it persist adds a further significant +8 pp (text memory, p = 0.039).
            </p>
          </div>

          <div className="card card-pad">
            <h3>Open-source replication — Qwen3-VL</h3>
            <div className="sub">Independent re-run via OpenRouter (DeepInfra), 100 tasks, standard vs. AOI full</div>
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={ossChart} margin={{ top: 20, right: 8, left: -16, bottom: 0 }} barGap={3}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e7eaf1" vertical={false} />
                <XAxis dataKey="model" tick={{ fontSize: 12, fill: '#4c566a' }} tickLine={false} axisLine={{ stroke: '#d6dbe6' }} interval={0} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#7b8499' }} tickLine={false} axisLine={false} />
                <Tooltip cursor={{ fill: 'rgba(94,129,172,0.07)' }} formatter={(v) => [`${v}%`]} />
                <Legend wrapperStyle={{ fontSize: 12.5 }} />
                <Bar dataKey="Standard" fill={COLORS.standard} radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="Standard" position="top" style={{ fontSize: 11.5, fill: '#96434b', fontWeight: 700 }} formatter={(v) => `${v}`} />
                </Bar>
                <Bar dataKey="AOI full" fill={COLORS.aoi} radius={[4, 4, 0, 0]}>
                  <LabelList dataKey="AOI full" position="top" style={{ fontSize: 11.5, fill: '#3f5b80', fontWeight: 700 }} formatter={(v) => `${v}`} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="note">
              A fully independent open-source replication lands inside the main-table band: Qwen3-VL-235B-A22B
              +{ossChart[1]?.delta} pp and Qwen3-VL-30B-A3B +{ossChart[0]?.delta} pp. The effect is not a quirk of
              one model family or one inference stack.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
