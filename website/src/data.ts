import { useEffect, useState } from 'react'

export interface Rate { pass: number; total: number; rate: number }

export interface ModelResult {
  model: string
  standard: Rate
  aoi_full: Rate
  delta: number
  per_category: { standard: Record<string, Rate>; aoi_full: Record<string, Rate> }
  per_difficulty: { standard: Record<string, Rate>; aoi_full: Record<string, Rate> }
}

export interface RunSummary {
  file: string
  label: string
  model: string
  mode: string
  pass: number
  total: number
  rate: number
  per_category: Record<string, Rate>
  per_difficulty: Record<string, Rate>
}

export interface ResultsData {
  main_results: ModelResult[]
  ablation: RunSummary[]
  oss_selection: RunSummary[]
  theta_sweep: { theta: number; pass: number; total: number; rate: number; avg_keyframes_per_step: number | null }[]
  streaming: ({ system: string } & Rate)[]
  gemini3_fourway: ({ mode: string; label: string } & Rate)[]
  newer_models: { model: string; standard: Rate; aoi_full: Rate; delta: number }[]
  static50: ({ mode: string } & Rate)[]
  seeds: ({ seed: number } & Rate)[]
  oss_replication: { model: string; standard: Rate; aoi_full: Rate; delta: number }[]
  prompt_decomposition: ({ mode: string; label: string } & Rate)[]
  narration_ablation: ({ mode: string; label: string } & Rate)[]
  keyframe_context: {
    model: string
    aoi_audio: Rate
    aoi_full: Rate
    kf_delta: number
    per_category_delta: Record<string, number>
  }[]
  categories: Record<string, [string, string]>
}

export interface TrajStep {
  step: number
  action: string
  narration: string
  audio_text: string
  n_keyframes: number
}

export interface Trajectory {
  task_id: string
  category: string
  difficulty: string
  success: boolean
  steps_taken: number
  total_time_s: number
  error: string | null
  steps: TrajStep[]
}

export interface RunInfo { id: string; label: string; pass: number; total: number; rate: number }

export interface TaskInfo {
  task_id: string
  category: string
  difficulty: string
  instruction: string
  ground_truth: string
  eval_type: string
  axes: string[]
  duration_s: number
  html_file: string
  requires_audio_out: boolean
}

export const BASE = import.meta.env.BASE_URL

export function useJson<T>(path: string | null): T | null {
  const [data, setData] = useState<T | null>(null)
  useEffect(() => {
    if (!path) return
    let live = true
    fetch(BASE + path)
      .then((r) => r.json())
      .then((d) => { if (live) setData(d) })
      .catch((e) => console.error('failed to load', path, e))
    return () => { live = false }
  }, [path])
  return data
}

export const COLORS = {
  standard: '#BF616A',
  aoi: '#5E81AC',
  visual: '#D08770',
  asr: '#B48EAD',
  narr: '#88C0D0',
  sage: '#A3BE8C',
  gold: '#EBCB8B',
  gray: '#4C566A',
}
