import { useJson, type ResultsData, type RunInfo, type TaskInfo, BASE } from './data'
import { Hero } from './sections/Hero'
import { Architecture } from './sections/Architecture'
import { Results } from './sections/Results'
import { Recordings } from './sections/Recordings'
import { Trajectories } from './sections/Trajectories'
import { Tasks } from './sections/Tasks'

export default function App() {
  const results = useJson<ResultsData>('data/results.json')
  const runs = useJson<RunInfo[]>('data/runs.json')
  const tasks = useJson<TaskInfo[]>('data/tasks.json')

  return (
    <>
      <nav className="nav">
        <div className="wrap nav-inner">
          <a className="nav-logo" href="#top">
            <span className="mark">◉</span> AOI
          </a>
          <div className="nav-links">
            <a href="#architecture">Architecture</a>
            <a href="#results">Results</a>
            <a href="#recordings">Recordings</a>
            <a href="#trajectories">Trajectories</a>
            <a href="#benchmark">Benchmark</a>
            <a className="cta" href={BASE + 'paper/aoi-paper.pdf'} target="_blank" rel="noreferrer">
              Paper PDF
            </a>
          </div>
        </div>
      </nav>
      <main id="top">
        <Hero results={results} />
        <Architecture />
        <Results data={results} />
        <Recordings tasks={tasks} />
        <Trajectories runs={runs} tasks={tasks} />
        <Tasks tasks={tasks} />
        <div style={{ height: 70 }} />
      </main>
    </>
  )
}
