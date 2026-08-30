import { useEffect, useMemo, useState } from 'react'
import { getActionTypes, getPolicies, getStatus, verifyAction } from './api.js'

// Key lives in sessionStorage only: survives a reload within the tab, never sent
// anywhere except as a per-request header to the backend.
const KEY_STORAGE = 'openrouter_key'

function defaultsFor(spec) {
  const values = {}
  for (const f of spec.fields) {
    if (f.type === 'boolean') values[f.name] = false
    else if (f.type === 'select') values[f.name] = f.options?.[0] ?? ''
    else if (f.type === 'number') values[f.name] = ''
    else values[f.name] = ''
  }
  return values
}

export default function App() {
  const [apiKey, setApiKey] = useState(() => sessionStorage.getItem(KEY_STORAGE) || '')
  const [actionTypes, setActionTypes] = useState({})
  const [selected, setSelected] = useState('')
  const [values, setValues] = useState({})

  const [result, setResult] = useState(null)
  const [status, setStatus] = useState(null)
  const [policies, setPolicies] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    sessionStorage.setItem(KEY_STORAGE, apiKey)
  }, [apiKey])

  useEffect(() => {
    getActionTypes()
      .then((data) => {
        setActionTypes(data.action_types)
        const first = Object.keys(data.action_types)[0]
        setSelected(first)
        setValues(defaultsFor(data.action_types[first]))
      })
      .catch((e) => setError(e.message))
    getStatus().then(setStatus).catch(() => {})
    getPolicies().then((d) => setPolicies(d.policies)).catch(() => {})
  }, [])

  const spec = actionTypes[selected]

  function selectType(type) {
    setSelected(type)
    setValues(defaultsFor(actionTypes[type]))
    setResult(null)
    setError('')
  }

  function setField(name, value) {
    setValues((v) => ({ ...v, [name]: value }))
  }

  // Coerce form strings into typed values before sending.
  function typedFields() {
    const out = {}
    for (const f of spec.fields) {
      let v = values[f.name]
      if (f.type === 'number') v = v === '' ? null : Number(v)
      out[f.name] = v
    }
    return out
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!apiKey.trim()) {
      setError('Enter your OpenRouter API key first.')
      return
    }
    setBusy(true)
    setError('')
    setResult(null)
    try {
      const data = await verifyAction(apiKey, { actionType: selected, fields: typedFields() })
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  // Index UI actions by kind for rendering.
  const ui = useMemo(() => {
    const map = { highlight: {}, citation: [], correction: [], warning: null, ok: null }
    for (const a of result?.ui_actions || []) {
      if (a.type === 'highlight') map.highlight[a.field] = a.message
      else if (a.type === 'citation') map.citation.push(a)
      else if (a.type === 'correction') map.correction.push(a)
      else if (a.type === 'warning') map.warning = a
      else if (a.type === 'ok') map.ok = a
    }
    return map
  }, [result])

  function applyCorrection(c) {
    if (c.field && c.payload?.corrected_value != null) {
      setField(c.field, c.payload.corrected_value)
    }
  }

  return (
    <div className="page">
      <header className="masthead">
        <div className="brand">
          <span className="eyebrow">Agentic Compliance</span>
          <h1>Company&nbsp;Dashboard</h1>
          <p className="lede">
            Perform an action. An agent verifies it against company policy, flags problems,
            cites the rules, and suggests fixes.
          </p>
        </div>
        <div className="keyfield">
          <label htmlFor="key">OpenRouter API key</label>
          <div className="key-input">
            <KeyIcon />
            <input
              id="key"
              type="password"
              placeholder="sk-or-..."
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              autoComplete="off"
              spellCheck="false"
            />
            <span className={`dot ${apiKey ? 'on' : ''}`} title={apiKey ? 'Key set' : 'No key'} />
          </div>
          <span className="key-hint">Kept in this tab only · sent per request, never stored.</span>
        </div>
      </header>

      {status && status.mode !== 'reference' && <ImplPanel status={status} />}

      <nav className="tabs">
        {Object.entries(actionTypes).map(([type, s]) => (
          <button
            key={type}
            className={type === selected ? 'tab active' : 'tab'}
            onClick={() => selectType(type)}
            type="button"
          >
            {s.label}
          </button>
        ))}
      </nav>

      <main className="workspace">
        <section className="panel panel--form">
          <div className="panel-head">
            <div>
              <h2>{spec?.label || 'Action'}</h2>
              <span className="panel-meta">{spec?.description}</span>
            </div>
          </div>

          {spec && (
            <form onSubmit={handleSubmit}>
              {spec.fields.map((f) => (
                <Field
                  key={f.name}
                  field={f}
                  value={values[f.name]}
                  onChange={setField}
                  problem={ui.highlight[f.name]}
                  correction={ui.correction.find((c) => c.field === f.name)}
                  onApply={applyCorrection}
                />
              ))}
              <button className="primary" type="submit" disabled={busy}>
                {busy ? (
                  <>
                    <span className="spinner" /> Verifying…
                  </>
                ) : (
                  <>
                    Submit action <ArrowIcon />
                  </>
                )}
              </button>
            </form>
          )}
        </section>

        <section className="panel panel--result">
          <div className="panel-head">
            <h2>Verification</h2>
          </div>

          {!result && !busy && <p className="empty">Submit an action to see the agent's verdict.</p>}
          {busy && <p className="empty">The agent is retrieving policy and reasoning…</p>}

          {result && (
            <div className="result">
              <div className={`verdict ${result.verdict.status}`}>
                <span className="verdict-badge">
                  {result.verdict.status === 'valid' ? '✓ Compliant' : '⛔ Problematic'}
                </span>
                <p>{result.verdict.summary}</p>
              </div>

              {result.verdict.problems.map((p, i) => (
                <article className="problem" key={i}>
                  <div className="problem-head">
                    <span className={`sev sev-${p.severity}`}>{p.severity}</span>
                    <code className="problem-fields">{p.field}</code>
                  </div>
                  <p className="problem-why">{p.explanation}</p>
                  <ProblemSolution problem={p} solutions={result.solutions} />
                  <div className="cites">
                    {ui.citation
                      // 1. Gather all citations matching this problem's field
                      .filter((cite) => cite.field === p.field)
                      // 2. Render each snippet attached to this field
                      .map((cite, j) => (
                        <details className="cite" key={j}>
                          <summary>📎 {cite.payload?.source || p.policy_source}</summary>
                          {cite.payload?.snippet && <p>{cite.payload.snippet}</p>}
                        </details>
                      ))}
                  </div>
                </article>
              ))}

              {result.verdict.status === 'valid' && (
                <p className="all-good">No policy issues found. {ui.ok?.message}</p>
              )}
            </div>
          )}

          {policies.length > 0 && (
            <div className="corpus">
              <span className="corpus-label">Grounded on {policies.length} policy documents</span>
            </div>
          )}
        </section>
      </main>

      {error && (
        <div className="toast error" role="alert" onClick={() => setError('')}>
          {error}
        </div>
      )}
    </div>
  )
}

/* --- one form field, with inline problem highlight + apply-fix --- */
function Field({ field, value, onChange, problem, correction, onApply }) {
  const flagged = Boolean(problem)
  return (
    <div className={`field ${flagged ? 'is-flagged' : ''}`}>
      <label htmlFor={field.name}>{field.label}</label>
      {field.type === 'boolean' ? (
        <label className="switch">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(field.name, e.target.checked)}
          />
          <span>{value ? 'Yes' : 'No'}</span>
        </label>
      ) : field.type === 'select' ? (
        <select id={field.name} value={value} onChange={(e) => onChange(field.name, e.target.value)}>
          {field.options.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      ) : (
        <input
          id={field.name}
          type={field.type === 'number' ? 'number' : 'text'}
          value={value}
          onChange={(e) => onChange(field.name, e.target.value)}
        />
      )}
      {flagged && <span className="field-problem">⚠ {problem}</span>}
      {correction && correction.payload?.corrected_value != null && (
        <button type="button" className="apply-fix" onClick={() => onApply(correction)}>
          💡 Apply suggested: <b>{String(correction.payload.corrected_value)}</b>
        </button>
      )}
    </div>
  )
}

function ProblemSolution({ problem, solutions }) {
  const sol = solutions.find((s) => s.problem_id === problem.problem_id)
  if (!sol) return null
  return (
    <p className="fix">
      <b>Suggested fix:</b> {sol.proposed_fix}
    </p>
  )
}

/* --- implementation progress dashboard (same idea as WE6) --- */
function ImplPanel({ status }) {
  const { functions, implemented, total, complete, mode } = status
  const pct = total ? Math.round((implemented / total) * 100) : 0
  const parts = [...new Set(functions.map((f) => f.part))]
  return (
    <section className={`panel impl ${complete ? 'is-complete' : 'is-incomplete'}`}>
      <div className="panel-head">
        <div>
          <h2>Implementation</h2>
          <span className="panel-meta">
            <span className={`mode-chip ${mode}`}>{mode}</span> mode · {implemented}/{total} functions live
          </span>
        </div>
        <span className={`impl-badge ${complete ? 'ok' : 'todo'}`}>
          {complete ? 'All implemented' : `${total - implemented} to go`}
        </span>
      </div>
      <div className="impl-bar" role="progressbar" aria-valuenow={pct}>
        <span style={{ width: `${pct}%` }} />
      </div>
      <div className="impl-grid">
        {parts.map((part) => (
          <div className="impl-col" key={part}>
            {functions
              .filter((f) => f.part === part)
              .map((f) => (
                <div key={f.name} className={`impl-row ${f.implemented ? 'done' : 'todo'}`}>
                  <span className={`impl-mark ${f.implemented ? 'done' : 'todo'}`}>
                    {f.implemented ? '✓' : ''}
                  </span>
                  <code>{f.name}</code>
                  <span className="impl-part">{part}</span>
                </div>
              ))}
          </div>
        ))}
      </div>
    </section>
  )
}

/* --- inline icons --- */
function KeyIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.7">
      <circle cx="8" cy="15" r="4" />
      <path d="M10.8 12.2 20 3m-3 3 2 2m-4 0 2 2" strokeLinecap="round" />
    </svg>
  )
}
function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M5 12h14m-6-6 6 6-6 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
