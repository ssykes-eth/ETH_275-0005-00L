// Thin wrapper around the FastAPI backend. The OpenRouter key is passed per
// request in the X-OpenRouter-Key header and never persisted server-side.

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

function authHeaders(apiKey) {
  return apiKey ? { 'X-OpenRouter-Key': apiKey } : {}
}

async function handle(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
      if (detail && detail.errors) detail = detail.errors.join('; ')
    } catch {
      /* response had no JSON body */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json()
}

export async function getStatus() {
  return handle(await fetch(`${API_BASE}/status`))
}

export async function getActionTypes() {
  return handle(await fetch(`${API_BASE}/action-types`))
}

export async function getPolicies() {
  return handle(await fetch(`${API_BASE}/policies`))
}

export async function verifyAction(apiKey, { actionType, fields, context }) {
  return handle(
    await fetch(`${API_BASE}/verify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders(apiKey) },
      body: JSON.stringify({ action_type: actionType, fields, context: context || {} }),
    }),
  )
}
