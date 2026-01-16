const API_BASE = import.meta.env.VITE_ADMIN_API_URL || 'http://localhost:9001'

const TOKEN_KEY = 'admin_token'

export function getToken() {
    return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
    localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
    localStorage.removeItem(TOKEN_KEY)
}

async function request(path, options = {}) {
    const token = getToken()
    const headers = {
        'Content-Type': 'application/json',
        ...(options.headers || {}),
    }

    if (token) {
        headers.Authorization = `Bearer ${token}`
    }

    const res = await fetch(`${API_BASE}${path}`, {
        ...options,
        headers,
    })

    if (!res.ok) {
        if (res.status === 401) {
            clearToken()
        }
        const payload = await res.json().catch(() => ({}))
        throw new Error(payload.error || '请求失败')
    }

    if (res.status === 204) {
        return null
    }

    return res.json()
}

export function login(payload) {
    return request('/auth/login', {
        method: 'POST',
        body: JSON.stringify(payload),
    })
}

export function fetchEngines() {
    return request('/engines')
}

export function createEngine(payload) {
    return request('/engines', {
        method: 'POST',
        body: JSON.stringify(payload),
    })
}

export function updateEngine(id, payload) {
    return request(`/engines/${id}`, {
        method: 'PUT',
        body: JSON.stringify(payload),
    })
}

export function fetchApiKeys() {
    return request('/api-keys')
}

export function createApiKey(payload) {
    return request('/api-keys', {
        method: 'POST',
        body: JSON.stringify(payload),
    })
}

export function rotateApiKey(id) {
    return request(`/api-keys/${id}/rotate`, {
        method: 'POST',
    })
}

export function deleteApiKey(id) {
    return request(`/api-keys/${id}`, {
        method: 'DELETE',
    })
}

export function fetchGatewaySettings() {
    return request('/gateway/settings')
}

export function updateGatewaySettings(payload) {
    return request('/gateway/settings', {
        method: 'PUT',
        body: JSON.stringify(payload),
    })
}
