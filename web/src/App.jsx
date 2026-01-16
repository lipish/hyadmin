import { useEffect, useMemo, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from './components/ui/dialog'
import { Badge } from './components/ui/badge'
import { Button } from './components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { Input } from './components/ui/input'
import { Label } from './components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './components/ui/select'
import { Switch } from './components/ui/switch'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from './components/ui/table'
import { Textarea } from './components/ui/textarea'
import {
  clearToken,
  createApiKey,
  createEngine,
  deleteApiKey,
  fetchApiKeys,
  fetchEngines,
  fetchGatewaySettings,
  getToken,
  login,
  rotateApiKey,
  setToken,
  updateEngine,
} from './lib/api'

const emptySettings = {
  strategy: 'weighted',
  failover_enabled: true,
  vip_enabled: false,
  notes: '',
}

function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark')
  const [token, setTokenState] = useState(getToken())
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [activeTab, setActiveTab] = useState('overview')
  const [engines, setEngines] = useState([])
  const [apiKeys, setApiKeys] = useState([])
  const [settings, setSettings] = useState(emptySettings)
  const [loginForm, setLoginForm] = useState({ username: '', password: '' })
  const [engineForm, setEngineForm] = useState({
    name: '',
    base_url: '',
    kind: 'local',
    group_name: 'default',
    weight: 100,
    priority: 0,
    enabled: true,
    status: 'active',
  })
  const [apiKeyForm, setApiKeyForm] = useState({ name: '' })
  const [lastCreatedKey, setLastCreatedKey] = useState('')

  const statusStyles = useMemo(
    () => ({
      active: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/40',
      running: 'bg-emerald-500/15 text-emerald-200 border-emerald-500/40',
      degraded: 'bg-amber-500/15 text-amber-200 border-amber-500/40',
      idle: 'bg-slate-500/15 text-slate-200 border-slate-500/40',
    }),
    [],
  )

  const loadAll = async () => {
    setLoading(true)
    setError('')
    try {
      const [enginesRes, apiKeysRes, settingsRes] = await Promise.all([
        fetchEngines(),
        fetchApiKeys(),
        fetchGatewaySettings(),
      ])
      setEngines(enginesRes || [])
      setApiKeys(apiKeysRes || [])
      setSettings(settingsRes || emptySettings)
    } catch (err) {
      if (!getToken()) {
        setTokenState('')
      }
      setError(err.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme])

  const themeToggleLabel = theme === 'dark' ? '切换亮色' : '切换暗色'
  const isDark = theme === 'dark'

  const pageBackground =
    theme === 'dark'
      ? 'bg-[radial-gradient(circle_at_top,_rgba(34,197,94,0.28),_transparent_52%),radial-gradient(circle_at_30%_20%,_rgba(56,189,248,0.28),_transparent_45%),linear-gradient(180deg,_#0b1224_0%,_#020617_55%,_#01040b_100%)] text-slate-100'
      : 'bg-[radial-gradient(circle_at_top,_rgba(16,185,129,0.18),_transparent_55%),radial-gradient(circle_at_20%_20%,_rgba(59,130,246,0.18),_transparent_45%),linear-gradient(180deg,_#f8fafc_0%,_#e2e8f0_55%,_#f1f5f9_100%)] text-slate-900'

  const handleToggleEngine = async (engine) => {
    setLoading(true)
    setError('')
    try {
      await updateEngine(engine.id, {
        name: engine.name,
        base_url: engine.base_url,
        kind: engine.kind,
        status: engine.status,
        group_name: engine.group_name,
        weight: engine.weight,
        priority: engine.priority,
        enabled: !engine.enabled,
      })
      await loadAll()
    } catch (err) {
      setError(err.message || '更新引擎失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (token) {
      loadAll()
    }
  }, [token])

  const handleLogin = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await login(loginForm)
      setToken(res.token)
      setTokenState(res.token)
    } catch (err) {
      setError(err.message || '登录失败')
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = () => {
    clearToken()
    setTokenState('')
  }

  const handleCreateEngine = async () => {
    setLoading(true)
    setError('')
    try {
      await createEngine(engineForm)
      setEngineForm({
        name: '',
        base_url: '',
        kind: 'local',
        group_name: 'default',
        weight: 100,
        priority: 0,
        enabled: true,
        status: 'active',
      })
      await loadAll()
    } catch (err) {
      setError(err.message || '创建引擎失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateApiKey = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await createApiKey(apiKeyForm)
      setApiKeyForm({ name: '' })
      setLastCreatedKey(res?.key || '')
      await loadAll()
    } catch (err) {
      setError(err.message || '创建 API Key 失败')
    } finally {
      setLoading(false)
    }
  }

  const handleRotateApiKey = async (id) => {
    setLoading(true)
    setError('')
    try {
      const res = await rotateApiKey(id)
      setLastCreatedKey(res?.key || '')
      await loadAll()
    } catch (err) {
      setError(err.message || '轮换 API Key 失败')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteApiKey = async (id) => {
    setLoading(true)
    setError('')
    try {
      await deleteApiKey(id)
      await loadAll()
    } catch (err) {
      setError(err.message || '删除 API Key 失败')
    } finally {
      setLoading(false)
    }
  }


  if (!token) {
    return (
      <div className={`min-h-screen ${pageBackground}`}>
        <div className="mx-auto flex min-h-screen max-w-lg items-center px-6">
          <Card className="w-full border-white/10 bg-white/5">
            <CardHeader>
              <CardTitle className="text-lg text-white">管理员登录</CardTitle>
              <p className="text-sm text-slate-400">请先登录以管理网关与引擎。</p>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="username">账号</Label>
                <Input
                  id="username"
                  value={loginForm.username}
                  onChange={(event) =>
                    setLoginForm((prev) => ({ ...prev, username: event.target.value }))
                  }
                  placeholder="admin"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password">密码</Label>
                <Input
                  id="password"
                  type="password"
                  value={loginForm.password}
                  onChange={(event) =>
                    setLoginForm((prev) => ({ ...prev, password: event.target.value }))
                  }
                  placeholder="admin123"
                />
              </div>
              {error && (
                <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                  {error}
                </div>
              )}
              <Button
                className="w-full bg-emerald-400 text-emerald-950 hover:bg-emerald-300"
                onClick={handleLogin}
                disabled={loading}
              >
                {loading ? '登录中...' : '登录'}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const sidebarItems = [
    { key: 'overview', label: '状态面板' },
    { key: 'engines', label: '引擎管理' },
    { key: 'gateway', label: '网关管理' },
    { key: 'apps', label: '应用管理' },
    { key: 'settings', label: '系统设置' },
  ]

  const engineStatusList = [
    {
      key: 'heyi',
      name: 'Heyi 引擎',
      data: {
        pending: 0,
        prefilling: 0,
        decoding: 1,
        throughput: 19.77,
        requests: [
          {
            state: '解码中',
            ttft: 706.29,
            avgTbt: 50.78,
            p95Tbt: 52.53,
            throughput: 19.69,
            promptTokens: 13,
            cacheHitTokens: 0,
            decodeTokens: 845,
          },
        ],
        config: {
          auto_license: true,
          kvcache_num_tokens: 65536,
          max_length: 65536,
          layerwise_prefill_device: 1,
          prefill_chunk_size: 512,
          model_name: 'Qwen3-Coder-480B-A35B',
          batch_sizes_per_runner: [1, 1],
          num_cpu_threads: 49,
          use_cuda_graph: true,
          max_batch_size: 32,
          max_new_tokens: 128000,
          enable_layerwise_prefill: true,
          layerwise_prefill_thresh_len: 4096,
          host: '0.0.0.0',
          port: 10814,
          api_key: '',
          trust_remote_code: true,
          thinking: false,
        },
      },
    },
    {
      key: 'vllm',
      name: 'vLLM 引擎',
      data: {
        pending: 2,
        prefilling: 1,
        decoding: 3,
        throughput: 42.3,
        requests: [],
        config: { model_name: 'vLLM-Qwen3', host: '0.0.0.0', port: 8000 },
      },
    },
  ]

  const engineProfiles = [
    {
      key: 'heyi',
      name: 'Heyi 引擎',
      desc: '本地优化推理引擎',
      gpu: 'GPU 0 / 1 (A100 80G)',
      params: '--model /data/heyi --max-batch-size 32 --kvcache-num-tokens 65536',
    },
    {
      key: 'vllm',
      name: 'vLLM 引擎',
      desc: '高吞吐 KV 缓存引擎',
      gpu: 'GPU 2 (H100 80G)',
      params: '--model /data/vllm --tensor-parallel-size 2 --max-num-batched-tokens 8192',
    },
  ]

  return (
    <div className={`h-screen ${pageBackground}`}>
      <div className="mx-auto flex h-full max-w-7xl gap-6 px-6 pb-10 pt-8">
        <aside className="flex h-full w-56 flex-col gap-6 rounded-3xl border border-slate-300/70 bg-white/70 p-5 shadow-lg dark:border-slate-700/60 dark:bg-slate-900/70 dark:text-white">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-emerald-700 dark:text-emerald-200">Heyi Admin</p>
            <p className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">控制台</p>
          </div>
          <nav className="flex flex-col gap-2">
            {sidebarItems.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setActiveTab(item.key)}
                className={`rounded-xl px-3 py-2 text-left text-sm transition ${activeTab === item.key
                  ? 'bg-emerald-400/20 text-emerald-900 dark:bg-emerald-500/20 dark:text-emerald-100'
                  : 'text-slate-700/80 dark:text-slate-200/80 hover:bg-slate-900/10 dark:hover:bg-slate-800/60'
                  }`}
              >
                {item.label}
              </button>
            ))}
          </nav>
          <div className="mt-auto flex items-center justify-between rounded-2xl border border-slate-300/70 bg-white/70 px-3 py-2 dark:border-slate-700/60 dark:bg-slate-900/70">
            <button
              type="button"
              aria-label={loading ? '刷新中' : '刷新数据'}
              onClick={loadAll}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-300/70 bg-white/70 text-slate-700 transition hover:border-slate-400/80 dark:border-slate-700/60 dark:bg-slate-900/70 dark:text-slate-100"
            >
              <svg viewBox="0 0 24 24" className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} fill="currentColor">
                <path d="M12 4a8 8 0 017.75 6h-2.1A6 6 0 106 12h2.1A4 4 0 1112 16a4 4 0 01-3.6-2.2H6.2A6 6 0 0012 18a6 6 0 006-6h2a8 8 0 01-8 8 8 8 0 010-16z" />
              </svg>
            </button>
            <button
              type="button"
              aria-label="退出登录"
              onClick={handleLogout}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-300/70 bg-white/70 text-slate-700 transition hover:border-slate-400/80 dark:border-slate-700/60 dark:bg-slate-900/70 dark:text-slate-100"
            >
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                <path d="M5 3h7a2 2 0 012 2v3h-2V5H5v14h7v-3h2v3a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2zm11.59 6H9v2h7.59l-2.3 2.29 1.42 1.42L21.41 12l-5.7-4.71-1.42 1.42L16.59 11z" />
              </svg>
            </button>
            <button
              type="button"
              aria-label={themeToggleLabel}
              onClick={() => setTheme(isDark ? 'light' : 'dark')}
              className="flex h-9 w-9 items-center justify-center rounded-full border border-slate-300/70 bg-white/70 text-slate-700 transition hover:border-slate-400/80 dark:border-slate-700/60 dark:bg-slate-900/70 dark:text-slate-100"
            >
              {isDark ? (
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                  <path d="M12 4.5a.75.75 0 01.75-.75h.5a.75.75 0 010 1.5h-.5A.75.75 0 0112 4.5zm6.53 2.47a.75.75 0 010-1.06l.35-.35a.75.75 0 111.06 1.06l-.35.35a.75.75 0 01-1.06 0zM19.5 12a.75.75 0 01.75-.75h.5a.75.75 0 010 1.5h-.5A.75.75 0 0119.5 12zM17.82 17.82a.75.75 0 011.06 0l.35.35a.75.75 0 11-1.06 1.06l-.35-.35a.75.75 0 010-1.06zM12 18.75a.75.75 0 01.75.75v.5a.75.75 0 01-1.5 0v-.5a.75.75 0 01.75-.75zM6.18 17.82a.75.75 0 010 1.06l-.35.35a.75.75 0 11-1.06-1.06l.35-.35a.75.75 0 011.06 0zM3.75 12a.75.75 0 01.75-.75h.5a.75.75 0 010 1.5h-.5A.75.75 0 013.75 12zM6.18 6.18a.75.75 0 01-1.06 0l-.35-.35A.75.75 0 014.83 4.77l.35.35a.75.75 0 010 1.06zM12 7a5 5 0 100 10 5 5 0 000-10z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor">
                  <path d="M21.64 13a9 9 0 01-10.64-10.64A9 9 0 1021.64 13z" />
                </svg>
              )}
            </button>
          </div>
        </aside>

        <main className="flex-1 space-y-6 overflow-y-auto pr-2">
          <header className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="text-2xl font-semibold text-slate-900 dark:text-white">
                {activeTab === 'overview' && '引擎状态面板'}
                {activeTab === 'engines' && '引擎与 GPU 管理'}
                {activeTab === 'gateway' && '网关管理'}
                {activeTab === 'apps' && '应用管理'}
                {activeTab === 'settings' && '系统设置'}
              </h1>
              <p className="mt-1 text-sm text-slate-700/80 dark:text-slate-200/80">
                {activeTab === 'overview' && '实时系统性能与健康监控。'}
                {activeTab === 'engines' && '管理 heyi / vllm 运行参数。'}
                {activeTab === 'gateway' && '分配 API 能力、调度策略与路由规则。'}
                {activeTab === 'apps' && '管理应用接入、权限与路由策略。'}
                {activeTab === 'settings' && '控制台基础参数与策略。'}
              </p>
            </div>
            {error && (
              <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                {error}
              </div>
            )}
          </header>

          {activeTab === 'overview' && (
            <div className="space-y-6">
              <div className="space-y-6">
                {engineStatusList.map((engine) => (
                  <Card key={engine.key} className="border border-slate-300/60 bg-white/80 shadow-[0_10px_30px_-20px_rgba(15,23,42,0.35)] ring-1 ring-slate-200/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/75 dark:ring-slate-800/80">
                    <CardHeader className="flex flex-row items-center justify-between">
                      <div>
                        <CardTitle className="text-lg text-slate-900 dark:text-white">{engine.name}</CardTitle>
                        <p className="text-sm text-slate-700/80 dark:text-slate-200/80">实时请求追踪</p>
                      </div>
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button size="sm" variant="secondary">
                            配置信息
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="max-w-2xl">
                          <DialogHeader>
                            <DialogTitle>{engine.name} 配置</DialogTitle>
                            <DialogDescription>当前引擎运行参数。</DialogDescription>
                          </DialogHeader>
                          <pre className="max-h-96 overflow-auto rounded-xl bg-slate-950/80 p-4 text-xs text-emerald-100">
                            {JSON.stringify(engine.data.config || {}, null, 2)}
                          </pre>
                        </DialogContent>
                      </Dialog>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-4">
                        {[
                          { label: '等待请求', value: engine.data.pending, color: 'text-sky-500' },
                          { label: '预填充请求', value: engine.data.prefilling, color: 'text-amber-500' },
                          { label: '解码请求', value: engine.data.decoding, color: 'text-rose-500' },
                          { label: '吞吐量 (tokens/s)', value: engine.data.throughput, color: 'text-emerald-500' },
                        ].map((item) => (
                          <div key={item.label} className="rounded-xl border border-slate-300/60 bg-white/70 p-4 dark:border-slate-700/60 dark:bg-slate-900/70">
                            <p className={`text-2xl font-semibold ${item.color}`}>{item.value}</p>
                            <p className="text-sm text-slate-700/80 dark:text-slate-200/80">{item.label}</p>
                          </div>
                        ))}
                      </div>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">状态</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">首字延迟(ms)</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">平均吐字延迟(ms)</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">P95 吐字延迟(ms)</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">吞吐量</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">Prefill Tokens</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">Cache Hit Tokens</TableHead>
                            <TableHead className="text-slate-900/80 dark:text-slate-100/90">Decode Tokens</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {(engine.data.requests || []).map((req, index) => (
                            <TableRow key={`${engine.key}-${req.state}-${index}`}>
                              <TableCell>{req.state}</TableCell>
                              <TableCell>{req.ttft}</TableCell>
                              <TableCell>{req.avgTbt}</TableCell>
                              <TableCell>{req.p95Tbt}</TableCell>
                              <TableCell>{req.throughput}</TableCell>
                              <TableCell>{req.promptTokens}</TableCell>
                              <TableCell>{req.cacheHitTokens}</TableCell>
                              <TableCell>{req.decodeTokens}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'apps' && (
            <Card className="border border-slate-300/60 bg-white/80 shadow-[0_10px_30px_-20px_rgba(15,23,42,0.35)] ring-1 ring-slate-200/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/75 dark:ring-slate-800/80">
              <CardHeader>
                <CardTitle className="text-lg text-slate-900 dark:text-white">应用接入</CardTitle>
                <p className="text-sm text-slate-700/80 dark:text-slate-200/80">为各应用分配访问权限与路由。</p>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {appProfiles.map((app) => (
                    <div
                      key={app.key}
                      className="flex flex-col gap-2 rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70"
                    >
                      <div>
                        <p className="text-sm font-medium text-slate-900 dark:text-white">{app.name}</p>
                        <p className="text-xs text-slate-700/80 dark:text-slate-200/80">{app.desc}</p>
                      </div>
                      <div className="grid gap-2 text-xs text-slate-700/80 dark:text-slate-200/80 sm:grid-cols-3">
                        <div>
                          <span className="block text-[10px] uppercase tracking-[0.2em]">权限</span>
                          <span>{app.access}</span>
                        </div>
                        <div>
                          <span className="block text-[10px] uppercase tracking-[0.2em]">路由</span>
                          <span>{app.route}</span>
                        </div>
                        <div>
                          <span className="block text-[10px] uppercase tracking-[0.2em]">目标引擎</span>
                          <span>{app.target}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}

          {activeTab === 'gateway' && (
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="border border-slate-300/60 bg-white/80 shadow-[0_10px_30px_-20px_rgba(15,23,42,0.35)] ring-1 ring-slate-200/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/75 dark:ring-slate-800/80">
                <CardHeader>
                  <CardTitle className="text-lg text-slate-900 dark:text-white">API 路由分配</CardTitle>
                  <p className="text-sm text-slate-700/80 dark:text-slate-200/80">统一管理外部接口与内部引擎的映射。</p>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                    <p className="text-sm font-medium text-slate-900 dark:text-white">/v1/chat/completions</p>
                    <p className="text-xs text-slate-700/80 dark:text-slate-200/80">默认路由：heyi / vllm</p>
                  </div>
                  <div className="rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                    <p className="text-sm font-medium text-slate-900 dark:text-white">/v1/embeddings</p>
                    <p className="text-xs text-slate-700/80 dark:text-slate-200/80">默认路由：vllm</p>
                  </div>
                </CardContent>
              </Card>

              <Card className="border border-slate-300/60 bg-white/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/70">
                <CardHeader>
                  <CardTitle className="text-lg text-slate-900 dark:text-white">调度策略</CardTitle>
                  <p className="text-sm text-slate-700/80 dark:text-slate-200/80">策略与优先级概览。</p>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                    <p className="text-sm text-slate-700/80 dark:text-slate-200/80">调度策略</p>
                    <p className="text-base font-semibold text-slate-900 dark:text-white">{settings.strategy || 'weighted'}</p>
                    <p className="text-xs text-slate-700/80 dark:text-slate-200/80">策略说明：{settings.notes || '默认策略'}</p>
                  </div>
                  <div className="flex items-center justify-between rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                    <div>
                      <p className="text-sm font-medium text-slate-900 dark:text-white">故障切换</p>
                      <p className="text-xs text-slate-700/80 dark:text-slate-200/80">主引擎不可用时自动切换</p>
                    </div>
                    <Switch checked={settings.failover_enabled} />
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeTab === 'engines' && (
            <div className="grid gap-4 md:grid-cols-2">
              {engineProfiles.map((engine) => (
                <Card key={engine.key} className="border border-slate-300/60 bg-white/80 shadow-[0_10px_30px_-20px_rgba(15,23,42,0.35)] ring-1 ring-slate-200/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/75 dark:ring-slate-800/80">
                  <CardHeader>
                    <CardTitle className="text-lg text-slate-900 dark:text-white">{engine.name}</CardTitle>
                    <p className="text-sm text-slate-700/80 dark:text-slate-200/80">{engine.desc}</p>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div>
                      <p className="text-sm font-medium text-slate-900 dark:text-slate-100">GPU 资源</p>
                      <p className="text-sm text-slate-700/80 dark:text-slate-200/80">{engine.gpu}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900 dark:text-slate-100">启动参数</p>
                      <div className="mt-2 rounded-xl bg-slate-950/80 p-3 text-xs text-emerald-100">
                        {engine.params}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}

          {activeTab === 'settings' && (
            <Card className="border border-slate-300/60 bg-white/70 backdrop-blur dark:border-slate-700/60 dark:bg-slate-900/70">
              <CardHeader>
                <CardTitle className="text-lg text-slate-900 dark:text-white">系统设置</CardTitle>
                <p className="text-sm text-slate-700/80 dark:text-slate-200/80">当前策略与运行状态</p>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/80 dark:bg-slate-900/80">
                  <div>
                    <p className="text-sm font-medium text-slate-900 dark:text-white">故障切换</p>
                    <p className="text-xs text-slate-700/80 dark:text-slate-200/80">主引擎不可用时自动切至备用分组</p>
                  </div>
                  <Switch
                    checked={settings.failover_enabled}
                    onCheckedChange={(value) =>
                      setSettings((prev) => ({ ...prev, failover_enabled: value }))
                    }
                  />
                </div>
                <div className="flex items-center justify-between rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                  <div>
                    <p className="text-sm font-medium text-slate-900 dark:text-white">高优先级通道</p>
                    <p className="text-xs text-slate-700/80 dark:text-slate-200/80">为 VIP 请求保留独占权重</p>
                  </div>
                  <Switch
                    checked={settings.vip_enabled}
                    onCheckedChange={(value) =>
                      setSettings((prev) => ({ ...prev, vip_enabled: value }))
                    }
                  />
                </div>
                <div className="rounded-lg border border-slate-300/60 bg-white/70 px-4 py-3 dark:border-slate-700/60 dark:bg-slate-900/70">
                  <p className="text-sm text-slate-700/80 dark:text-slate-200/80">调度策略</p>
                  <p className="mt-1 text-base font-semibold text-slate-900 dark:text-white">{settings.strategy || 'weighted'}</p>
                  <p className="mt-2 text-sm text-slate-700/80 dark:text-slate-200/80">策略说明：{settings.notes || '默认策略'}</p>
                </div>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  )
}

export default App
