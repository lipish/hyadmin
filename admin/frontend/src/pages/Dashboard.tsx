import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '../components/ui/dialog';
import {
  Activity,
  Cpu,
  HardDrive,
  Zap,
  Play,
  Square,
  RotateCcw,
  Settings,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { EngineStatus, SystemMetrics } from '../types';
import { getApiUrl } from '../config/api';

interface EngineConfig {
  model_path: string;
  model_name: string;
  host: string;
  port: number;
  num_cpu_threads: number;
  max_batch_size: number;
  api_key?: string;
}

const Dashboard: React.FC = () => {
  const { t, i18n } = useTranslation();

  // 调试信息
  console.log('Current language:', i18n.language);
  console.log('Dashboard title:', t('dashboard.title'));
  console.log('Available languages:', i18n.languages);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>({
    state: 'stopped',
    request_count: 0,
    error_count: 0,
  });
  const [metrics, setMetrics] = useState<SystemMetrics>({
    timestamp: new Date().toISOString(),
    cpu_usage: 0,
    memory_usage: 0,
    disk_usage: 0,
    network_rx: 0,
    network_tx: 0,
  });
  const [engineConfig, setEngineConfig] = useState<EngineConfig>({
    model_path: '',
    model_name: '',
    host: '0.0.0.0',
    port: 10814,
    num_cpu_threads: 49,
    max_batch_size: 32,
    api_key: '',
  });
  const [isConfigDialogOpen, setIsConfigDialogOpen] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [startSuccess, setStartSuccess] = useState<string | null>(null);

  useEffect(() => {
    // Fetch engine status and metrics
    fetchEngineStatus();
    fetchMetrics();

    // Set up polling
    const interval = setInterval(() => {
      fetchEngineStatus();
      fetchMetrics();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const fetchEngineStatus = async () => {
    try {
      const response = await fetch(getApiUrl('/api/monitoring/engine'));
      const data = await response.json();
      setEngineStatus(data);
    } catch (error) {
      console.error('Failed to fetch engine status:', error);
      // 设置默认的中文状态
      setEngineStatus({
        state: 'stopped',
        request_count: 0,
        error_count: 0,
      });
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(getApiUrl('/api/monitoring/metrics'));
      const data = await response.json();
      if (data.length > 0) {
        setMetrics(data[0]);
      }
    } catch (error) {
      console.error('Failed to fetch metrics:', error);
    }
  };

  const controlEngine = async (action: 'start' | 'stop' | 'restart') => {
    if (action === 'start') {
      // Validate configuration before starting
      if (!engineConfig.model_path.trim()) {
        setStartError('请先配置模型路径');
        return;
      }
      if (!engineConfig.model_name.trim()) {
        setStartError('请先配置模型名称');
        return;
      }
      setIsStarting(true);
      setStartError(null);
      setStartSuccess(null);
    }

    try {
      const response = await fetch(getApiUrl('/api/engine/control'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ action }),
      });
      const result = await response.json();
      if (result.success) {
        fetchEngineStatus(); // Refresh status
        if (action === 'start') {
          setStartSuccess('服务启动成功！');
          setIsConfigDialogOpen(false);
        }
      } else {
        if (action === 'start') {
          setStartError(result.message || '启动失败');
        }
      }
    } catch (error) {
      console.error('Failed to control engine:', error);
      if (action === 'start') {
        setStartError('网络错误，无法启动服务');
      }
    } finally {
      setIsStarting(false);
    }
  };

  const getStatusBadgeVariant = (state: string) => {
    switch (state) {
      case 'running':
        return 'default';
      case 'stopped':
        return 'secondary';
      case 'starting':
        return 'outline';
      case 'error':
        return 'destructive';
      default:
        return 'secondary';
    }
  };

  const stats = [
    {
      title: t('monitoring.engineStatus'),
      value: (
        <Badge variant={getStatusBadgeVariant(engineStatus.state)}>
          {t(`monitoring.${engineStatus.state}`)}
        </Badge>
      ),
      icon: Activity,
      description: t('monitoring.engineStatus'),
    },
    {
      title: t('monitoring.cpuUsage'),
      value: `${metrics.cpu_usage.toFixed(1)}%`,
      icon: Cpu,
      description: t('monitoring.cpuUsage'),
    },
    {
      title: t('monitoring.memoryUsage'),
      value: `${(metrics.memory_usage / 1024 / 1024 / 1024).toFixed(1)}GB`,
      icon: HardDrive,
      description: t('monitoring.memoryUsage'),
    },
    {
      title: t('monitoring.requests'),
      value: engineStatus.request_count.toString(),
      icon: Zap,
      description: t('monitoring.requests'),
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">{t('dashboard.title')}</h1>
        <div className="flex space-x-2">
          <Dialog open={isConfigDialogOpen} onOpenChange={setIsConfigDialogOpen}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                配置启动
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>服务启动配置</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="model-path">模型路径</Label>
                  <Input
                    id="model-path"
                    placeholder="/path/to/model"
                    value={engineConfig.model_path}
                    onChange={(e) => setEngineConfig(prev => ({ ...prev, model_path: e.target.value }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="model-name">模型名称</Label>
                  <Input
                    id="model-name"
                    placeholder="deepseek-v3"
                    value={engineConfig.model_name}
                    onChange={(e) => setEngineConfig(prev => ({ ...prev, model_name: e.target.value }))}
                  />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="host">主机地址</Label>
                    <Input
                      id="host"
                      value={engineConfig.host}
                      onChange={(e) => setEngineConfig(prev => ({ ...prev, host: e.target.value }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="port">端口</Label>
                    <Input
                      id="port"
                      type="number"
                      value={engineConfig.port}
                      onChange={(e) => setEngineConfig(prev => ({ ...prev, port: parseInt(e.target.value) || 10814 }))}
                    />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="cpu-threads">CPU线程数</Label>
                    <Input
                      id="cpu-threads"
                      type="number"
                      value={engineConfig.num_cpu_threads}
                      onChange={(e) => setEngineConfig(prev => ({ ...prev, num_cpu_threads: parseInt(e.target.value) || 49 }))}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="batch-size">最大批处理大小</Label>
                    <Input
                      id="batch-size"
                      type="number"
                      value={engineConfig.max_batch_size}
                      onChange={(e) => setEngineConfig(prev => ({ ...prev, max_batch_size: parseInt(e.target.value) || 32 }))}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="api-key">API密钥 (可选)</Label>
                  <Input
                    id="api-key"
                    type="password"
                    placeholder="sk-..."
                    value={engineConfig.api_key}
                    onChange={(e) => setEngineConfig(prev => ({ ...prev, api_key: e.target.value }))}
                  />
                </div>
                {startError && (
                  <div className="flex items-center space-x-2 p-3 rounded-md bg-red-50 border border-red-200 text-red-800">
                    <AlertCircle className="h-4 w-4" />
                    <span className="text-sm">{startError}</span>
                  </div>
                )}
                {startSuccess && (
                  <div className="flex items-center space-x-2 p-3 rounded-md bg-green-50 border border-green-200 text-green-800">
                    <CheckCircle className="h-4 w-4" />
                    <span className="text-sm">{startSuccess}</span>
                  </div>
                )}
                <div className="flex justify-end space-x-2">
                  <Button
                    variant="outline"
                    onClick={() => setIsConfigDialogOpen(false)}
                  >
                    取消
                  </Button>
                  <Button
                    onClick={() => controlEngine('start')}
                    disabled={isStarting || !engineConfig.model_path.trim() || !engineConfig.model_name.trim()}
                  >
                    {isStarting ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        启动中...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        启动服务
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </DialogContent>
          </Dialog>
          <Button
            onClick={() => controlEngine('stop')}
            disabled={engineStatus.state === 'stopped'}
            variant="destructive"
            size="sm"
          >
            <Square className="h-4 w-4 mr-2" />
            {t('common.stop')}
          </Button>
          <Button
            onClick={() => controlEngine('restart')}
            variant="outline"
            size="sm"
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            {t('common.restart')}
          </Button>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <Card key={index}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">
                  {stat.title}
                </CardTitle>
                <Icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
                <p className="text-xs text-muted-foreground">
                  {stat.description}
                </p>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Additional Info */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>{t('monitoring.systemMetrics')}</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.diskUsage')}</dt>
                <dd className="text-sm font-medium">{metrics.disk_usage.toFixed(1)}%</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.network')} RX</dt>
                <dd className="text-sm font-medium">{(metrics.network_rx / 1024 / 1024).toFixed(1)} MB</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.network')} TX</dt>
                <dd className="text-sm font-medium">{(metrics.network_tx / 1024 / 1024).toFixed(1)} MB</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>{t('monitoring.engineStatus')}</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.requests')}</dt>
                <dd className="text-sm font-medium">{engineStatus.request_count}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.errors')}</dt>
                <dd className="text-sm font-medium text-red-600">{engineStatus.error_count}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.throughput')}</dt>
                <dd className="text-sm font-medium">
                  {engineStatus.throughput ? `${engineStatus.throughput.toFixed(1)} req/s` : 'N/A'}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-gray-600">{t('monitoring.memoryUsage')}</dt>
                <dd className="text-sm font-medium">
                  {engineStatus.memory_usage ? `${(engineStatus.memory_usage / 1024 / 1024 / 1024).toFixed(1)}GB` : 'N/A'}
                </dd>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;