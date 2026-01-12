import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import {
  Cpu,
  HardDrive,
  Activity,
  Zap,
  Clock
} from 'lucide-react';
import { SystemMetrics, RequestLog, EngineStatus } from '../types';
import { getApiUrl } from '../config/api';

const Monitoring: React.FC = () => {
  const { t } = useTranslation();
  const [metrics, setMetrics] = useState<SystemMetrics[]>([]);
  const [logs, setLogs] = useState<RequestLog[]>([]);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>({
    state: 'stopped',
    request_count: 0,
    error_count: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();

    // Set up polling
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchData = async () => {
    try {
      const [metricsRes, logsRes, engineRes] = await Promise.all([
        fetch(getApiUrl('/api/monitoring/metrics')),
        fetch(getApiUrl('/api/monitoring/logs?limit=50')),
        fetch(getApiUrl('/api/monitoring/engine'))
      ]);

      const [metricsData, logsData, engineData] = await Promise.all([
        metricsRes.json(),
        logsRes.json(),
        engineRes.json()
      ]);

      setMetrics(metricsData);
      setLogs(logsData);
      setEngineStatus(engineData);
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
    } finally {
      setLoading(false);
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

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const currentMetrics = metrics[metrics.length - 1];

  if (loading) {
    return <div className="text-center py-8">{t('common.loading')}</div>;
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">{t('monitoring.title')}</h1>

      {/* Engine Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Activity className="h-5 w-5 mr-2" />
            {t('monitoring.engineStatus')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold mb-2">
                <Badge variant={getStatusBadgeVariant(engineStatus.state)} className="text-lg px-3 py-1">
                  {t(`monitoring.${engineStatus.state}`).toUpperCase()}
                </Badge>
              </div>
              <div className="text-sm text-gray-600">{t('monitoring.status')}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold mb-2">{engineStatus.request_count}</div>
              <div className="text-sm text-gray-600">{t('monitoring.totalRequests')}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold mb-2 text-red-600">{engineStatus.error_count}</div>
              <div className="text-sm text-gray-600">{t('monitoring.errors')}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold mb-2">
                {engineStatus.throughput ? `${engineStatus.throughput.toFixed(1)} req/s` : 'N/A'}
              </div>
              <div className="text-sm text-gray-600">{t('monitoring.throughput')}</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System Metrics */}
      {currentMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Cpu className="h-5 w-5 mr-2" />
              {t('monitoring.systemMetrics')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">{currentMetrics.cpu_usage.toFixed(1)}%</div>
                <div className="text-sm text-gray-600">{t('monitoring.cpuUsage')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  {(currentMetrics.memory_usage / 1024 / 1024 / 1024).toFixed(1)}GB
                </div>
                <div className="text-sm text-gray-600">{t('monitoring.memoryUsage')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">{currentMetrics.disk_usage.toFixed(1)}%</div>
                <div className="text-sm text-gray-600">{t('monitoring.diskUsage')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  <Clock className="h-5 w-5 mx-auto mb-1" />
                </div>
                <div className="text-sm text-gray-600">{t('monitoring.lastUpdate')}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Network Stats */}
      {currentMetrics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <HardDrive className="h-5 w-5 mr-2" />
              {t('monitoring.networkStatistics')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  {(currentMetrics.network_rx / 1024 / 1024).toFixed(1)} MB
                </div>
                <div className="text-sm text-gray-600">{t('monitoring.received')}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold mb-2">
                  {(currentMetrics.network_tx / 1024 / 1024).toFixed(1)} MB
                </div>
                <div className="text-sm text-gray-600">{t('monitoring.transmitted')}</div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Request Logs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Zap className="h-5 w-5 mr-2" />
            {t('monitoring.recentRequestLogs')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('monitoring.time')}</TableHead>
                <TableHead>{t('monitoring.method')}</TableHead>
                <TableHead>{t('monitoring.url')}</TableHead>
                <TableHead>{t('monitoring.status')}</TableHead>
                <TableHead>{t('monitoring.responseTime')}</TableHead>
                <TableHead>{t('monitoring.clientIp')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {logs.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="font-mono text-sm">
                    {formatTimestamp(log.timestamp)}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline" className="font-mono">
                      {log.method}
                    </Badge>
                  </TableCell>
                  <TableCell className="font-mono text-sm max-w-xs truncate">
                    {log.url}
                  </TableCell>
                  <TableCell>
                    <Badge variant={log.status_code >= 400 ? 'destructive' : 'default'}>
                      {log.status_code}
                    </Badge>
                  </TableCell>
                  <TableCell>{log.response_time}ms</TableCell>
                  <TableCell className="font-mono text-sm">{log.client_ip}</TableCell>
                </TableRow>
              ))}
              {logs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                    {t('monitoring.noRequestLogs')}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default Monitoring;