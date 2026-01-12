import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { useToast } from '../hooks/use-toast';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '../components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '../components/ui/dialog';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Plus, Power, PowerOff } from 'lucide-react';
import { ApiEndpoint } from '../types';
import { getApiUrl } from '../config/api';

const APIs: React.FC = () => {
  const { t } = useTranslation();
  const [apis, setApis] = useState<ApiEndpoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newApi, setNewApi] = useState({
    name: '',
    api_type: 'OpenAI' as const,
    base_url: '',
  });
  const { toast } = useToast();

  useEffect(() => {
    fetchApis();
  }, []);

  const fetchApis = async () => {
    try {
      const response = await fetch(getApiUrl('/api/apis'));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setApis(data || []);
    } catch (error) {
      // 只在真正的网络错误或API错误时显示错误提示
      console.error('Failed to fetch APIs:', error);
      toast({
        title: t('common.error'),
        description: t('apis.failedToFetch'),
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateApi = async () => {
    if (!newApi.name || !newApi.base_url) {
      toast({
        title: t('common.error'),
        description: t('apis.fillAllFields'),
        variant: 'destructive',
      });
      return;
    }

    try {
      const response = await fetch(getApiUrl('/api/apis'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...newApi,
          enabled: false,
          config: {},
        }),
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: t('apis.apiCreated'),
        });
        setDialogOpen(false);
        setNewApi({ name: '', api_type: 'OpenAI', base_url: '' });
        fetchApis();
      } else {
        throw new Error('Failed to create API');
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('apis.failedToCreate'),
        variant: 'destructive',
      });
    }
  };

  const handleToggleApi = async (apiId: string, enabled: boolean) => {
    try {
      const endpoint = enabled ? 'disable' : 'enable';
      const response = await fetch(getApiUrl(`/api/apis/${apiId}/${endpoint}`), {
        method: 'POST',
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: enabled ? t('apis.apiDisabled') : t('apis.apiEnabled'),
        });
        fetchApis();
      } else {
        throw new Error(`Failed to ${enabled ? 'disable' : 'enable'} API`);
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('apis.failedToToggle'),
        variant: 'destructive',
      });
    }
  };

  if (loading) {
    return <div className="text-center py-8">{t('common.loading')}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">{t('apis.title')}</h1>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              {t('apis.addApi')}
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>{t('apis.addNewApi')}</DialogTitle>
              <DialogDescription>
                {t('apis.addNewApiDesc')}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="api-name" className="text-right">
                  {t('apis.name')}
                </Label>
                <Input
                  id="api-name"
                  value={newApi.name}
                  onChange={(e) => setNewApi(prev => ({ ...prev, name: e.target.value }))}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="api-type" className="text-right">
                  {t('apis.type')}
                </Label>
                <select
                  id="api-type"
                  value={newApi.api_type}
                  onChange={(e) => setNewApi(prev => ({ ...prev, api_type: e.target.value as any }))}
                  className="col-span-3 px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="OpenAI">{t('apis.openai')}</option>
                  <option value="Anthropic">{t('apis.anthropic')}</option>
                  <option value="Codex">{t('apis.codex')}</option>
                  <option value="OpenCode">{t('apis.opencode')}</option>
                  <option value="Custom">{t('apis.custom')}</option>
                </select>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="base-url" className="text-right">
                  {t('apis.baseUrlLabel')}
                </Label>
                <Input
                  id="base-url"
                  value={newApi.base_url}
                  onChange={(e) => setNewApi(prev => ({ ...prev, base_url: e.target.value }))}
                  className="col-span-3"
                  placeholder="https://api.example.com"
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleCreateApi}>{t('apis.createApi')}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{t('apis.apiEndpoints')}</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('apis.apiName')}</TableHead>
                <TableHead>{t('apis.apiType')}</TableHead>
                <TableHead>{t('apis.baseUrl')}</TableHead>
                <TableHead>{t('apis.status')}</TableHead>
                <TableHead>{t('apis.actions')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {apis.map((api) => (
                <TableRow key={api.id}>
                  <TableCell className="font-medium">{api.name}</TableCell>
                  <TableCell>{api.api_type}</TableCell>
                  <TableCell className="font-mono text-sm">{api.base_url}</TableCell>
                  <TableCell>
                    <Badge variant={api.enabled ? 'default' : 'secondary'}>
                      {api.enabled ? t('apis.enabled') : t('apis.disabled')}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Button
                      size="sm"
                      variant={api.enabled ? 'destructive' : 'default'}
                      onClick={() => handleToggleApi(api.id, api.enabled)}
                    >
                      {api.enabled ? (
                        <>
                          <PowerOff className="h-4 w-4 mr-1" />
                          {t('apis.disable')}
                        </>
                      ) : (
                        <>
                          <Power className="h-4 w-4 mr-1" />
                          {t('apis.enable')}
                        </>
                      )}
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {apis.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                    {t('apis.noApisFound')}
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

export default APIs;