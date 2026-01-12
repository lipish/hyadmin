import React, { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
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
import { Label } from '../components/ui/label';
import { Plus, Play, Square, Trash2 } from 'lucide-react';
import { Model } from '../types';
import { getApiUrl } from '../config/api';

const Models: React.FC = () => {
  const { t } = useTranslation();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [modelToDelete, setModelToDelete] = useState<string | null>(null);
  const [newModel, setNewModel] = useState({
    name: '',
    path: '',
  });
  const { toast } = useToast();

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      const response = await fetch(getApiUrl('/api/models'));
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setModels(data || []);
    } catch (error) {
      // 只在真正的网络错误或API错误时显示错误提示
      console.error('Failed to fetch models:', error);
      toast({
        title: t('common.error'),
        description: t('models.failedToFetch'),
        variant: 'destructive',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleCreateModel = async () => {
    if (!newModel.name || !newModel.path) {
      toast({
        title: t('common.error'),
        description: t('models.fillAllFields'),
        variant: 'destructive',
      });
      return;
    }

    try {
      const response = await fetch(getApiUrl('/api/models'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newModel),
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: t('models.modelCreated'),
        });
        setDialogOpen(false);
        setNewModel({ name: '', path: '' });
        fetchModels();
      } else {
        throw new Error('Failed to create model');
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('models.failedToCreate'),
        variant: 'destructive',
      });
    }
  };

  const handleLoadModel = async (modelId: string) => {
    try {
      const response = await fetch(getApiUrl(`/api/models/${modelId}/load`), {
        method: 'POST',
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: t('models.modelLoaded'),
        });
        fetchModels();
      } else {
        throw new Error('Failed to load model');
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('models.failedToLoad'),
        variant: 'destructive',
      });
    }
  };

  const handleUnloadModel = async (modelId: string) => {
    try {
      const response = await fetch(getApiUrl(`/api/models/${modelId}/unload`), {
        method: 'POST',
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: t('models.modelUnloaded'),
        });
        fetchModels();
      } else {
        throw new Error('Failed to unload model');
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('models.failedToUnload'),
        variant: 'destructive',
      });
    }
  };

  const handleDeleteModel = (modelId: string) => {
    setModelToDelete(modelId);
    setDeleteDialogOpen(true);
  };

  const confirmDeleteModel = async () => {
    if (!modelToDelete) return;

    try {
      const response = await fetch(getApiUrl(`/api/models/${modelToDelete}`), {
        method: 'DELETE',
      });

      if (response.ok) {
        toast({
          title: t('common.success'),
          description: t('models.modelDeleted'),
        });
        fetchModels();
      } else {
        throw new Error('Failed to delete model');
      }
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('models.failedToDelete'),
        variant: 'destructive',
      });
    } finally {
      setDeleteDialogOpen(false);
      setModelToDelete(null);
    }
  };

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'loaded':
        return 'default';
      case 'loading':
        return 'secondary';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  if (loading) {
    return <div className="text-center py-8">{t('common.loading')}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-gray-900">{t('models.title')}</h1>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              {t('models.addModel')}
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>{t('models.addNewModel')}</DialogTitle>
              <DialogDescription>
                {t('models.addNewModelDesc')}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="name" className="text-right">
                  {t('models.name')}
                </Label>
                <Input
                  id="name"
                  value={newModel.name}
                  onChange={(e) => setNewModel(prev => ({ ...prev, name: e.target.value }))}
                  className="col-span-3"
                />
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <Label htmlFor="path" className="text-right">
                  {t('models.path')}
                </Label>
                <Input
                  id="path"
                  value={newModel.path}
                  onChange={(e) => setNewModel(prev => ({ ...prev, path: e.target.value }))}
                  className="col-span-3"
                  placeholder="/path/to/model"
                />
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleCreateModel}>{t('models.createModel')}</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>{t('models.delete')}</DialogTitle>
              <DialogDescription>
                {t('models.confirmDelete')}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>
                {t('common.cancel')}
              </Button>
              <Button variant="destructive" onClick={confirmDeleteModel}>
                {t('common.confirm')}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{t('models.modelManagement')}</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t('models.nameLabel')}</TableHead>
                <TableHead>{t('models.typeLabel')}</TableHead>
                <TableHead>{t('models.pathLabel')}</TableHead>
                <TableHead>{t('models.statusLabel')}</TableHead>
                <TableHead>{t('models.actionsLabel')}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models.map((model) => (
                <TableRow key={model.id}>
                  <TableCell className="font-medium">{model.name}</TableCell>
                  <TableCell>{model.model_type}</TableCell>
                  <TableCell className="font-mono text-sm">{model.path}</TableCell>
                  <TableCell>
                    <Badge variant={getStatusBadgeVariant(model.status)}>
                      {t(`models.${model.status}`)}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex space-x-2">
                      {model.status === 'unloaded' && (
                        <Button
                          size="sm"
                          onClick={() => handleLoadModel(model.id)}
                        >
                          <Play className="h-4 w-4" />
                        </Button>
                      )}
                      {model.status === 'loaded' && (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleUnloadModel(model.id)}
                        >
                          <Square className="h-4 w-4" />
                        </Button>
                      )}
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => handleDeleteModel(model.id)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {models.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                    {t('models.noModelsFound')}
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

export default Models;