// Heyi Admin JavaScript Utilities

class HeyiAdmin {
    constructor() {
        this.baseUrl = window.location.origin;
        this.init();
    }

    init() {
        // Auto-refresh functionality
        this.setupAutoRefresh();
        // Form handling
        this.setupForms();
        // Modal handling
        this.setupModals();
    }

    // HTTP utilities
    async apiRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        try {
            const response = await fetch(url, config);
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.message || `HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            this.showAlert(error.message, 'error');
            throw error;
        }
    }

    // UI utilities
    showAlert(message, type = 'info') {
        const alertContainer = document.getElementById('alert-container') ||
            (() => {
                const container = document.createElement('div');
                container.id = 'alert-container';
                container.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 1000;';
                document.body.appendChild(container);
                return container;
            })();

        const alert = document.createElement('div');
        alert.className = `alert alert-${type}`;
        alert.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" style="float: right; border: none; background: none; cursor: pointer;">×</button>
        `;
        alertContainer.appendChild(alert);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (alert.parentElement) {
                alert.remove();
            }
        }, 5000);
    }

    showLoading(element) {
        element.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>Loading...</p>
            </div>
        `;
    }

    // Model management
    async loadModels() {
        const container = document.getElementById('models-container');
        if (!container) return;

        this.showLoading(container);

        try {
            const models = await this.apiRequest('/models');
            this.renderModels(models);
        } catch (error) {
            container.innerHTML = '<p class="text-center text-error">Failed to load models</p>';
        }
    }

    renderModels(models) {
        const container = document.getElementById('models-container');
        if (!container) return;

        if (models.length === 0) {
            container.innerHTML = '<p class="text-center">No models found</p>';
            return;
        }

        container.innerHTML = `
            <table class="table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${models.map(model => `
                        <tr>
                            <td>${model.name}</td>
                            <td>${model.model_type}</td>
                            <td><span class="status-badge status-${model.status.toLowerCase()}">${model.status}</span></td>
                            <td>
                                ${model.status === 'unloaded' ?
                                    `<button class="btn btn-success btn-sm" onclick="admin.loadModel('${model.id}')">Load</button>` :
                                    `<button class="btn btn-warning btn-sm" onclick="admin.unloadModel('${model.id}')">Unload</button>`
                                }
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    async loadModel(modelId) {
        try {
            await this.apiRequest(`/models/${modelId}/load`, { method: 'POST' });
            this.showAlert('Model loaded successfully', 'success');
            this.loadModels();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    async unloadModel(modelId) {
        try {
            await this.apiRequest(`/models/${modelId}/unload`, { method: 'POST' });
            this.showAlert('Model unloaded successfully', 'success');
            this.loadModels();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    // API management
    async loadApis() {
        const container = document.getElementById('apis-container');
        if (!container) return;

        this.showLoading(container);

        try {
            const apis = await this.apiRequest('/apis');
            this.renderApis(apis);
        } catch (error) {
            container.innerHTML = '<p class="text-center text-error">Failed to load APIs</p>';
        }
    }

    renderApis(apis) {
        const container = document.getElementById('apis-container');
        if (!container) return;

        if (apis.length === 0) {
            container.innerHTML = '<p class="text-center">No APIs configured</p>';
            return;
        }

        container.innerHTML = `
            <table class="table">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    ${apis.map(api => `
                        <tr>
                            <td>${api.name}</td>
                            <td>${api.api_type}</td>
                            <td><span class="status-badge ${api.enabled ? 'status-loaded' : 'status-unloaded'}">${api.enabled ? 'Enabled' : 'Disabled'}</span></td>
                            <td>
                                ${api.enabled ?
                                    `<button class="btn btn-warning btn-sm" onclick="admin.disableApi('${api.id}')">Disable</button>` :
                                    `<button class="btn btn-success btn-sm" onclick="admin.enableApi('${api.id}')">Enable</button>`
                                }
                            </td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    async enableApi(apiId) {
        try {
            await this.apiRequest(`/apis/${apiId}/enable`, { method: 'POST' });
            this.showAlert('API enabled successfully', 'success');
            this.loadApis();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    async disableApi(apiId) {
        try {
            await this.apiRequest(`/apis/${apiId}/disable`, { method: 'POST' });
            this.showAlert('API disabled successfully', 'success');
            this.loadApis();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    // Auto-refresh functionality
    setupAutoRefresh() {
        // Auto-refresh every 30 seconds
        setInterval(() => {
            if (window.location.pathname.includes('/models')) {
                this.loadModels();
            } else if (window.location.pathname.includes('/apis')) {
                this.loadApis();
            }
        }, 30000);
    }

    // Form handling
    setupForms() {
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form.id === 'model-form') {
                e.preventDefault();
                this.handleModelForm(form);
            } else if (form.id === 'api-form') {
                e.preventDefault();
                this.handleApiForm(form);
            }
        });
    }

    async handleModelForm(form) {
        const formData = new FormData(form);
        const modelData = {
            name: formData.get('name'),
            path: formData.get('path')
        };

        try {
            await this.apiRequest('/models', {
                method: 'POST',
                body: JSON.stringify(modelData)
            });
            this.showAlert('Model added successfully', 'success');
            form.reset();
            this.loadModels();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    async handleApiForm(form) {
        const formData = new FormData(form);
        const apiData = {
            name: formData.get('name'),
            api_type: formData.get('api_type'),
            base_url: formData.get('base_url'),
            enabled: formData.get('enabled') === 'on',
            config: {
                model_name: formData.get('model_name'),
                api_key: formData.get('api_key'),
                max_tokens: parseInt(formData.get('max_tokens')) || null,
                timeout: parseInt(formData.get('timeout')) || null,
                retry_count: parseInt(formData.get('retry_count')) || null
            }
        };

        try {
            await this.apiRequest('/apis', {
                method: 'POST',
                body: JSON.stringify(apiData)
            });
            this.showAlert('API added successfully', 'success');
            form.reset();
            this.loadApis();
        } catch (error) {
            // Error already shown by apiRequest
        }
    }

    // Modal handling
    setupModals() {
        // Close modals when clicking outside
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                e.target.style.display = 'none';
            }
        });
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.admin = new HeyiAdmin();
});