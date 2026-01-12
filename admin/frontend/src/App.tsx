import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from './components/ui/toaster';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Models from './pages/Models';
import APIs from './pages/APIs';
import Monitoring from './pages/Monitoring';
import Login from './pages/Login';
import './i18n';

function App() {
  return (
    <Suspense fallback={<div>加载中...</div>}>
      <Router>
        <div className="min-h-screen bg-background">
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={<Layout />}>
              <Route index element={<Dashboard />} />
              <Route path="models" element={<Models />} />
              <Route path="apis" element={<APIs />} />
              <Route path="monitoring" element={<Monitoring />} />
            </Route>
          </Routes>
          <Toaster />
        </div>
      </Router>
    </Suspense>
  );
}

export default App;