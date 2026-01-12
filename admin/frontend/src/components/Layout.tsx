import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { Button } from './ui/button';
import LanguageSwitcher from './LanguageSwitcher';
import {
  BarChart3,
  Cpu,
  Settings,
  LogOut,
  Home
} from 'lucide-react';

const Layout: React.FC = () => {
  const location = useLocation();
  const { t } = useTranslation();

  const navigation = [
    { name: t('nav.dashboard'), href: '/', icon: Home },
    { name: t('nav.models'), href: '/models', icon: Cpu },
    { name: t('nav.apis'), href: '/apis', icon: Settings },
    { name: t('nav.monitoring'), href: '/monitoring', icon: BarChart3 },
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card shadow-sm border-b w-full">
        <div className="flex justify-between items-center h-16 px-6 w-full">
          <div className="flex items-center flex-1">
            <img 
              src="/logo.png" 
              alt="Heyi Logo" 
              className="h-8 w-8 object-contain flex-shrink-0"
            />
            <span className="ml-3 text-xl font-bold text-gray-900">
              Heyi Admin
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <LanguageSwitcher />
            <Button variant="ghost" size="sm">
              <LogOut className="h-4 w-4 mr-2" />
              {t('nav.logout')}
            </Button>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-card shadow-sm min-h-screen border-r">
          <div className="p-4">
            <ul className="space-y-2">
              {navigation.map((item) => {
                const Icon = item.icon;
                const isActive = location.pathname === item.href;
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
                        isActive
                          ? 'bg-primary/10 text-primary border-r-2 border-primary'
                          : 'text-gray-700 hover:bg-accent hover:text-primary'
                      }`}
                    >
                      <Icon className="h-5 w-5 mr-3" />
                      {item.name}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 p-8">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;