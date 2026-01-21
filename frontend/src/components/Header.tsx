/**
 * Header component for the application.
 */

import { Battery, Github, FileText } from 'lucide-react';

interface HeaderProps {
  isApiConnected: boolean;
}

export default function Header({ isApiConnected }: HeaderProps) {
  return (
    <header className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Battery className="w-8 h-8 mr-3" />
            <div>
              <h1 className="text-xl font-bold">Grid-Aware Curtailment Engine</h1>
              <p className="text-sm text-blue-200">
                Renewable Energy Optimization System
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* API Status Indicator */}
            <div className="flex items-center">
              <div
                className={`w-2 h-2 rounded-full mr-2 ${
                  isApiConnected ? 'bg-green-400' : 'bg-red-400'
                }`}
              />
              <span className="text-sm text-blue-200">
                {isApiConnected ? 'API Connected' : 'API Offline'}
              </span>
            </div>

            {/* Links */}
            <a
              href="/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-blue-200 hover:text-white transition-colors"
            >
              <FileText className="w-5 h-5 mr-1" />
              <span className="text-sm">API Docs</span>
            </a>
            <a
              href="https://github.com/iamjeerge/grid-aware-curtailment-engine"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center text-blue-200 hover:text-white transition-colors"
            >
              <Github className="w-5 h-5 mr-1" />
              <span className="text-sm">GitHub</span>
            </a>
          </div>
        </div>
      </div>
    </header>
  );
}
