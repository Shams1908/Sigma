import { useState } from 'react';
import LandingView from './pages/LandingView';
import AnalysisDashboard from './pages/AnalysisDashboard';

type View = 'landing' | 'dashboard';

export default function App() {
  const [currentView, setCurrentView] = useState<View>('landing');

  return (
    <div className="min-h-screen">
      {currentView === 'landing' ? (
        <LandingView onLaunch={() => setCurrentView('dashboard')} />
      ) : (
        <AnalysisDashboard onBack={() => setCurrentView('landing')} />
      )}
    </div>
  );
}
