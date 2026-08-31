import { useState } from 'react';
import { AnalysisDashboard } from './pages/AnalysisDashboard';
import type { AnalyzedSignal } from './types';

export default function App() {
  const [signal, setSignal] = useState<AnalyzedSignal | null>(null);

  return (
    <div className="h-dvh overflow-hidden bg-background text-primary">
      <AnalysisDashboard
        signal={signal}
        onLoadSignal={setSignal}
        onClearSignal={() => setSignal(null)}
      />
    </div>
  );
}
