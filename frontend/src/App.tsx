import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingView from './pages/LandingView';
import Workstation from './pages/WorkstationNew';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingView onLaunch={() => window.location.href = '/workstation'} />} />
        <Route path="/workstation" element={<Workstation />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
