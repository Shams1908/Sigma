import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LandingView from './pages/LandingView';
import Workstation from './pages/WorkstationNew';
import InfoPage from './pages/InfoPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingView onLaunch={() => window.location.href = '/workstation'} />} />
        <Route path="/workstation" element={<Workstation />} />
        <Route path="/info" element={<InfoPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
