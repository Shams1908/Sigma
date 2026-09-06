import React, { useState, useEffect } from 'react';
import { checkBackendHealth } from '../../services/api';

export const AnalysisDashboard: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<'connecting' | 'online' | 'offline'>('connecting');
  const [backendInfo, setBackendInfo] = useState<{ project: string; status: string } | null>(null);

  useEffect(() => {
    const verifyHealth = async () => {
      try {
        const response = await checkBackendHealth();
        if (response.status === 'ok') {
          setBackendStatus('online');
          setBackendInfo(response);
        } else {
          setBackendStatus('offline');
        }
      } catch (err) {
        console.error('Failed to contact backend:', err);
        setBackendStatus('offline');
      }
    };
    verifyHealth();
    
    // Auto-poll health every 10 seconds
    const interval = setInterval(verifyHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      {/* Top Navigation / Header */}
      <header className="dashboard-header">
        <div className="logo-section">
          <div className="glow-dot"></div>
          <h1>SIGMA</h1>
        </div>
        <div className="subtitle-section">
          <span>Signal Intelligence & Guided Modulation Analysis</span>
        </div>
        <div className="connection-badge-container">
          <span className={`connection-badge ${backendStatus}`}>
            <span className="badge-pulse"></span>
            Backend: {backendStatus.toUpperCase()}
          </span>
        </div>
      </header>

      {/* Main Grid Layout */}
      <main className="dashboard-grid">
        {/* Left column: Controls and Status */}
        <section className="dashboard-sidebar">
          {/* Phase Information Card */}
          <div className="glass-card main-info-card">
            <h2 className="text-glow">System Information</h2>
            <p className="card-description">
              SIGMA is an Automated RF Signal Analysis & Hypothesis Validation Platform.
            </p>
            <div className="info-list">
              <div className="info-item">
                <span className="info-label">Active Phase:</span>
                <span className="info-value highlight-cyan">Phase 0 — Setup</span>
              </div>
              <div className="info-item">
                <span className="info-label">Environment:</span>
                <span className="info-value">Development</span>
              </div>
              {backendInfo && (
                <div className="info-item">
                  <span className="info-label">Remote Node:</span>
                  <span className="info-value text-green">{backendInfo.project} (OK)</span>
                </div>
              )}
            </div>
            <div className="phase-indicator-bar">
              <div className="phase-progress" style={{ width: '12%' }}></div>
            </div>
            <div className="phase-text-footer">12% skeleton setup complete</div>
          </div>

          {/* Ingestion Control Placeholder */}
          <div className="glass-card">
            <h3>File Ingestion</h3>
            <div className="placeholder-zone">
              <svg className="placeholder-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p>Drag and drop IQ or WAV recordings here</p>
              <span className="hint-text">Supported formats: .iq, .wav, .raw</span>
            </div>
          </div>

          {/* Parameter Panel Placeholder */}
          <div className="glass-card">
            <h3>Estimated Parameters</h3>
            <div className="parameter-placeholder-list">
              {['Estimated SNR', 'Symbol Rate', 'Carrier Offset', 'Bandwidth'].map((param) => (
                <div key={param} className="param-row-skeleton">
                  <span className="param-name">{param}</span>
                  <span className="param-value-skele">--</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Center column: Visualization & Hypotheses */}
        <section className="dashboard-main-content">
          {/* Top visual placeholders */}
          <div className="visuals-grid">
            <div className="glass-card visual-card">
              <div className="card-header-with-actions">
                <h3>Spectrum (PSD)</h3>
                <span className="badge-future">Phase 2</span>
              </div>
              <div className="visual-placeholder-graph">
                <div className="mock-gridline-y"></div>
                <div className="mock-gridline-x"></div>
                <svg className="mock-wave-svg" viewBox="0 0 400 150">
                  <path 
                    d="M 0 130 Q 80 130 120 128 T 160 120 T 180 50 T 200 45 T 220 50 T 240 120 T 280 128 T 400 130" 
                    fill="none" 
                    stroke="rgba(0, 242, 254, 0.4)" 
                    strokeWidth="2.5"
                  />
                </svg>
                <div className="graph-label-overlay">Spectral Power Density visualizer placeholder</div>
              </div>
            </div>

            <div className="glass-card visual-card">
              <div className="card-header-with-actions">
                <h3>Waterfall Spectrogram</h3>
                <span className="badge-future">Phase 2</span>
              </div>
              <div className="visual-placeholder-graph waterfall-bg">
                <div className="waterfall-overlay-grid"></div>
                <div className="graph-label-overlay">Waterfall time-frequency visualizer placeholder</div>
              </div>
            </div>
          </div>

          {/* Bottom row: Constellation and Hypothesis */}
          <div className="bottom-grid">
            <div className="glass-card flex-grow-1">
              <div className="card-header-with-actions">
                <h3>Hypothesis Rankings</h3>
                <span className="badge-future">Phase 7</span>
              </div>
              <div className="hypothesis-table-placeholder">
                <div className="table-header">
                  <div>Modulation</div>
                  <div>Estimated Rate</div>
                  <div>Confidence</div>
                  <div>Status</div>
                </div>
                <div className="table-rows">
                  <div className="table-row-skeleton-item">
                    <div className="skele-cell wide">Waiting for signal...</div>
                    <div className="skele-cell"></div>
                    <div className="skele-cell"></div>
                    <div className="skele-cell"></div>
                  </div>
                </div>
              </div>
            </div>

            <div className="glass-card constellation-card">
              <div className="card-header-with-actions">
                <h3>Constellation Diagram</h3>
                <span className="badge-future">Phase 5</span>
              </div>
              <div className="visual-placeholder-graph circle-center">
                <div className="polar-grid"></div>
                <div className="polar-axis-y"></div>
                <div className="polar-axis-x"></div>
                <div className="graph-label-overlay">I/Q constellation plot</div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
};
