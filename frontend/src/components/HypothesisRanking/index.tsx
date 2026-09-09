import React, { useState } from 'react';
import type { HypothesisCandidate, EvidenceComponent } from '../../types';

interface HypothesisRankingProps {
  hypotheses?: HypothesisCandidate[];
  isLoading?: boolean;
}

// Sample fallback hypotheses for demo when no live signal is processing
const DEMO_HYPOTHESES: HypothesisCandidate[] = [
  {
    id: 'hyp_qpsk_9600_top',
    modulation: 'QPSK',
    symbolRate: 9600,
    rawScore: 0.842,
    confidenceScore: 0.728,
    details: 'Rank #1: QPSK at 9600 Baud. Strong ML and symbol-rate agreement.',
    status: 'success',
    fec_config: 'none',
    evidence: {
      ml: { status: 'available', score: 0.86, details: { probability: 0.86 } },
      symbolRate: { status: 'available', score: 0.94, details: { estimated_baud: 9587, uncertainty: 45 } },
      timing: { status: 'available', score: 0.81, details: { peak_prominence: 0.81 } },
      constellation: { status: 'available', score: 0.90, details: { evm: 0.08 } },
      snr: { status: 'not_evaluated', details: { estimated_snr_db: 22.5, status: 'telemetry_only' } },
      fec: { status: 'not_evaluated' },
      bitstream: { status: 'not_evaluated' },
    },
  },
  {
    id: 'hyp_bpsk_9600_comp',
    modulation: 'BPSK',
    symbolRate: 9600,
    rawScore: 0.612,
    confidenceScore: 0.186,
    details: 'Rank #2: BPSK at 9600 Baud. Moderate ML probability, symbol rate agrees.',
    status: 'pending',
    fec_config: 'none',
    evidence: {
      ml: { status: 'available', score: 0.38, details: { probability: 0.38 } },
      symbolRate: { status: 'available', score: 0.94, details: { estimated_baud: 9587, uncertainty: 45 } },
      timing: { status: 'available', score: 0.78 },
      constellation: { status: 'available', score: 0.62 },
      snr: { status: 'not_evaluated', details: { estimated_snr_db: 22.5 } },
      fec: { status: 'not_evaluated' },
      bitstream: { status: 'not_evaluated' },
    },
  },
  {
    id: 'hyp_8psk_9600_comp',
    modulation: '8PSK',
    symbolRate: 9600,
    rawScore: 0.435,
    confidenceScore: 0.086,
    details: 'Rank #3: 8PSK at 9600 Baud. Low ML confidence.',
    status: 'pending',
    fec_config: 'none',
    evidence: {
      ml: { status: 'available', score: 0.14, details: { probability: 0.14 } },
      symbolRate: { status: 'available', score: 0.94 },
      timing: { status: 'not_evaluated' },
      constellation: { status: 'available', score: 0.45 },
      snr: { status: 'not_evaluated', details: { estimated_snr_db: 22.5 } },
      fec: { status: 'not_evaluated' },
      bitstream: { status: 'not_evaluated' },
    },
  },
];

export const HypothesisRanking: React.FC<HypothesisRankingProps> = ({
  hypotheses = DEMO_HYPOTHESES,
  isLoading = false,
}) => {
  const [expandedId, setExpandedId] = useState<string | null>(hypotheses[0]?.id || null);

  const toggleExpand = (id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  };

  const renderEvidenceBar = (label: string, component?: EvidenceComponent) => {
    const status = component?.status || 'not_evaluated';
    const score = component?.score !== undefined ? component.score : null;
    const pct = score !== null ? Math.round(score * 100) : 0;

    let statusClass = 'status-not-eval';
    let statusLabel = 'NOT EVALUATED';
    let barColor = 'var(--text-muted)';

    if (status === 'available') {
      statusClass = 'status-available';
      statusLabel = `${pct}%`;
      barColor = pct >= 80 ? 'var(--accent-cyan)' : pct >= 50 ? 'var(--accent-purple)' : 'var(--accent-amber)';
    } else if (status === 'failed') {
      statusClass = 'status-failed';
      statusLabel = 'FAILED (0%)';
      barColor = 'var(--accent-red)';
    } else if (status === 'not_supported') {
      statusClass = 'status-not-supported';
      statusLabel = 'NOT SUPPORTED';
    }

    return (
      <div className="evidence-dimension-row" key={label}>
        <div className="evidence-dimension-label">{label}</div>
        <div className="evidence-bar-track">
          {status === 'available' && (
            <div
              className="evidence-bar-fill"
              style={{ width: `${Math.max(4, pct)}%`, backgroundColor: barColor }}
            />
          )}
          {status === 'failed' && (
            <div className="evidence-bar-fill failed" style={{ width: '100%' }} />
          )}
        </div>
        <div className={`evidence-status-pill ${statusClass}`}>{statusLabel}</div>
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="hypothesis-loading-state">
        <div className="telemetry-spinner"></div>
        <span>Evaluating hypothesis candidate evidence...</span>
      </div>
    );
  }

  return (
    <div className="hypothesis-ranking-container">
      <div className="hypothesis-list-header">
        <div>Rank & Modulation</div>
        <div>Estimated Rate</div>
        <div>Relative Confidence</div>
        <div>Action</div>
      </div>

      <div className="hypothesis-cards-list">
        {hypotheses.map((cand, idx) => {
          const isExpanded = expandedId === cand.id;
          const confPct = Math.round(cand.confidenceScore * 100);
          const ev = cand.evidence || {};

          return (
            <div
              className={`hypothesis-card ${isExpanded ? 'expanded' : ''} ${cand.status}`}
              key={cand.id}
            >
              <div className="hypothesis-card-summary" onClick={() => toggleExpand(cand.id)}>
                <div className="mod-info">
                  <span className="rank-badge">#{idx + 1}</span>
                  <span className="mod-title">{cand.modulation}</span>
                  {cand.fec_config && cand.fec_config !== 'none' && (
                    <span className="fec-badge">+{cand.fec_config}</span>
                  )}
                </div>

                <div className="baud-info">
                  <span className="baud-value">
                    {cand.symbolRate >= 1000
                      ? `${(cand.symbolRate / 1000).toFixed(1)}k`
                      : cand.symbolRate}{' '}
                    Baud
                  </span>
                </div>

                <div className="confidence-info">
                  <div className="confidence-text-row">
                    <span className="conf-percent">{confPct}%</span>
                    {cand.rawScore !== undefined && (
                      <span className="raw-score-tag">Raw: {cand.rawScore.toFixed(3)}</span>
                    )}
                  </div>
                  <div className="confidence-progress-track">
                    <div
                      className="confidence-progress-fill"
                      style={{
                        width: `${Math.max(5, confPct)}%`,
                        background:
                          idx === 0
                            ? 'linear-gradient(90deg, #00ffcc, #00b4d8)'
                            : 'linear-gradient(90deg, #8a2be2, #6a0dad)',
                      }}
                    />
                  </div>
                </div>

                <div className="details-toggle">
                  <button className="btn-details-toggle">
                    {isExpanded ? '▲ Hide' : '▼ Evidence'}
                  </button>
                </div>
              </div>

              {isExpanded && (
                <div className="hypothesis-evidence-drawer">
                  <div className="evidence-trace-header">
                    <span>Evidence Trace (Normalized Sum: {cand.rawScore?.toFixed(4)})</span>
                    <span className="details-hint">{cand.details}</span>
                  </div>

                  <div className="evidence-dimensions-grid">
                    {renderEvidenceBar('ML Modulation', ev.ml || ev.mlModulation)}
                    {renderEvidenceBar('Symbol Rate', ev.symbolRate || ev.symbol_rate)}
                    {renderEvidenceBar('Timing / Sync', ev.timing)}
                    {renderEvidenceBar('Constellation', ev.constellation)}
                    {renderEvidenceBar('SNR Feasibility', ev.snr)}
                    {renderEvidenceBar('FEC Validation', ev.fec)}
                    {renderEvidenceBar('Bitstream Correlation', ev.bitstream)}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
