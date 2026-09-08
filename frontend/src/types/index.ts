/**
 * Core type definitions for the SIGMA platform.
 */

export interface SignalMetadata {
  fileName: string;
  sampleRate: number;
  centerFrequency: number;
  duration: number;
  fileSize: number;
  ingestionTime: string;
}

export interface EstimatedParameters {
  snr: number;
  bandwidth: number;
  carrierOffset: number;
  symbolRate: number;
}

export interface HypothesisCandidate {
  id: string;
  modulation: string;
  symbolRate: number;
  /**
   * Normalized relative confidence across the current hypothesis candidate set.
   * Computed via temperature-scaled softmax. Not a calibrated Bayesian posterior probability.
   */
  confidenceScore: number;
  /**
   * Unnormalized composite evidence score dynamically weighted over available evidence dimensions.
   */
  rawScore?: number;
  details: string;
  status: 'pending' | 'success' | 'failed';
  fec_config?: string | null;
  interleaver_config?: string | null;
  sync_assumptions?: Record<string, any> | null;
  evidence?: Record<string, any>;
}
