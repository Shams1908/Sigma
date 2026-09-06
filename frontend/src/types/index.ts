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
  confidenceScore: number;
  details: string;
  status: 'pending' | 'success' | 'failed';
}
