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

/** Units: snr in dB, bandwidth/carrierOffset in Hz, symbolRate in sps. */
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

export interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

export interface ConstellationPoint {
  i: number;
  q: number;
}

export type PipelineStageStatus = 'complete' | 'active' | 'pending';

export interface PipelineStage {
  label: string;
  status: PipelineStageStatus;
}

export interface AnalyzedSignal {
  metadata: SignalMetadata;
  spectrum: SpectrumPoint[];
  waterfall: number[][];
  parameters: EstimatedParameters;
  constellation: ConstellationPoint[];
  hypotheses: HypothesisCandidate[];
  pipeline: PipelineStage[];
}
