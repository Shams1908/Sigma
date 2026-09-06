// API client for SIGMA backend communication
// Backend runs on localhost:8000 (FastAPI)

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const API_V1 = `${API_BASE}/api/v1`;

// Type definitions matching backend schemas
export interface UploadResponse {
  signal_id: string;
  metadata: {
    fileName: string;
    sampleRate: number;
    centerFrequency: number;
    duration: number;
    fileSize: number;
    ingestionTime: string;
  };
}

export interface SpectrumPoint {
  frequency: number;
  magnitudeDb: number;
}

export interface SpectrumResponse {
  data: SpectrumPoint[];
  sampleRate: number;
  centerFrequency: number;
}

export interface WaterfallResponse {
  data: number[][];
  timeRange: [number, number];
  frequencyRange: [number, number];
}

export interface ConstellationPoint {
  i: number;
  q: number;
}

export interface ConstellationResponse {
  data: ConstellationPoint[];
  modulation: string | null;
}

export interface SignalParameters {
  carrierFrequency: number;
  sampleRate: number;
  bandwidth: number;
  snr: number;
  symbolRate: number;
}

export interface HypothesisValidation {
  syncPassed: boolean;
  demodPassed: boolean;
  fecPassed: boolean;
}

export interface Hypothesis {
  id: string;
  modulation: string;
  symbolRate: number;
  mlConfidence: number;
  validation: HypothesisValidation;
  isWinner: boolean;
  evidence?: {
    syncMethod?: string;
    demodConfig?: Record<string, any>;
    fecParams?: Record<string, any>;
  };
}

export interface HypothesesResponse {
  hypotheses: Hypothesis[];
  totalTested: number;
  winnerCount: number;
}

export interface AnalysisStatus {
  analysisId: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStage?: string;
  error?: string;
}

// Error handling
class APIError extends Error {
  constructor(
    message: string,
    public status?: number,
    public data?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

// Helper function for fetch with error handling
async function fetchJSON<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  try {
    const response = await fetch(`${API_V1}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.message || `HTTP ${response.status}: ${response.statusText}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

// API Methods

/**
 * Upload a signal file (.iq or .wav) for analysis
 */
export async function uploadFile(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(`${API_V1}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.message || 'File upload failed',
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      `Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`
    );
  }
}

/**
 * Start analysis for an uploaded signal
 */
export async function startAnalysis(signalId: string): Promise<{ analysis_id: string; status: string }> {
  return fetchJSON(`/analysis/${signalId}`, {
    method: 'POST',
  });
}

/**
 * Get analysis status and progress
 */
export async function getAnalysisStatus(analysisId: string): Promise<AnalysisStatus> {
  return fetchJSON(`/analysis/${analysisId}/status`);
}

/**
 * Get FFT spectrum data (real DSP computation from IQ file)
 */
export async function getSpectrum(signalId: string): Promise<SpectrumResponse> {
  return fetchJSON(`/visualizations/${signalId}/spectrum`);
}

/**
 * Get spectrogram/waterfall data (real DSP computation from IQ file)
 */
export async function getWaterfall(signalId: string): Promise<WaterfallResponse> {
  return fetchJSON(`/visualizations/${signalId}/waterfall`);
}

/**
 * Get IQ constellation diagram data (real IQ samples from file)
 */
export async function getConstellation(signalId: string): Promise<ConstellationResponse> {
  return fetchJSON(`/visualizations/${signalId}/constellation`);
}

/**
 * Get signal parameters (carrier frequency, sample rate, etc.) - REAL DSP analysis
 */
export async function getSignalParameters(signalId: string): Promise<SignalParameters> {
  return fetchJSON(`/visualizations/${signalId}/parameters`);
}

/**
 * Get hypothesis validation results
 */
export async function getHypotheses(analysisId: string): Promise<HypothesesResponse> {
  return fetchJSON(`/results/${analysisId}/hypotheses`);
}

/**
 * Get complete analysis results (all data in one call)
 */
export async function getResults(analysisId: string): Promise<{
  spectrum: SpectrumResponse;
  waterfall: WaterfallResponse;
  constellation: ConstellationResponse;
  parameters: SignalParameters;
  hypotheses: HypothesesResponse;
}> {
  return fetchJSON(`/results/${analysisId}`);
}

/**
 * Poll analysis status until completion or timeout
 * @param analysisId Analysis ID to poll
 * @param onProgress Callback for progress updates
 * @param timeout Maximum time to wait in milliseconds (default: 5 minutes)
 * @param interval Polling interval in milliseconds (default: 1 second)
 */
export async function pollAnalysisStatus(
  analysisId: string,
  onProgress?: (status: AnalysisStatus) => void,
  timeout: number = 5 * 60 * 1000,
  interval: number = 1000
): Promise<AnalysisStatus> {
  const startTime = Date.now();

  while (true) {
    const status = await getAnalysisStatus(analysisId);
    
    if (onProgress) {
      onProgress(status);
    }

    if (status.status === 'completed' || status.status === 'failed') {
      return status;
    }

    if (Date.now() - startTime > timeout) {
      throw new APIError('Analysis timeout exceeded');
    }

    await new Promise(resolve => setTimeout(resolve, interval));
  }
}

/**
 * Upload file and automatically start analysis with progress tracking
 */
export async function uploadAndAnalyze(
  file: File,
  onProgress?: (status: AnalysisStatus) => void
): Promise<{
  analysisId: string;
  results: Awaited<ReturnType<typeof getResults>>;
}> {
  // Upload file
  const uploadResponse = await uploadFile(file);
  const { analysisId } = uploadResponse;

  // Start analysis
  await startAnalysis(analysisId);

  // Poll until complete
  const finalStatus = await pollAnalysisStatus(analysisId, onProgress);

  if (finalStatus.status === 'failed') {
    throw new APIError(
      `Analysis failed: ${finalStatus.error || 'Unknown error'}`,
      500,
      finalStatus
    );
  }

  // Get results
  const results = await getResults(analysisId);

  return { analysisId, results };
}

// Export error class for external use
export { APIError };

// Health check
export async function healthCheck(): Promise<{ status: string; version?: string }> {
  return fetch(`${API_BASE}/health`).then(r => r.json());
}


export async function getWaveform(signalId: string): Promise<{
  i: number[];
  q: number[];
  time: number[];
  sampleRate: number;
}> {
  return fetchJSON(`/visualizations/${signalId}/waveform`);
}

export async function getDiagnostics(analysisId: string): Promise<{
  evm_rms: number | null;
  timing_error_rms: number | null;
  sync_locked: boolean;
  demod_locked: boolean;
  fec_valid: boolean;
  snr: number;
  carrier_offset: number;
}> {
  return fetchJSON(`/results/${analysisId}/diagnostics`);
}

export async function getBitstream(analysisId: string): Promise<{
  available: boolean;
  bits?: string;
  total_bits?: number;
  entropy?: number;
  ones_ratio?: number;
  reason?: string;
}> {
  return fetchJSON(`/results/${analysisId}/bitstream`);
}
