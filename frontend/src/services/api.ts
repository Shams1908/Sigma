const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/health';

export interface HealthCheckResponse {
  status: string;
  project: string;
}

/**
 * Fetch backend health status
 */
export async function checkBackendHealth(): Promise<HealthCheckResponse> {
  const response = await fetch(API_BASE_URL);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return response.json();
}
