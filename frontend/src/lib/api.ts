export const API_BASE: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';

export async function documentError(response: Response, fallback: string): Promise<string> {
  if (response.status !== 422 && response.status !== 503) return fallback;
  const data = await response.json().catch(() => null);
  return typeof data?.detail === 'string' ? data.detail : fallback;
}
