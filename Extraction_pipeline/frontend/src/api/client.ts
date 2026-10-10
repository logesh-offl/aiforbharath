import axios from 'axios';
import type {
  BatchResponse,
  ConfigResponse,
  InputType,
  ProcessResponse,
} from './types';

// Requests to /api/* and /health are proxied to the FastAPI backend by
// Vite's dev server config (see vite.config.ts), so no base URL/CORS
// setup is needed here.
const api = axios.create();

export async function fetchConfig(): Promise<ConfigResponse> {
  const { data } = await api.get<ConfigResponse>('/api/config');
  return data;
}

export async function processSingleFile(params: {
  file: File;
  inputType: InputType;
  instruction: string;
  maxTokens: number;
  modelDir: string;
}): Promise<ProcessResponse> {
  const form = new FormData();
  form.append('file', params.file);
  form.append('input_type', params.inputType);
  form.append('instruction', params.instruction);
  form.append('max_tokens', String(params.maxTokens));
  form.append('model_dir', params.modelDir);

  const { data } = await api.post<ProcessResponse>('/api/process', form);
  return data;
}

export async function processBatch(params: {
  files: File[];
  instruction: string;
  maxTokens: number;
  modelDir: string;
}): Promise<BatchResponse> {
  const form = new FormData();
  for (const file of params.files) {
    form.append('files', file);
  }
  form.append('instruction', params.instruction);
  form.append('max_tokens', String(params.maxTokens));
  form.append('model_dir', params.modelDir);

  const { data } = await api.post<BatchResponse>('/api/batch', form);
  return data;
}

export function extractErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return 'Something went wrong.';
}
