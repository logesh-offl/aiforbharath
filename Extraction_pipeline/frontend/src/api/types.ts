export type InputType = 'Image' | 'PDF';

export interface ConfigResponse {
  default_model_dir: string;
  supported_input_types: InputType[];
}

export interface ProcessResponse {
  filename: string;
  summary: string;
}

export interface BatchResultItem {
  filename: string;
  status: 'ok' | 'error';
  summary?: string;
  error?: string;
}

export interface BatchResponse {
  results: BatchResultItem[];
}

export interface ApiErrorPayload {
  detail?: string;
}
