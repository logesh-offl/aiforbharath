import type { InputType } from '../api/types';

const IMAGE_EXTENSIONS = new Set([
  'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp',
]);
const PDF_EXTENSIONS = new Set(['pdf']);

/** Mirrors pipeline.infer_input_type() on the backend: guesses Image
 * vs PDF from the filename extension. Returns null for unrecognized
 * types (caller should flag/skip rather than guess wrong). */
export function inferInputType(filename: string): InputType | null {
  const ext = filename.split('.').pop()?.toLowerCase() ?? '';
  if (IMAGE_EXTENSIONS.has(ext)) return 'Image';
  if (PDF_EXTENSIONS.has(ext)) return 'PDF';
  return null;
}

export const ACCEPTED_EXTENSIONS = [
  ...IMAGE_EXTENSIONS,
  ...PDF_EXTENSIONS,
].map((ext) => `.${ext}`);
