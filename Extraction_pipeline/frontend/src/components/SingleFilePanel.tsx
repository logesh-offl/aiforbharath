import { useState } from 'react';
import { FileUp, Loader2 } from 'lucide-react';
import { extractErrorMessage, processSingleFile } from '../api/client';
import type { InputType } from '../api/types';
import InstructionInput from './InstructionInput';

interface SingleFilePanelProps {
  modelDir: string;
  maxTokens: number;
}

export default function SingleFilePanel({
  modelDir,
  maxTokens,
}: SingleFilePanelProps) {
  const [inputType, setInputType] = useState<InputType>('Image');
  const [file, setFile] = useState<File | null>(null);
  const [instruction, setInstruction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);

  const accept = inputType === 'Image'
    ? '.png,.jpg,.jpeg,.bmp,.tiff,.tif,.webp'
    : '.pdf';

  async function handleRun() {
    if (!file) {
      setError('Please choose a file first.');
      return;
    }
    if (!modelDir) {
      setError(
        'No edge0 model directory configured. Set it in Settings or export EDGE0_35B_MODEL.',
      );
      return;
    }

    setLoading(true);
    setError(null);
    setSummary(null);
    try {
      const result = await processSingleFile({
        file, inputType, instruction, maxTokens, modelDir,
      });
      setSummary(result.summary);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <div className="field">
        <label htmlFor="input-type">Input type</label>
        <select
          id="input-type"
          value={inputType}
          onChange={(e) => {
            setInputType(e.target.value as InputType);
            setFile(null);
          }}
        >
          <option value="Image">Image</option>
          <option value="PDF">PDF</option>
        </select>
      </div>

      <div className="field">
        <label htmlFor="single-file">
          Upload {inputType === 'Image' ? 'an image' : 'a PDF'}
        </label>
        <input
          id="single-file"
          type="file"
          accept={accept}
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>

      <InstructionInput value={instruction} onChange={setInstruction} />

      <button className="primary-button" onClick={handleRun} disabled={loading}>
        {loading ? <Loader2 className="spin" size={16} /> : <FileUp size={16} />}
        {loading ? 'Extracting & running edge0...' : 'Extract & Run'}
      </button>

      {error && <div className="alert alert-error">{error}</div>}

      {summary !== null && (
        <div className="result-card">
          <h3>{instruction.trim() ? 'Result' : 'Summary'}</h3>
          <p>{summary || 'No text could be extracted from this file.'}</p>
        </div>
      )}
    </div>
  );
}
