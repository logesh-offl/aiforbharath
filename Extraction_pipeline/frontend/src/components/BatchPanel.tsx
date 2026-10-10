import { useState } from 'react';
import { CheckCircle2, Layers, Loader2, XCircle } from 'lucide-react';
import { extractErrorMessage, processBatch } from '../api/client';
import type { BatchResultItem } from '../api/types';
import { ACCEPTED_EXTENSIONS } from '../utils/fileType';
import InstructionInput from './InstructionInput';

interface BatchPanelProps {
  modelDir: string;
  maxTokens: number;
}

export default function BatchPanel({ modelDir, maxTokens }: BatchPanelProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [instruction, setInstruction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BatchResultItem[] | null>(null);

  async function handleRun() {
    if (files.length === 0) {
      setError('Please choose at least one file first.');
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
    setResults(null);
    try {
      const response = await processBatch({
        files, instruction, maxTokens, modelDir,
      });
      setResults(response.results);
    } catch (err) {
      setError(extractErrorMessage(err));
    } finally {
      setLoading(false);
    }
  }

  function downloadResults() {
    if (!results) return;
    const blob = new Blob([JSON.stringify(results, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'batch_results.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  const okCount = results?.filter((r) => r.status === 'ok').length ?? 0;
  const errorCount = (results?.length ?? 0) - okCount;

  return (
    <div className="panel">
      <p className="panel-caption">
        Upload multiple images/PDFs. Each file's type is detected from
        its extension automatically. edge0 processes one file at a time
        by design, so files run sequentially, not in parallel.
      </p>

      <div className="field">
        <label htmlFor="batch-files">Upload files for batch processing</label>
        <input
          id="batch-files"
          type="file"
          multiple
          accept={ACCEPTED_EXTENSIONS.join(',')}
          onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
        />
        {files.length > 0 && (
          <p className="hint">{files.length} file(s) selected</p>
        )}
      </div>

      <InstructionInput
        value={instruction}
        onChange={setInstruction}
        label="Instruction (optional, applied to every file in the batch)"
      />

      <button className="primary-button" onClick={handleRun} disabled={loading}>
        {loading ? <Loader2 className="spin" size={16} /> : <Layers size={16} />}
        {loading ? 'Processing batch...' : 'Run Batch'}
      </button>

      {error && <div className="alert alert-error">{error}</div>}

      {results && (
        <div className="batch-results">
          <div className="batch-summary">
            <h3>
              Batch results: {okCount} succeeded, {errorCount} failed
            </h3>
            <button className="secondary-button" onClick={downloadResults}>
              Download all results (JSON)
            </button>
          </div>

          {results.map((r) => (
            <details key={r.filename} className="result-item">
              <summary>
                {r.status === 'ok' ? (
                  <CheckCircle2 size={16} className="icon-ok" />
                ) : (
                  <XCircle size={16} className="icon-error" />
                )}
                {r.filename}
              </summary>
              <div className="result-item-body">
                {r.status === 'ok' ? (
                  <p>{r.summary || 'No text could be extracted from this file.'}</p>
                ) : (
                  <p className="alert-error">{r.error}</p>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
    </div>
  );
}
