import { useEffect, useState } from 'react';
import { FileText } from 'lucide-react';
import { fetchConfig } from './api/client';
import SettingsPanel from './components/SettingsPanel';
import SingleFilePanel from './components/SingleFilePanel';
import BatchPanel from './components/BatchPanel';

type Mode = 'single' | 'batch';

export default function App() {
  const [modelDir, setModelDir] = useState('');
  const [maxTokens, setMaxTokens] = useState(1024);
  const [mode, setMode] = useState<Mode>('single');
  const [configError, setConfigError] = useState<string | null>(null);

  useEffect(() => {
    fetchConfig()
      .then((cfg) => setModelDir(cfg.default_model_dir))
      .catch(() =>
        setConfigError(
          'Could not reach the DocLens backend. Is it running on port 8000?',
        ),
      );
  }, []);

  return (
    <div className="app-shell">
      <SettingsPanel
        modelDir={modelDir}
        onModelDirChange={setModelDir}
        maxTokens={maxTokens}
        onMaxTokensChange={setMaxTokens}
      />

      <main className="main-content">
        <header className="app-header">
          <FileText size={28} />
          <div>
            <h1>DocLens</h1>
            <p>
              Upload an image or PDF, extract its text with EasyOCR, and
              get a summary — or a custom instruction result — from the
              local edge0 model.
            </p>
          </div>
        </header>

        {configError && <div className="alert alert-error">{configError}</div>}

        <div className="mode-toggle">
          <button
            className={mode === 'single' ? 'active' : ''}
            onClick={() => setMode('single')}
          >
            Single file
          </button>
          <button
            className={mode === 'batch' ? 'active' : ''}
            onClick={() => setMode('batch')}
          >
            Batch (multiple files)
          </button>
        </div>

        {mode === 'single' ? (
          <SingleFilePanel modelDir={modelDir} maxTokens={maxTokens} />
        ) : (
          <BatchPanel modelDir={modelDir} maxTokens={maxTokens} />
        )}
      </main>
    </div>
  );
}
