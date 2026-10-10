interface SettingsPanelProps {
  modelDir: string;
  onModelDirChange: (value: string) => void;
  maxTokens: number;
  onMaxTokensChange: (value: number) => void;
}

export default function SettingsPanel({
  modelDir,
  onModelDirChange,
  maxTokens,
  onMaxTokensChange,
}: SettingsPanelProps) {
  return (
    <aside className="sidebar">
      <h2>Settings</h2>

      <div className="field">
        <label htmlFor="model-dir">edge0 model directory</label>
        <input
          id="model-dir"
          type="text"
          value={modelDir}
          onChange={(e) => onModelDirChange(e.target.value)}
        />
        <p className="hint">
          Defaults to the EDGE0_35B_MODEL environment variable on the
          backend, if set.
        </p>
      </div>

      <div className="field">
        <label htmlFor="max-tokens">
          Max output tokens: {maxTokens}
        </label>
        <input
          id="max-tokens"
          type="range"
          min={32}
          max={8192}
          step={64}
          value={maxTokens}
          onChange={(e) => onMaxTokensChange(Number(e.target.value))}
        />
        <p className="hint">
          edge0-35b's context window is 262,144 tokens (input + output
          combined); this cap just keeps the slider usable. Larger
          values take proportionally longer to generate.
        </p>
      </div>
    </aside>
  );
}
