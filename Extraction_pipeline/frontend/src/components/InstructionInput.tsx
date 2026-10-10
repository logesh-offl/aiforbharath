interface InstructionInputProps {
  value: string;
  onChange: (value: string) => void;
  label?: string;
}

export default function InstructionInput({
  value,
  onChange,
  label = 'Instruction (optional)',
}: InstructionInputProps) {
  return (
    <div className="field">
      <label htmlFor="instruction">{label}</label>
      <input
        id="instruction"
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="e.g. extract only dates and amounts / translate to English and summarize / list action items"
      />
      <p className="hint">
        Leave blank for the default summary, or tell edge0 exactly what
        to do with the extracted text.
      </p>
    </div>
  );
}
