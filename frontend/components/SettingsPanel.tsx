import { Cloud, Cpu, SlidersHorizontal, Sparkles } from "lucide-react";
import type { Mode } from "@/services/api";

const modes = [
  { value: "auto", title: "Auto", description: "Choose the best backend for each prompt.", icon: Sparkles },
  { value: "local", title: "Local Only", description: "Use Ollama with read-only GitHub PR tools.", icon: Cpu },
  { value: "cloud", title: "Cloud Only", description: "Use the cloud agent and its tools.", icon: Cloud },
] as const;

export default function SettingsPanel({ mode, setMode, temperature, setTemperature, disabled }: {
  mode: Mode; setMode: (mode: Mode) => void;
  temperature: number | null; setTemperature: (value: number | null) => void; disabled: boolean;
}) {
  return <section className="settings" aria-label="Chat settings">
    <div className="section-label"><SlidersHorizontal size={14} /> MODEL SETTINGS</div>
    <fieldset disabled={disabled} className="mode-field">
      <legend>Routing mode</legend>
      <div className="mode-switch">
        {modes.map(({ value, title, icon: Icon }) =>
          <label key={value} className={mode === value ? "selected" : ""}>
            <input type="radio" name="routing-mode" value={value} checked={mode === value}
              onChange={() => setMode(value)} />
            <Icon size={14} />{title}
          </label>
        )}
      </div>
      <p className="muted settings-help">{modes.find(item => item.value === mode)?.description}</p>
      <div className="temperature-row">
        <label htmlFor="temperature">Temperature</label>
        <label className="default-toggle"><input type="checkbox" checked={temperature === null}
          onChange={event => setTemperature(event.target.checked ? null : 0.7)} /> Model default</label>
      </div>
      <div className="range-row">
        <input id="temperature" type="range" min={0} max={2} step={0.1} value={temperature ?? 0.7}
          disabled={temperature === null} onChange={event => setTemperature(Number(event.target.value))} />
        <output htmlFor="temperature">{temperature === null ? "—" : temperature.toFixed(1)}</output>
      </div>
    </fieldset>
  </section>;
}
