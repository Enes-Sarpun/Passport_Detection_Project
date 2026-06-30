// Passport Detection — Trust Console.
// Single-page scroll layout: Hero (tam ekran, üst) → aşağı kaydır → Konsol bölümü.
// "taramaya başla" → #console-section'a smooth scroll.
// Faz geçişleri artık sadece konsol bölümü içinde (upload → scan → done).
import { useMemo, useRef, useState } from 'react';
import './App.css';
import HeroLanding from './components/HeroLanding';
import UploadZone from './components/UploadZone';
import ScanConsole from './components/ScanConsole';
import ReliabilityChart from './components/ReliabilityChart';
import FieldTable from './components/FieldTable';
import SaveGate from './components/SaveGate';
import { extractFields } from './fields';

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | scanning | done | error
  const [filename, setFilename] = useState('');
  const [durationMs, setDurationMs] = useState(null);
  const [data, setData] = useState(null);       // { result, preview }
  const [values, setValues] = useState({});     // kullanıcı düzeltmeleri
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState('');
  const consoleSectionRef = useRef(null);

  const fields = useMemo(
    () => (data?.result ? extractFields(data.result) : []),
    [data]
  );

  // Zorunlu alanlardan dolu olmayanları say (save gate canlı engeli).
  const unresolved = useMemo(() => {
    return fields.filter((f) => {
      if (!f.mandatory) return false;
      const v = (values[f.key] ?? f.value ?? '').trim();
      return v === '';
    }).length;
  }, [fields, values]);

  // Hero CTA → konsol bölümüne animasyonlu (smooth) kaydır.
  function scrollToConsole() {
    consoleSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  }

  async function handleFile(file) {
    setFilename(file.name);
    setPhase('scanning');
    setSaved(false);
    setValues({});
    setError('');
    const started = performance.now();
    try {
      const body = new FormData();
      body.append('file', file);
      const apiUrl = import.meta.env.VITE_API_URL || '/api/scan';
      const res = await fetch(apiUrl, { method: 'POST', body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error (${res.status})`);
      }
      const json = await res.json();
      const elapsed = performance.now() - started;
      const minShow = 1600;
      if (elapsed < minShow) await new Promise((r) => setTimeout(r, minShow - elapsed));
      setDurationMs(Math.round(performance.now() - started));
      setData(json);
      setPhase('done');
    } catch (e) {
      setError(e.message || 'Operation failed');
      setPhase('error');
    }
  }

  function reset() {
    setPhase('idle');
    setData(null);
    setValues({});
    setSaved(false);
    setError('');
    setDurationMs(null);
    setFilename('');
  }

  const noMrz = data?.result?.status && !data?.result?.document;

  return (
    <div className="app-root">
      {/* ── HERO — tam ekran, sayfa başı ── */}
      <HeroLanding onStart={scrollToConsole} />

      {/* ── KONSOL BÖLÜMİ — kaydırınca görünür ── */}
      <section
        id="console-section"
        ref={consoleSectionRef}
        className="console-section"
        aria-label="MRZ Scan Console"
      >
        {/* Bölüm başlığı + eylem butonu */}
        <div className="console-header">
          <div className="console-header__brand">
            <span className="topbar__mark" />
            <span className="topbar__name">Passport OCR-YOLO</span>
            <span className="topbar__sub label">Trust Console</span>
          </div>
          {phase !== 'idle' && (
            <button className="btn btn--secondary" onClick={reset}>
              New Scan
            </button>
          )}
        </div>

        <div className="console-body">
          {/* Adım 1 — Yükle */}
          {phase === 'idle' && <UploadZone onFile={handleFile} />}

          {/* Adım 2 — Tarama konsolu */}
          {(phase === 'scanning' || phase === 'done') && (
            <ScanConsole
              scanning={phase === 'scanning'}
              done={phase === 'done'}
              filename={filename}
              durationMs={durationMs}
            />
          )}

          {phase === 'error' && (
            <div className="errorbox">
              <strong>Operation failed:</strong> {error}
              <button className="btn btn--secondary" onClick={reset} style={{ marginLeft: 12 }}>
                Try again
              </button>
            </div>
          )}

          {phase === 'done' && noMrz && (
            <div className="errorbox">
              No MRZ region could be detected or parsed in this image.
              <button className="btn btn--secondary" onClick={reset} style={{ marginLeft: 12 }}>
                New image
              </button>
            </div>
          )}

          {/* Adımlar 3–6 — Sonuçlar */}
          {phase === 'done' && !noMrz && (
            <>
              {data.preview && (
                <div className="preview">
                  <img src={data.preview} alt="Detected MRZ" />
                </div>
              )}
              <ReliabilityChart fields={fields} />
              <FieldTable
                fields={fields}
                values={values}
                onChange={(k, v) => {
                  setValues((p) => ({ ...p, [k]: v }));
                  setSaved(false);
                }}
              />
              <SaveGate
                unresolved={unresolved}
                saved={saved}
                onSave={() => setSaved(true)}
              />
            </>
          )}
        </div>
      </section>
    </div>
  );
}
