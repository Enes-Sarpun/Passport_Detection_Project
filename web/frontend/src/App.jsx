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
import MrzEditor from './components/MrzEditor';
import SaveGate from './components/SaveGate';
import { extractFields } from './fields';

// Decorative MRZ text for the side strips (passport-themed ambience only).
const MRZ_DECOR = (
  'P<UTOERIKSSON<<ANNA<<<<<<<<<<<<<<<<<<<<<<<<<<' +
  'L898902C36UTO7408122F1204159ZE184226B<<<<<10'
).split('').join('\n');

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | scanning | done | error
  const [filename, setFilename] = useState('');
  const [durationMs, setDurationMs] = useState(null);
  const [data, setData] = useState(null);       // { result, preview }
  const [values, setValues] = useState({});     // kullanıcı düzeltmeleri
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);  // /api/save isteği sürüyor
  const [saveError, setSaveError] = useState('');
  const [scanFile, setScanFile] = useState(null); // save'de yeniden gönderilecek dosya
  const [mrzLines, setMrzLines] = useState([]);   // düzenlenebilir ham MRZ satırları
  const [error, setError] = useState('');
  const consoleSectionRef = useRef(null);

  const fields = useMemo(
    () => (data?.result ? extractFields(data.result) : []),
    [data]
  );

  // Kural A: zorunlu (mandatory) bir alan BOŞ ise VEYA model değeriyle AYNI
  // (düzeltilmemiş) ise "çözülmemiş" say — düzeltilmeden kaydedilemez.
  const unresolved = useMemo(() => {
    return fields.filter((f) => {
      if (!f.mandatory) return false;
      const v = (values[f.key] ?? f.value ?? '').trim();
      if (v === '') return true;
      // Kullanıcı değeri model çıktısıyla aynıysa düzeltme yapılmamış demektir.
      return v === String(f.value ?? '').trim();
    }).length;
  }, [fields, values]);

  // Hero CTA → konsol bölümüne animasyonlu (smooth) kaydır.
  function scrollToConsole() {
    consoleSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  }

  async function handleFile(file) {
    setFilename(file.name);
    setScanFile(file);
    setPhase('scanning');
    setSaved(false);
    setSaveError('');
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
      // Ham MRZ satırlarını düzenlenebilir başlangıç değeri olarak al.
      setMrzLines(Array.isArray(json?.result?.raw_mrz) ? [...json.result.raw_mrz] : []);
      setPhase('done');
    } catch (e) {
      setError(e.message || 'Operation failed');
      setPhase('error');
    }
  }

  // "Verify and Save" → görseli + model çıktısını + son alan değerlerini
  // /api/save'e gönderir. Görsel backend'e yeniden yüklenir (stateless).
  async function handleSave() {
    if (!data?.result || !scanFile || saving) return;
    setSaving(true);
    setSaveError('');
    try {
      // Her alanın son değeri: kullanıcı düzeltmesi varsa o, yoksa model değeri.
      const corrected = {};
      for (const f of fields) {
        corrected[f.key] = (values[f.key] ?? f.value ?? '').trim();
      }
      const body = new FormData();
      body.append('file', scanFile);
      body.append('model_output', JSON.stringify(data.result));
      body.append('corrected_fields', JSON.stringify(corrected));
      body.append('corrected_mrz', JSON.stringify(mrzLines));

      const scanUrl = import.meta.env.VITE_API_URL || '/api/scan';
      const saveUrl = import.meta.env.VITE_SAVE_URL || scanUrl.replace(/\/scan$/, '/save');
      const res = await fetch(saveUrl, { method: 'POST', body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Server error (${res.status})`);
      }
      setSaved(true);
    } catch (e) {
      setSaveError(e.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  }

  function reset() {
    setPhase('idle');
    setData(null);
    setValues({});
    setSaved(false);
    setSaving(false);
    setSaveError('');
    setScanFile(null);
    setMrzLines([]);
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
        {/* Dekoratif MRZ şeritleri — sol/sağ kenarda soluk dikey mono desen */}
        <div className="mrz-strip mrz-strip--left" aria-hidden="true">
          <div className="mrz-strip__scroll">{MRZ_DECOR}{MRZ_DECOR}</div>
        </div>
        <div className="mrz-strip mrz-strip--right" aria-hidden="true">
          <div className="mrz-strip__scroll">{MRZ_DECOR}{MRZ_DECOR}</div>
        </div>

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
              <MrzEditor
                lines={mrzLines}
                mrzFormat={data.result?.document?.mrz_format}
                onChange={(next) => {
                  setMrzLines(next);
                  setSaved(false);
                }}
              />
              <SaveGate
                unresolved={unresolved}
                saved={saved}
                saving={saving}
                saveError={saveError}
                onSave={handleSave}
              />
            </>
          )}
        </div>
      </section>
    </div>
  );
}
