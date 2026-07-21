// Passport Detection — Trust Console.
// Single-page layout: full-screen hero, then the console section (upload → scan → done).
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

// Two MRZ lines for the side strips — each rendered as a vertical column of
// characters, the two columns side by side (passport-themed ambience only).
const MRZ_DECOR_L1 = 'P<UTOERIKSSON<<ANNA<<<<<<<<<<<<<<<<<<<<<<<<<<';
const MRZ_DECOR_L2 = 'L898902C36UTO7408122F1204159ZE184226B<<<<<10';
const toColumn = (s) => s.split('').join('\n');

export default function App() {
  const [phase, setPhase] = useState('idle'); // idle | scanning | done | error
  const [filename, setFilename] = useState('');
  const [durationMs, setDurationMs] = useState(null);
  const [data, setData] = useState(null);       // { result, preview }
  const [values, setValues] = useState({});     // user edits
  const [confirmed, setConfirmed] = useState({}); // per-field "value is correct" flags
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState('');
  const [scanFile, setScanFile] = useState(null); // re-sent on save
  const [mrzLines, setMrzLines] = useState([]);   // editable raw MRZ lines
  const [error, setError] = useState('');
  const consoleSectionRef = useRef(null);

  const fields = useMemo(
    () => (data?.result ? extractFields(data.result) : []),
    [data]
  );

  // A mandatory field counts as resolved when it is filled AND (its value was
  // edited OR the user confirmed the model's read is correct). Without the
  // confirm path, a correctly-read low-score field could never be saved.
  const unresolved = useMemo(() => {
    return fields.filter((f) => {
      if (!f.mandatory) return false;
      const v = (values[f.key] ?? f.value ?? '').trim();
      if (v === '') return true;                  // empty → unresolved
      if (confirmed[f.key]) return false;         // confirmed → resolved
      return v === String(f.value ?? '').trim();  // unchanged → unresolved
    }).length;
  }, [fields, values, confirmed]);

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
      setMrzLines(Array.isArray(json?.result?.raw_mrz) ? [...json.result.raw_mrz] : []);
      setPhase('done');
    } catch (e) {
      setError(e.message || 'Operation failed');
      setPhase('error');
    }
  }

  // Sends the image, the model output and the final field values to /api/save.
  // The image is re-uploaded because the backend is stateless.
  async function handleSave() {
    if (!data?.result || !scanFile || saving) return;
    setSaving(true);
    setSaveError('');
    try {
      // Final value per field: the user's edit if present, otherwise the model's.
      const corrected = {};
      for (const f of fields) {
        corrected[f.key] = (values[f.key] ?? f.value ?? '').trim();
      }
      const confirmedKeys = Object.keys(confirmed).filter((k) => confirmed[k]);

      const body = new FormData();
      body.append('file', scanFile);
      body.append('model_output', JSON.stringify(data.result));
      body.append('corrected_fields', JSON.stringify(corrected));
      body.append('confirmed_fields', JSON.stringify(confirmedKeys));
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
    setConfirmed({});
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
      <HeroLanding onStart={scrollToConsole} />

      <section
        id="console-section"
        ref={consoleSectionRef}
        className="console-section"
        aria-label="MRZ Scan Console"
      >
        {/* Decorative MRZ strips on the left/right edges */}
        <div className="mrz-strip mrz-strip--left" aria-hidden="true">
          <div className="mrz-strip__scroll">
            <span className="mrz-strip__col">{toColumn(MRZ_DECOR_L1 + MRZ_DECOR_L1)}</span>
            <span className="mrz-strip__col">{toColumn(MRZ_DECOR_L2 + MRZ_DECOR_L2)}</span>
          </div>
        </div>
        <div className="mrz-strip mrz-strip--right" aria-hidden="true">
          <div className="mrz-strip__scroll">
            <span className="mrz-strip__col">{toColumn(MRZ_DECOR_L1 + MRZ_DECOR_L1)}</span>
            <span className="mrz-strip__col">{toColumn(MRZ_DECOR_L2 + MRZ_DECOR_L2)}</span>
          </div>
        </div>

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
          {phase === 'idle' && <UploadZone onFile={handleFile} />}

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
                confirmed={confirmed}
                onChange={(k, v) => {
                  setValues((p) => ({ ...p, [k]: v }));
                  // Editing a value invalidates a previous "correct" confirmation.
                  setConfirmed((p) => (p[k] ? { ...p, [k]: false } : p));
                  setSaved(false);
                }}
                onConfirm={(k, ok) => {
                  setConfirmed((p) => ({ ...p, [k]: ok }));
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
