// The dark "instrument" surface (DESIGN.md §4 + §6). While scanning it shows a
// streaming terminal log with a blinking caret; on completion it collapses to a
// thin status strip (filename · scan duration). The one elevated, dark element.
import { useEffect, useRef, useState } from 'react';

const SCAN_STEPS = [
  'YOLO ile MRZ bölgesi tespit ediliyor…',
  'MRZ bölgesi kırpılıyor ve hizalanıyor…',
  'OCR-B ile satır 1 okunuyor…',
  'OCR-B ile satır 2 okunuyor…',
  'ICAO 9303 alanları çözümleniyor…',
  'Kontrol haneleri doğrulanıyor…',
  'Güvenilirlik skoru hesaplanıyor…',
];

export default function ScanConsole({ scanning, done, filename, durationMs }) {
  const [lines, setLines] = useState([]);
  const logRef = useRef(null);

  useEffect(() => {
    if (!scanning) return;
    setLines([]);
    let i = 0;
    const id = setInterval(() => {
      setLines((prev) => (i < SCAN_STEPS.length ? [...prev, SCAN_STEPS[i++]] : prev));
      if (i >= SCAN_STEPS.length) clearInterval(id);
    }, 260);
    return () => clearInterval(id);
  }, [scanning]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [lines]);

  if (done) {
    return (
      <div className="scan-strip">
        <span className="scan-strip__dot" />
        <span className="mono">{filename}</span>
        <span className="scan-strip__sep">·</span>
        <span className="mono">{durationMs != null ? `${durationMs} ms` : 'tamamlandı'}</span>
        <span className="scan-strip__sep">·</span>
        <span className="label">Tarama tamam</span>
      </div>
    );
  }

  return (
    <div className="scan-console" ref={logRef}>
      {lines.map((l, idx) => (
        <div key={idx} className="scan-console__line">
          <span className="scan-console__prompt">›</span> {l}
        </div>
      ))}
      <div className="scan-console__line">
        <span className="scan-console__prompt">›</span>
        <span className="scan-console__caret" />
      </div>
    </div>
  );
}
