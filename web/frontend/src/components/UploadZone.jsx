// Step 1 — full-width upload dropzone (DESIGN.md §4). Dashed border, drag-active
// turns solid accent + bg-alt tint. Replaces itself with the scan console once a
// file is accepted (no modal).
import { useRef, useState } from 'react';

export default function UploadZone({ onFile }) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (files) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div
      className={`dropzone ${dragActive ? 'dropzone--active' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
      onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }}
      onDrop={(e) => {
        e.preventDefault();
        setDragActive(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/bmp,image/webp"
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
      <svg className="dropzone__icon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <div className="dropzone__title">Drop a passport image here</div>
      <div className="dropzone__formats mono">JPG · PNG · BMP · WEBP — max 15 MB</div>
    </div>
  );
}
