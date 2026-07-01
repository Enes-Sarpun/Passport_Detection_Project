// Raw MRZ editor — shows the exact machine-readable lines the OCR produced and
// lets the operator confirm or correct them character-for-character. This is the
// gold-label source for active-learning re-calibration: the confirmed lines are
// stored as corrected_mrz, in the same format as the ground truth.
// A live length hint (TD3=44, TD2=36, TD1=30) flags mis-typed lines.

const EXPECTED_LEN = { TD1: 30, TD2: 36, TD3: 44 };

export default function MrzEditor({ lines, mrzFormat, onChange }) {
  if (!Array.isArray(lines) || lines.length === 0) return null;
  const expected = EXPECTED_LEN[mrzFormat] || null;

  const setLine = (i, text) => {
    const next = [...lines];
    next[i] = text.toUpperCase();
    onChange(next);
  };

  return (
    <section className="mrz-editor">
      <div className="mrz-editor__head">
        <h3>Raw MRZ</h3>
        <span className="mrz-editor__note label">
          Confirm or correct the machine-readable lines
          {expected ? ` · ${mrzFormat} expects ${expected} chars` : ''}
        </span>
      </div>

      {lines.map((line, i) => {
        const len = (line || '').length;
        const off = expected != null && len !== expected;
        return (
          <div className="mrz-editor__row" key={i}>
            <input
              className={`mrz-editor__input mono ${off ? 'mrz-editor__input--off' : ''}`}
              value={line}
              spellCheck={false}
              autoCapitalize="characters"
              onChange={(e) => setLine(i, e.target.value)}
              aria-label={`MRZ line ${i + 1}`}
            />
            <span className={`mrz-editor__len mono ${off ? 'mrz-editor__len--off' : ''}`}>
              {len}{expected ? `/${expected}` : ''}
            </span>
          </div>
        );
      })}
    </section>
  );
}
