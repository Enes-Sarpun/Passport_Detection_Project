// Field result table (DESIGN.md §4). One row per MRZ field: label · value (mono) ·
// reliability score · mini score bar (with 0.75 tick) · status badge · inline
// correction. Row background stays white regardless of status — color lives only
// in the badge and the score bar. Correction rings: mandatory (persistent red 2px)
// vs optional (blue 1px focus). Below 768px the rows collapse to stacked cards.
import { THRESHOLD, scoreColor } from '../fields';

const BADGE = {
  verified: { text: 'Verified', cls: 'badge--high' },
  review: { text: 'Needs Review', cls: 'badge--mid' },
  low: { text: 'Needs Review', cls: 'badge--low' },
  not_found: { text: 'Not Found', cls: 'badge--notfound' },
};

function ScoreBar({ reliability, found }) {
  return (
    <div className="scorebar">
      <div className="scorebar__threshold" style={{ left: `${THRESHOLD * 100}%` }} />
      <div
        className="scorebar__fill"
        style={{ width: `${Math.max(reliability, 0) * 100}%`, background: scoreColor(reliability, found) }}
      />
    </div>
  );
}

export default function FieldTable({ fields, values, confirmed, onChange, onConfirm }) {
  confirmed = confirmed || {};
  return (
    <section className="ftable">
      <div className="ftable__head">
        <h3>Parsed Fields</h3>
      </div>

      <div className="ftable__grid ftable__grid--header label">
        <span>Field</span>
        <span>Value</span>
        <span className="ftable__num">Score</span>
        <span>Reliability</span>
        <span>Status</span>
      </div>

      {fields.map((f) => {
        const badge = BADGE[f.status];
        const corrected = values[f.key] ?? f.value;
        const editable = true; // any field is editable (§7)
        const isConfirmed = !!confirmed[f.key];
        const cur = String(corrected ?? '').trim();
        const changed = cur !== '' && cur !== String(f.value ?? '').trim();
        // A mandatory field that is filled but unchanged can be confirmed as
        // correct (the model may have read it right despite a low score).
        const canConfirm = f.mandatory && f.found && !changed;
        return (
          <div className="ftable__row" key={f.key}>
            <span className="ftable__label label">{f.label}</span>

            <div className="ftable__value">
              {editable ? (
                <input
                  className={`field-input mono ${f.mandatory ? 'field-input--mandatory' : 'field-input--optional'}`}
                  value={corrected}
                  placeholder={f.found ? '' : 'Enter a value'}
                  onChange={(e) => onChange(f.key, e.target.value)}
                  aria-label={f.label}
                />
              ) : (
                <span className="mono">{corrected}</span>
              )}
              {f.mandatory && !f.found && (
                <span className="field-hint field-hint--req">
                  <Lock /> Could not read this field — enter a value
                </span>
              )}
              {canConfirm && (
                <label className={`field-confirm ${isConfirmed ? 'field-confirm--on' : ''}`}>
                  <input
                    type="checkbox"
                    checked={isConfirmed}
                    onChange={(e) => onConfirm?.(f.key, e.target.checked)}
                  />
                  {isConfirmed ? 'Confirmed correct' : 'Value is correct — confirm, or edit above'}
                </label>
              )}
              {f.mandatory && f.found && changed && (
                <span className="field-hint field-hint--opt">Edited</span>
              )}
              {!f.mandatory && f.status === 'review' && (
                <span className="field-hint field-hint--opt">Optional — score above threshold</span>
              )}
              {f.extra && <span className="field-extra">{f.extra}</span>}
            </div>

            <span className="ftable__num mono">{f.found ? f.reliability.toFixed(2) : '—'}</span>

            <div className="ftable__bar">
              <ScoreBar reliability={f.reliability} found={f.found} />
            </div>

            <span className={`badge ${badge.cls}`}>{badge.text}</span>
          </div>
        );
      })}
    </section>
  );
}

function Lock() {
  return (
    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ verticalAlign: '-1px' }}>
      <rect x="5" y="11" width="14" height="9" rx="1.5" />
      <path d="M8 11V8a4 4 0 018 0v3" />
    </svg>
  );
}
