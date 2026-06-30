// Submit / Save gate (DESIGN.md §4). The primary button stays visually present
// but disabled while any mandatory field is unresolved. A persistent helper line
// with live mono numerals counts down the blocking fields; the button enables
// only at zero. Never silently disabled. Saving persists the record to the
// backend (image + model output + corrected fields) for active learning.
export default function SaveGate({ unresolved, saved, saving, saveError, onSave }) {
  const blocked = unresolved > 0;
  return (
    <section className="savegate">
      <div className="savegate__helper">
        {saved ? (
          <span className="savegate__ok">Saved.</span>
        ) : saveError ? (
          <span className="savegate__error">{saveError}</span>
        ) : saving ? (
          <span>Saving…</span>
        ) : blocked ? (
          <span>
            <span className="mono savegate__count">{unresolved}</span>{' '}
            {unresolved === 1 ? 'field requires' : 'fields require'} correction before saving
          </span>
        ) : (
          <span>All fields resolved — ready to save.</span>
        )}
      </div>
      <button
        className="btn btn--primary"
        disabled={blocked || saved || saving}
        onClick={onSave}
      >
        {saved ? 'Saved' : saving ? 'Saving…' : 'Verify and Save'}
      </button>
    </section>
  );
}
