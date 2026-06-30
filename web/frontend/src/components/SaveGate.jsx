// Submit / Save gate (DESIGN.md §4). The primary button stays visually present
// but disabled while any mandatory field is unresolved. A persistent helper line
// with live mono numerals counts down the blocking fields; the button enables
// only at zero. Never silently disabled.
export default function SaveGate({ unresolved, saved, onSave }) {
  const blocked = unresolved > 0;
  return (
    <section className="savegate">
      <div className="savegate__helper">
        {saved ? (
          <span className="savegate__ok">Kaydedildi.</span>
        ) : blocked ? (
          <span>
            <span className="mono savegate__count">{unresolved}</span>{' '}
            alan kaydetmeden önce düzeltme gerektiriyor
          </span>
        ) : (
          <span>Tüm alanlar çözüldü — kaydedilebilir.</span>
        )}
      </div>
      <button
        className="btn btn--primary"
        disabled={blocked || saved}
        onClick={onSave}
      >
        {saved ? 'Kaydedildi' : 'Doğrula ve Kaydet'}
      </button>
    </section>
  );
}
