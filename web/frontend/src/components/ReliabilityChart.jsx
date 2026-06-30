// Per-document reliability overview (DESIGN.md §4). Horizontal bar chart, one bar
// per field, sorted lowest→highest. A dashed vertical rule crosses all bars at the
// 0.75 threshold (Datadog/ClickHouse convention). Fields left of the line are
// grouped under a "Below threshold" divider. Color only in bars, never the row.
import { THRESHOLD, scoreColor } from '../fields';

export default function ReliabilityChart({ fields }) {
  // Sorted ascending so the riskiest fields are at the top.
  const sorted = [...fields].sort((a, b) => a.reliability - b.reliability);
  const below = sorted.filter((f) => f.reliability < THRESHOLD || !f.found);
  const above = sorted.filter((f) => f.reliability >= THRESHOLD && f.found);

  const Bar = ({ f }) => (
    <div className="chart-row">
      <div className="chart-row__label label">{f.label}</div>
      <div className="chart-row__track">
        {/* threshold line at 75% of the track width */}
        <div className="chart-row__threshold" style={{ left: `${THRESHOLD * 100}%` }} />
        <div
          className="chart-row__fill"
          style={{ width: `${Math.max(f.reliability, 0) * 100}%`, background: scoreColor(f.reliability, f.found) }}
        />
      </div>
      <div className="chart-row__value mono">{f.found ? f.reliability.toFixed(2) : '—'}</div>
    </div>
  );

  return (
    <section className="chart">
      <div className="chart__head">
        <h3>Reliability Overview</h3>
        <span className="chart__threshold-note mono">threshold 0.75</span>
      </div>

      {below.length > 0 && (
        <>
          <div className="chart__divider chart__divider--below">Below threshold</div>
          {below.map((f) => <Bar key={f.key} f={f} />)}
        </>
      )}
      {above.length > 0 && (
        <>
          {below.length > 0 && <div className="chart__divider">Above threshold</div>}
          {above.map((f) => <Bar key={f.key} f={f} />)}
        </>
      )}
    </section>
  );
}
