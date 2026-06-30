// Flattens the pipeline's nested JSON into a flat list of MRZ field rows the
// Trust Console renders, and derives reliability status per field.
// Threshold logic mirrors passport-detection-DESIGN.md §2 (0.75 cutoff).

export const THRESHOLD = 0.75;
export const HIGH = 0.85;

// status: 'verified' (>=0.85), 'review' (0.75–0.85), 'low' (<0.75), 'not_found'
export function fieldStatus(reliability, found) {
  if (!found) return 'not_found';
  if (reliability >= HIGH) return 'verified';
  if (reliability >= THRESHOLD) return 'review';
  return 'low';
}

// A field correction is MANDATORY when the value is missing or below threshold.
export function isMandatory(reliability, found) {
  return !found || reliability < THRESHOLD;
}

const isPresent = (v) => v !== undefined && v !== null && String(v).replace(/[<\s]/g, '') !== '';

// UI labels per §9.
const LABELS = {
  document_type: 'Document Type',
  document_number: 'Document No.',
  personal_number: 'Personal No.',
  nationality: 'Nationality',
  surname: 'Surname',
  given_names: 'Given Names',
  date_of_birth: 'Date of Birth',
  date_of_expiry: 'Date of Expiry',
  sex: 'Sex',
};

// Build the ordered field rows from the API result object.
export function extractFields(result) {
  if (!result || !result.document) return [];
  const d = result.document;
  const h = result.holder || {};
  const dt = result.dates || {};

  const rows = [
    { key: 'document_type', value: d.type?.code, reliability: 1.0, hasReliability: false },
    { key: 'document_number', value: d.number?.value, reliability: d.number?.reliability },
    { key: 'personal_number', value: d.personal_number?.value, reliability: d.personal_number?.reliability },
    { key: 'nationality', value: h.nationality?.code, reliability: h.nationality?.reliability, extra: h.nationality?.name },
    { key: 'surname', value: h.surname?.value, reliability: h.surname?.reliability },
    { key: 'given_names', value: h.given_names?.value, reliability: h.given_names?.reliability },
    { key: 'date_of_birth', value: dt.date_of_birth?.iso, reliability: dt.date_of_birth?.reliability, raw: dt.date_of_birth?.raw },
    { key: 'date_of_expiry', value: dt.date_of_expiry?.iso, reliability: dt.date_of_expiry?.reliability, raw: dt.date_of_expiry?.raw },
    { key: 'sex', value: h.sex?.code, reliability: h.sex?.reliability, extra: h.sex?.description },
  ];

  return rows.map((r) => {
    const found = isPresent(r.value);
    const reliability = typeof r.reliability === 'number' ? r.reliability : (found ? 1.0 : 0);
    return {
      ...r,
      label: LABELS[r.key] || r.key,
      value: found ? String(r.value) : '',
      found,
      reliability,
      status: fieldStatus(reliability, found),
      mandatory: r.hasReliability === false ? false : isMandatory(reliability, found),
    };
  });
}

export function scoreColor(reliability, found) {
  if (!found) return 'var(--score-low)';
  if (reliability >= HIGH) return 'var(--score-high)';
  if (reliability >= THRESHOLD) return 'var(--score-mid)';
  return 'var(--score-low)';
}
