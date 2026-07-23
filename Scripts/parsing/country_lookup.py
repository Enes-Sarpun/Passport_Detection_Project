from __future__ import annotations
import pycountry

from .mrz_constants import repair_digits_to_letters

# ICAO-special / reserved codes that are not in ISO 3166-1 alpha-3.
_ICAO_OVERRIDES: dict[str, str] = {
    "UTO": "Utopia (ICAO test code)",
    "D<<": "Germany",
    "GBD": "British Dependent Territories Citizen",
    "GBN": "British National (Overseas)",
    "GBO": "British Overseas Citizen",
    "GBP": "British Protected Person",
    "GBS": "British Subject",
    "EUE": "European Union",
    "UNA": "United Nations Agency",
    "UNO": "United Nations Organization",
    "XOM": "Sovereign Military Order of Malta",
    "XXA": "Stateless person (Convention 1954)",
    "XXB": "Refugee (Convention 1951)",
    "XXC": "Refugee (other)",
    "XXX": "Unspecified nationality",
}

def _repair_country_digits(code: str) -> str:
    return repair_digits_to_letters(code)


def resolve_country(alpha3: str, repaired: list[str] | None = None, field_name: str = "") -> dict[str, str]:
    raw = alpha3.strip().upper()

    # Check ICAO overrides before stripping '<' — codes like 'D<<' must match as-is.
    if raw in _ICAO_OVERRIDES:
        return {"code": raw, "name": _ICAO_OVERRIDES[raw]}

    code = raw.replace("<", "")
    if not code:
        return {"code": raw, "name": "Unknown"}

    if code in _ICAO_OVERRIDES:
        return {"code": code, "name": _ICAO_OVERRIDES[code]}

    def _lookup(c: str) -> str | None:
        try:
            country = pycountry.countries.get(alpha_3=c)
            return country.name if country else None
        except Exception:
            return None

    name = _lookup(code)
    if name:
        return {"code": code, "name": name}

    # Attempt digit→letter repair if any digit is present.
    if any(c.isdigit() for c in code):
        repaired_code = _repair_country_digits(code)
        if repaired_code != code:
            name = _lookup(repaired_code)
            if name is None and repaired_code in _ICAO_OVERRIDES:
                name = _ICAO_OVERRIDES[repaired_code]
            if name:
                if repaired is not None and field_name:
                    repaired.append(field_name)
                return {"code": repaired_code, "name": name}

    return {"code": code, "name": "Unknown"}
