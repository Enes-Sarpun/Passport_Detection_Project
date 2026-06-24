from __future__ import annotations
import pycountry

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

def resolve_country(alpha3: str) -> dict[str, str]:
    raw = alpha3.strip().upper()

    # Check ICAO overrides before stripping '<' — codes like 'D<<' must match as-is.
    if raw in _ICAO_OVERRIDES:
        return {"code": raw, "name": _ICAO_OVERRIDES[raw]}

    code = raw.replace("<", "")
    if not code:
        return {"code": raw, "name": "Unknown"}

    if code in _ICAO_OVERRIDES:
        return {"code": code, "name": _ICAO_OVERRIDES[code]}

    try:
        country = pycountry.countries.get(alpha_3=code)
        if country:
            return {"code": code, "name": country.name}
    except Exception:
        pass

    return {"code": code, "name": "Unknown"}
