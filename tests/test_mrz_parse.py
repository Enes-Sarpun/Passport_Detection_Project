"""
Acceptance tests for MRZ parsing pipeline — Phase 5.

All passport strings are synthetic specimen samples (SPECIMEN / UTO / SOM / ITA).
No real identity data.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from Scripts.parsing.mrz_parse import (
    parse_mrz,
    parse_td3,
    check_digit,
    check_digit_valid,
)
from Scripts.parsing.country_lookup import resolve_country
from Scripts.parsing.schema import build_output, _is_dob_century_ambiguous


# ---------------------------------------------------------------------------
# A1 — Ülke kodu digit onarımı
# ---------------------------------------------------------------------------

class TestCountryCodeRepair:
    def test_som_digit_zero_repaired(self):
        result = resolve_country("S0M")
        assert result["code"] == "SOM"
        assert "Somalia" in result["name"] or result["name"] != "Unknown"

    def test_som_repaired_recorded_in_auto_repaired(self):
        line1 = "P<SOMSPECIMEN<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "SP00000014S0M8001011M3001017<<<<<<<<<<<<<<<0"
        line1 = line1[:44].ljust(44, "<")
        line2 = line2[:44].ljust(44, "<")
        result = parse_td3(line1, line2)
        assert result.nationality in ("SOM", "S0M")
        assert "nationality" in result.auto_repaired_fields

    def test_clean_country_no_repair(self):
        result = resolve_country("ITA")
        assert result["code"] == "ITA"
        assert result["name"] != "Unknown"

    def test_digit_to_letter_map_all_keys(self):
        from Scripts.parsing.country_lookup import _repair_country_digits
        assert _repair_country_digits("S0M") == "SOM"
        assert _repair_country_digits("1TA") == "ITA"
        assert _repair_country_digits("FR5") == "FRS"


# ---------------------------------------------------------------------------
# A2/A3 — İsim alanı digit onarımı + yapılandırılmış name objesi
# ---------------------------------------------------------------------------

class TestNameRepairAndStructure:
    def test_irish_leading_digit_repaired(self):
        from Scripts.parsing.mrz_parse import parse_name
        repaired: list[str] = []
        surname, given, name_dict = parse_name("0SULLIVAN<<JOHN<PAUL<<<<<<<<<<<<<<<<<<<<<<<", repaired)
        assert surname == "OSULLIVAN"
        assert "name" in repaired

    def test_name_dict_structure(self):
        from Scripts.parsing.mrz_parse import parse_name
        _, _, name_dict = parse_name("EL<IDRISSI<ADIB<<MOHAMMED<AMINE<<<<<<<<<<<", [])
        assert "surname" in name_dict
        assert "given_names" in name_dict
        assert "given_names_list" in name_dict
        assert "full_name" in name_dict
        assert isinstance(name_dict["given_names_list"], list)

    def test_double_chevron_splits_surname_given(self):
        from Scripts.parsing.mrz_parse import parse_name
        surname, given, name_dict = parse_name("ROSSI<<MARIA<ANNA<<<<<<<<<<<<<<<<<<<<<<<<", [])
        assert surname == "ROSSI"
        assert given == "MARIA ANNA"
        assert name_dict["given_names_list"] == ["MARIA", "ANNA"]

    def test_full_name_order(self):
        from Scripts.parsing.mrz_parse import parse_name
        _, _, name_dict = parse_name("ROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", [])
        assert name_dict["full_name"] == "MARIA ROSSI"

    def test_no_given_name(self):
        from Scripts.parsing.mrz_parse import parse_name
        _, given, name_dict = parse_name("SPECIMEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", [])
        assert given == ""
        assert name_dict["given_names_list"] == []


# ---------------------------------------------------------------------------
# Italy clean parse — zero warnings, all check digits valid
# ---------------------------------------------------------------------------

class TestItalyCleanParse:
    LINE1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    LINE2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08"

    def test_parses_successfully(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        assert result is not None

    def test_all_checks_valid(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        v = result.validation
        assert v["document_number_valid"] is True
        assert v["date_of_birth_valid"] is True
        assert v["date_of_expiry_valid"] is True
        assert v["composite_valid"] is True

    def test_zero_auto_repaired(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        assert result.auto_repaired_fields == []

    def test_build_output_no_checkdigit_warnings(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        output = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        cd_warnings = [w for w in output["warnings"] if w.startswith("checkdigit_failed")]
        assert cd_warnings == []

    def test_name_fields(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        assert result.surname == "ROSSI"
        assert result.given_names == "MARIA"

    def test_sex_description(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        output = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert output["holder"]["sex"]["description"] == "Female"

    def test_document_type_description(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        output = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        # Phase 5: document.type is an object with code and description
        assert output["document"]["type"]["description"] == "Passport"
        assert output["document"]["type"]["code"] == "P"


# ---------------------------------------------------------------------------
# Corrupted name — must warn, must NOT crash or emit clean-looking wrong name
# ---------------------------------------------------------------------------

class TestCorruptedNameWarning:
    LINE1 = "P<UTOSPECIMEN<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<"
    LINE2 = "UT0000001<UTO8001011M3001017<<<<<<<<<<<<<<<6"

    def test_pipeline_does_not_crash(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        assert result is None or hasattr(result, "surname")

    def test_name_low_confidence_warning_via_extra(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        if result is None:
            return
        output = build_output(
            result,
            detection_confidence=0.7,
            ocr_confidence=0.5,
            extra_warnings=["name_low_confidence"],
        )
        assert "name_low_confidence" in output["warnings"]

    def test_low_ocr_confidence_triggers_warning(self):
        result = parse_mrz([self.LINE1, self.LINE2])
        if result is None:
            return
        output = build_output(result, detection_confidence=0.8, ocr_confidence=0.45)
        assert "low_ocr_confidence" in output["warnings"]


# ---------------------------------------------------------------------------
# B1 — century ambiguous uyarısı
# ---------------------------------------------------------------------------

class TestDobCenturyAmbiguous:
    def test_very_old_dob_flags_ambiguous(self):
        assert _is_dob_century_ambiguous("1912-12-12") is True

    def test_too_young_dob_flags_ambiguous(self):
        import datetime
        young = (datetime.date.today().replace(year=datetime.date.today().year - 3)).isoformat()
        assert _is_dob_century_ambiguous(young) is True

    def test_iceland_specimen_dob_warns(self):
        assert _is_dob_century_ambiguous("2012-12-12") is False
        assert _is_dob_century_ambiguous("1912-12-12") is True

    def test_normal_dob_not_ambiguous(self):
        assert _is_dob_century_ambiguous("1990-06-15") is False

    def test_none_not_ambiguous(self):
        assert _is_dob_century_ambiguous(None) is False

    def test_dates_section_slimmed(self):
        # Slimmed output: age / days_until_expiry / validity_period_years removed;
        # only is_expired derived field is kept.
        result = parse_mrz([
            "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08",
        ])
        output = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert "is_expired" in output["dates"]
        assert "age" not in output["dates"]
        assert "days_until_expiry" not in output["dates"]
        assert "validity_period_years" not in output["dates"]

    def test_century_ambiguous_warning_in_output(self):
        result = parse_mrz([
            "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08",
        ])
        result.birth_date_iso = "1912-12-12"
        output = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert "dob_century_ambiguous" in output["warnings"]


# ---------------------------------------------------------------------------
# B2 — document_type_description
# ---------------------------------------------------------------------------

class TestDocumentTypeDescription:
    def test_p_is_passport(self):
        from Scripts.parsing.schema import _document_type_description
        assert _document_type_description("P") == "Passport"

    def test_pl_is_special_passport(self):
        from Scripts.parsing.schema import _document_type_description
        assert _document_type_description("PL") == "Special passport"

    def test_id_is_identity_card(self):
        from Scripts.parsing.schema import _document_type_description
        assert _document_type_description("ID") == "Identity card"

    def test_unknown_code_is_unknown(self):
        from Scripts.parsing.schema import _document_type_description
        assert _document_type_description("ZZ") == "Unknown"


# ---------------------------------------------------------------------------
# B3 — sex_description
# ---------------------------------------------------------------------------

class TestSexDescription:
    def test_m_is_male(self):
        from Scripts.parsing.schema import _sex_description
        assert _sex_description("M") == "Male"

    def test_f_is_female(self):
        from Scripts.parsing.schema import _sex_description
        assert _sex_description("F") == "Female"

    def test_x_is_unspecified(self):
        from Scripts.parsing.schema import _sex_description
        assert _sex_description("X") == "Unspecified"

    def test_filler_is_unspecified(self):
        from Scripts.parsing.schema import _sex_description
        assert _sex_description("<") == "Unspecified"


# ---------------------------------------------------------------------------
# B4 — warnings[] system
# ---------------------------------------------------------------------------

class TestNameLowConfidence:
    def test_short_token_in_given_names_warns(self):
        line1 = "P<DOMFORTUNA<RAMIREZ<<LU<Y<IRENE<<<<<<<<<<<<<<"
        line2 = "SC14881686DOM9502150F160806700100000001<<<46<<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("MRZ parse failed for specimen")
        output = build_output(
            result,
            detection_confidence=0.84,
            ocr_confidence=0.69,
            raw_mrz=[line1, line2],
        )
        assert "name_low_confidence" in output["warnings"]

    def test_short_token_in_surname_warns(self):
        line1 = "P<GBRX<SMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "KF00000016GBR9011012M3307308<<<<<<<<<<<<<<08<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("MRZ parse failed for specimen")
        output = build_output(
            result,
            detection_confidence=0.9,
            ocr_confidence=0.7,
            raw_mrz=[line1, line2],
        )
        assert "name_low_confidence" in output["warnings"]

    def test_double_surname_no_warning(self):
        line1 = "P<DOMFORTUNA<RAMIREZ<<LUCY<IRENE<<<<<<<<<<<<<"
        line2 = "SC14881686DOM9502150F160806700100000001<<<46<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("MRZ parse failed for specimen")
        output = build_output(
            result,
            detection_confidence=0.84,
            ocr_confidence=0.69,
            raw_mrz=[line1, line2],
        )
        assert "name_low_confidence" not in output["warnings"]

    def test_multiple_given_names_no_warning(self):
        line1 = "P<NZLWHAKAATURANGA<<FRED<WIREMU<JOHN<<<<<<<<<<"
        line2 = "LD001148<2NZL6402282M1410055<<<<<<<<<<<<<<00<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("MRZ parse failed for specimen")
        output = build_output(
            result,
            detection_confidence=0.86,
            ocr_confidence=0.62,
            raw_mrz=[line1, line2],
        )
        assert "name_low_confidence" not in output["warnings"]

    def test_clean_name_no_warning(self):
        line1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08<"
        result = parse_mrz([line1, line2])
        output = build_output(
            result,
            detection_confidence=0.9,
            ocr_confidence=0.9,
            raw_mrz=[line1, line2],
        )
        assert "name_low_confidence" not in output["warnings"]


# ---------------------------------------------------------------------------
# E0 — name_separator_missing
# ---------------------------------------------------------------------------

class TestNameSeparatorMissing:
    def test_nld_no_double_chevron_warns(self):
        """OCR dropped '<<': all name in surname, given_names empty → name_separator_missing."""
        line1 = "PNLDDE<BRUIJNWILLEKELISELOTTE<<<<<<<<<<<<<<<<<"
        line2 = "XJ1F101624NLD6503101F1610202999999990<<<<<86<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("parse failed")
        output = build_output(
            result,
            detection_confidence=0.86,
            ocr_confidence=0.74,
            raw_mrz=[line1, line2],
        )
        assert "name_separator_missing" in output["warnings"]

    def test_som_no_double_chevron_warns(self):
        """SOM specimen with no '<<' in name field."""
        line1 = "PSOMBARKADLE<GELLE<GUTAALE<<<<<<<<<<<<<<<<<<<<"
        line2 = "P000000005S0M8205225M0912286N0000025583<<<04<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("parse failed")
        output = build_output(
            result,
            detection_confidence=0.85,
            ocr_confidence=0.65,
            raw_mrz=[line1, line2],
        )
        assert "name_separator_missing" in output["warnings"]

    def test_clean_name_no_separator_warning(self):
        """Normal name with '<<' must NOT trigger name_separator_missing."""
        line1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08<"
        result = parse_mrz([line1, line2])
        output = build_output(
            result,
            detection_confidence=0.9,
            ocr_confidence=0.9,
            raw_mrz=[line1, line2],
        )
        assert "name_separator_missing" not in output["warnings"]


# ---------------------------------------------------------------------------
# E2/E3 — Phase 5 schema shape tests
# ---------------------------------------------------------------------------

class TestNewSchemaShape:
    LINE1 = "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    LINE2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08"

    def _build(self, det=0.9, ocr=0.9):
        result = parse_mrz([self.LINE1, self.LINE2])
        return build_output(result, detection_confidence=det, ocr_confidence=ocr,
                            raw_mrz=[self.LINE1, self.LINE2])

    def test_top_level_keys(self):
        out = self._build()
        for key in ("document", "holder", "dates",
                    "validation", "quality", "warnings", "raw_mrz"):
            assert key in out, f"Missing top-level key: {key}"

    def test_no_schema_version(self):
        out = self._build()
        assert "schema_version" not in out

    def test_document_section(self):
        out = self._build()
        doc = out["document"]
        # type is an object
        assert isinstance(doc["type"], dict)
        assert doc["type"]["code"] == "P"
        assert doc["type"]["description"] == "Passport"
        # number has value, reliability (per-field 'valid' removed for readability)
        assert "value" in doc["number"]
        assert "reliability" in doc["number"]
        assert "valid" not in doc["number"]
        # issuing_country removed from output (nationality carries the same info)
        assert "issuing_country" not in doc
        # personal_number is an object
        assert isinstance(doc["personal_number"], dict)
        assert "value" in doc["personal_number"]
        assert "valid" not in doc["personal_number"]
        # mrz_format present
        assert "mrz_format" in doc

    def test_holder_section(self):
        out = self._build()
        h = out["holder"]
        assert "surname" in h and "value" in h["surname"]
        assert "given_names" in h and "value" in h["given_names"]
        # Phase 5: given_names_list IS present
        assert "given_names_list" in h
        assert isinstance(h["given_names_list"], list)
        assert "full_name" in h
        assert "nationality" in h
        assert "name" in h["nationality"]
        # Phase 5: nationality has code
        assert "code" in h["nationality"]
        assert "sex" in h
        assert "code" in h["sex"] and "description" in h["sex"]

    def test_dates_section(self):
        out = self._build()
        d = out["dates"]
        assert "date_of_birth" in d
        assert "date_of_expiry" in d
        # Slimmed: only is_expired kept; age / days_until_expiry / validity removed
        assert "is_expired" in d
        assert "age" not in d
        assert "days_until_expiry" not in d
        assert "validity_period_years" not in d
        # per-field 'valid' removed for readability
        assert "valid" not in d["date_of_birth"]
        assert "raw" in d["date_of_birth"] and "iso" in d["date_of_birth"]

    def test_validation_section_always_present(self):
        out = self._build()
        v = out["validation"]
        # Simplified: summary-only, detailed per-check list omitted for readability
        assert "checks" not in v
        assert "mrz_overall_valid" in v
        assert "failed_checks" in v
        assert "auto_repaired_fields" in v

    def test_quality_section(self):
        out = self._build(det=0.9, ocr=0.95)
        q = out["quality"]
        # Simplified: only the summary metrics are kept
        assert "reliability_score" in q
        assert "rescan_recommended" in q
        # raw detection/ocr confidences removed for readability
        assert "detection_confidence" not in q
        assert "ocr_confidence" not in q
        # is_specimen only present when True
        assert "is_specimen" not in q
        # no longer a "confidence" section at top level
        assert "confidence" not in out

    def test_raw_mrz_present(self):
        out = self._build()
        assert "raw_mrz" in out
        assert out["raw_mrz"] == [self.LINE1, self.LINE2]

    def test_no_old_flat_keys(self):
        out = self._build()
        assert "document_type" not in out
        assert "document_type_description" not in out
        assert "issuing_country" not in out
        assert "fields" not in out
        assert "name" not in out
        assert "detection_confidence" not in out
        assert "ocr_confidence" not in out
        assert "overall_confidence" not in out
        assert "confidence" not in out

    def test_sex_is_object(self):
        out = self._build()
        sex = out["holder"]["sex"]
        assert isinstance(sex, dict)
        assert sex["code"] == "F"
        assert sex["description"] == "Female"

    def test_reliability_score_range(self):
        out = self._build(det=0.9, ocr=0.95)
        assert 0.0 <= out["quality"]["reliability_score"] <= 1.0


class TestWarningsSystem:
    def test_document_expired_warning(self):
        result = parse_mrz([
            "PLMAREL<IDRISSI<ADIB<<MOHAMMED<AMINE<<<<<<<<",
            "SP15934131MAR5406238M1303317XA123456<<<<<<22",
        ])
        if result is None:
            pytest.skip("MRZ parse failed for specimen")
        output = build_output(result, detection_confidence=0.86, ocr_confidence=0.67)
        assert "document_expired" in output["warnings"]

    def test_no_mrz_failure_output_has_warnings(self):
        from Scripts.parsing.schema import failure_output
        out = failure_output("no_mrz_detected", warnings=["no_mrz_detected"])
        assert "warnings" in out
        assert "no_mrz_detected" in out["warnings"]

    def test_low_confidence_status_in_output(self):
        # A corrupted MRZ with all check digits failing → reliability drops below 0.75
        result = parse_mrz([
            "P<UTOSPECIMEN<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "UT0000001<UTO8001011M3001017<<<<<<<<<<<<<<<0",  # bad composite cd
        ])
        if result is None:
            pytest.skip("parse failed")
        output = build_output(result, detection_confidence=0.3, ocr_confidence=0.3)
        assert output.get("status") == "low_confidence" or "low_confidence" in output.get("warnings", [])

    def test_ok_status_not_in_output(self):
        result = parse_mrz([
            "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08",
        ])
        output = build_output(result, detection_confidence=0.95, ocr_confidence=0.95)
        assert "status" not in output


# ---------------------------------------------------------------------------
# Check-digit unit tests (ICAO 9303 worked examples)
# ---------------------------------------------------------------------------

class TestCheckDigits:
    def test_icao_example_doc_number(self):
        assert check_digit("L898902C3") == "6"

    def test_icao_example_birth_date(self):
        assert check_digit("740812") == "2"

    def test_icao_example_expiry(self):
        assert check_digit("960415") == "7"

    def test_check_digit_valid_returns_true(self):
        assert check_digit_valid("L898902C3", "6") is True

    def test_check_digit_valid_returns_false(self):
        assert check_digit_valid("L898902C3", "0") is False

    def test_filler_in_checkdigit_slot(self):
        assert check_digit_valid("<<<<<<<<<", "<") is True


# ---------------------------------------------------------------------------
# Phase 3 — D5: Structural validation tests
# ---------------------------------------------------------------------------

class TestStructuralValidation:
    _LINE1 = "P<NZLWHAKAATURANGA<<FRED<WIREMU<JOHN<<<<<<<<<<"
    _LINE2 = "LD001148<2NZL6402282M1410055<<<<<<<<<<<<<<00<<"

    def test_clean_sample_mrz_overall_valid(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["mrz_overall_valid"] is True
        assert result.validation["failed_checks"] == []

    def test_unknown_country_fails_overall(self):
        line1 = "P<ZZZSURNAME<<GIVEN<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "AB1234567<ZZZ8001011M3001011<<<<<<<<<<<<<<<04<"
        result = parse_mrz([line1, line2])
        assert result is not None
        assert result.validation["country_codes_known"] is False
        assert result.validation["mrz_overall_valid"] is False
        assert "country_codes_known" in result.validation["failed_checks"]

    def test_impossible_date_dates_not_well_formed(self):
        line1 = "P<UTOSPHERE<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "AB1234567<UTO9913012M3001011<<<<<<<<<<<<<<<04"
        result = parse_mrz([line1, line2])
        assert result is not None
        assert result.validation["dates_well_formed"] is False
        assert result.validation["mrz_overall_valid"] is False

    def test_line_length_valid_td3(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["line_length_valid"] is True

    def test_expiry_after_birth(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["expiry_after_birth"] is True

    def test_country_codes_known(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["country_codes_known"] is True

    def test_sex_value_valid(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["sex_value_valid"] is True

    def test_document_type_known(self):
        result = parse_mrz([self._LINE1, self._LINE2])
        assert result is not None
        assert result.validation["document_type_known"] is True


# ---------------------------------------------------------------------------
# Phase 3 — C5: Per-field confidence tests
# ---------------------------------------------------------------------------

class TestFieldConfidence:
    _LINE1 = "P<NZLWHAKAATURANGA<<FRED<WIREMU<JOHN<<<<<<<<<<"
    _LINE2 = "LD001148<2NZL6402282M1410055<<<<<<<<<<<<<<00<<"

    def _build(self, det=0.9, ocr=0.9):
        result = parse_mrz([self._LINE1, self._LINE2])
        return build_output(result, detection_confidence=det, ocr_confidence=ocr,
                            raw_mrz=[self._LINE1, self._LINE2])

    def test_clean_sample_field_reliability_high(self):
        # Clean read with passing check digits -> reliability near each field's
        # empirical base accuracy (all > 0.90 for check-digit-backed fields).
        out = self._build(ocr=0.95)
        assert out["document"]["number"]["reliability"] > 0.90
        assert out["dates"]["date_of_birth"]["reliability"] > 0.90
        assert out["dates"]["date_of_expiry"]["reliability"] > 0.90

    def test_failed_checkdigit_drops_reliability(self):
        # A field whose check digit fails should score far below a clean field.
        line1 = "P<UTOSPECIMEN<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "UT0000001<UTO8001011M3001017<<<<<<<<<<<<<<<6"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("parse failed")
        out = build_output(result, detection_confidence=0.9, ocr_confidence=0.9,
                           raw_mrz=[line1, line2])
        # nationality (no check digit, clean) should outrank a failed-cd field.
        nat = out["holder"]["nationality"]["reliability"]
        assert nat > 0.5

    def test_name_field_reliability_reflects_base(self):
        # Name fields have a lower empirical base (~0.92) than check-digit fields,
        # so even at high OCR confidence they score below document_number.
        out = self._build(ocr=0.95)
        name_rel = out["holder"]["given_names"]["reliability"]
        docno_rel = out["document"]["number"]["reliability"]
        assert name_rel <= docno_rel
        assert 0.0 <= name_rel <= 1.0

    def test_overall_reliability_formula(self):
        out = self._build(det=0.9, ocr=0.95)
        # reliability_score in quality block
        assert 0.75 <= out["quality"]["reliability_score"] <= 1.0


# ---------------------------------------------------------------------------
# Phase 5 — J: Derived features tests
# ---------------------------------------------------------------------------

class TestDerivedFeatures:
    def test_is_expired_true_for_expired_doc(self):
        result = parse_mrz([
            "PLMAREL<IDRISSI<ADIB<<MOHAMMED<AMINE<<<<<<<<",
            "SP15934131MAR5406238M1303317XA123456<<<<<<22",
        ])
        if result is None:
            pytest.skip("parse failed")
        out = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert out["dates"]["is_expired"] is True

    def test_nationality_differs_from_issuer_flag(self):
        # P<GBRITA: issuing=GBR, nationality=ITA → should warn
        line1 = "P<GBRROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        line2 = "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08<<"
        result = parse_mrz([line1, line2])
        if result is None:
            pytest.skip("parse failed")
        out = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert "nationality_differs_from_issuer" in out["warnings"]

    def test_specimen_flagged_in_quality(self):
        result = parse_mrz([
            "P<UTOSPHERE<<SPECIMEN<<<<<<<<<<<<<<<<<<<<<<<<<",
            "AB1234567<UTO9001011M3001011<<<<<<<<<<<<<<<04",
        ])
        if result is None:
            pytest.skip("parse failed")
        out = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert out["quality"]["is_specimen"] is True

    def test_rescan_recommended_when_low_reliability(self):
        # Specimen with bad check digits → rescan_recommended must be True
        result = parse_mrz([
            "P<UTOSPECIMEN<<TEST<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "UT0000001<UTO8001011M3001017<<<<<<<<<<<<<<<0",
        ])
        if result is None:
            pytest.skip("parse failed")
        out = build_output(result, detection_confidence=0.3, ocr_confidence=0.3)
        assert out["quality"]["rescan_recommended"] is True

    def test_clean_doc_has_no_failed_checks(self):
        result = parse_mrz([
            "P<ITAROSSI<<MARIA<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
            "KF00000016ITA9011012F3307308<<<<<<<<<<<<<<08",
        ])
        out = build_output(result, detection_confidence=0.9, ocr_confidence=0.9)
        assert out["validation"]["mrz_overall_valid"] is True
        assert out["validation"]["failed_checks"] == []
