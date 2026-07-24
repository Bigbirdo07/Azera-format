"""Skyward-export field mapping: canonical detection, sensitivity, and the
discipline (department) vs conduct_status (behavioral) disambiguation."""

from __future__ import annotations

from core.field_policy import is_protected
from core.privacy import classify_sensitivity
from core.schema import canonical_for
from nlp.synonym_mapper import load_json, match_column_for_concept


def test_new_skyward_canonical_fields_detected():
    assert canonical_for("Grad Year") == "grad_year"
    assert canonical_for("Home Address") == "home_address"
    assert canonical_for("Mailing Address") == "mailing_address"
    assert canonical_for("Emergency Contact") == "emergency_contact"
    assert canonical_for("Entry Date") == "entry_date"
    assert canonical_for("Withdrawal Date") == "withdrawal_date"


def test_discipline_information_maps_to_conduct_not_department():
    # "Discipline Information" is Skyward's real, specific behavioral report
    # name -> conduct_status. Bare "Discipline" is ambiguous in general school-
    # roster usage and means academic field of study far more often (the
    # higher-ed sense: Engineering, Nursing, ...) -- it maps to department,
    # matching the chat-facing "discipline" concept in knowledge/synonyms.json
    # and avoiding a real collision found on the (non-Skyward) mock rosters.
    assert canonical_for("Discipline Information") == "conduct_status"
    assert canonical_for("Discipline") == "department"
    assert canonical_for("Department") == "department"


def test_discipline_information_is_sensitive():
    sensitive, sensitivity_type = classify_sensitivity("Discipline Information")
    assert sensitive
    assert sensitivity_type == "disciplinary"


def test_bare_department_word_not_flagged_disciplinary():
    sensitive, sensitivity_type = classify_sensitivity("Department")
    assert not (sensitive and sensitivity_type == "disciplinary")


def test_emergency_contact_and_addresses_are_sensitive():
    for column in ("Emergency Contact", "Home Address", "Mailing Address"):
        sensitive, sensitivity_type = classify_sensitivity(column)
        assert sensitive, column
        assert sensitivity_type == "contact", column
        assert is_protected(column), column


def test_conduct_status_concept_does_not_collide_with_department():
    synonyms = load_json("synonyms.json")
    columns = ["Department", "Discipline Information", "GPA"]
    dept_column, _ = match_column_for_concept("discipline", columns, synonyms)
    conduct_column, _ = match_column_for_concept("conduct_status", columns, synonyms)
    assert dept_column == "Department"
    assert conduct_column == "Discipline Information"
