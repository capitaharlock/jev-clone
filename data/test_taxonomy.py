"""The taxonomy is a SOURCE or it is not an arm — #T-option-text."""
from __future__ import annotations

import json

import pytest

from data import taxonomy as TX


def test_banking77_defines_every_label_of_the_space():
    desc = TX.load()
    assert len(desc) == 77
    assert all(isinstance(v, str) and v.strip() for v in desc.values())


def test_card_declares_source_revision_license_and_its_caveat():
    card = TX.card()
    for key in ("source_original", "revision", "sha256", "license",
                "license_caveat", "attribution", "usage"):
        assert card.get(key), f"the card does not declare {key}"
    assert len(card["revision"]) == 40, "revision must be an immutable commit"
    assert len(card["sha256"]) == 64
    # the arm is eval-only on purpose: these definitions belong to the
    # experiment we compare ourselves against
    assert card["usage"] == "eval-only"


def test_load_refuses_a_file_that_drifted_from_its_card(tmp_path,
                                                        monkeypatch):
    home = tmp_path / "toy"
    home.mkdir()
    (home / "descriptions.json").write_text(json.dumps({"a": "one"}))
    (home / "card.json").write_text(json.dumps(
        {"sha256": "0" * 64, "labels": 1, "mirror": "nowhere"}))
    monkeypatch.setattr(TX, "TAXONOMY_DIR", tmp_path)
    with pytest.raises(TX.TaxonomyError, match="hashes to"):
        TX.load("toy")
    assert TX.load("toy", verify=False) == {"a": "one"}


def test_load_refuses_a_label_count_the_card_does_not_claim(tmp_path,
                                                            monkeypatch):
    home = tmp_path / "toy"
    home.mkdir()
    body = json.dumps({"a": "one", "b": "two"})
    (home / "descriptions.json").write_text(body)
    (home / "card.json").write_text(json.dumps(
        {"sha256": TX.sha256_file(home / "descriptions.json"), "labels": 3}))
    monkeypatch.setattr(TX, "TAXONOMY_DIR", tmp_path)
    with pytest.raises(TX.TaxonomyError, match="carries 2 labels"):
        TX.load("toy")


def test_readable_is_one_rule_and_reversible_to_the_identifier():
    assert TX.readable("card_arrival") == "Card arrival"
    assert TX.readable("Refund_not_showing_up") == "Refund not showing up"
    # no acronym table: `atm` stays lowercase on purpose, and the artifact
    # says so rather than hiding an undeclared choice inside the arm
    assert TX.readable("atm_support") == "Atm support"
    for label in TX.load():
        assert TX.readable(label).lower().replace(" ", "_") == label.lower()


def test_option_text_carries_the_definition_and_falls_back_loudly():
    desc = TX.load()
    text = TX.option_text("card_arrival", desc)
    assert text.startswith("Card arrival — ")
    assert desc["card_arrival"] in text
    # a label the taxonomy does not cover gets the readable name only, not
    # a silent `label: label` that would make arm B part arm A
    assert TX.option_text("not_a_label", desc) == "Not a label"


def test_coverage_of_the_banking77_space_is_complete():
    desc = TX.load()
    cover = TX.coverage(list(desc))
    assert cover["complete"] and cover["labels_defined"] == 77
    assert TX.coverage(["card_arrival", "nope"])["missing"] == ["nope"]
