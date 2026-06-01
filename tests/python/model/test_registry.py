"""Tests for midigpt.inference.model.registry (plan section 3.9).

NOTE: the plan says ``get_model_class`` raises ``KeyError`` on unknown arch,
but the actual implementation raises ``ValueError``. Tests are written
against the actual behavior, per hard-rule #1.
"""

from __future__ import annotations

import pytest

from midigpt.inference.model import registry as registry_module
from midigpt.inference.model.gpt2 import GPT2LMHeadModel
from midigpt.inference.model.registry import (
    REGISTRY,
    get_model_class,
    register,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry():
    """Snapshot REGISTRY before the test and restore exactly after.

    Lets us add/overwrite entries without leaking state across tests.
    """
    snapshot = dict(REGISTRY)
    try:
        yield REGISTRY
    finally:
        REGISTRY.clear()
        REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Built-in registrations
# ---------------------------------------------------------------------------


def test_registry_contains_gpt2_entry():
    assert "gpt2" in REGISTRY, f"expected 'gpt2' in REGISTRY, got keys={sorted(REGISTRY)}"
    assert REGISTRY["gpt2"] is GPT2LMHeadModel


def test_get_model_class_returns_gpt2_class():
    cls = get_model_class("gpt2")
    assert cls is GPT2LMHeadModel
    # The class declares its own arch identifier consistent with the key.
    assert getattr(cls, "arch", None) == "gpt2"


def test_get_model_class_is_idempotent():
    first = get_model_class("gpt2")
    second = get_model_class("gpt2")
    third = get_model_class("gpt2")
    assert first is second is third is GPT2LMHeadModel
    # Repeated lookup must not mutate the registry.
    assert "gpt2" in REGISTRY


def test_registry_module_exports_singleton_dict():
    # The module-level REGISTRY and the one accessed via the module attr are
    # the same object — registration mutates a single shared dict.
    assert registry_module.REGISTRY is REGISTRY


# ---------------------------------------------------------------------------
# Unknown arch
# ---------------------------------------------------------------------------


def test_get_model_class_unknown_arch_raises_value_error(isolated_registry):
    with pytest.raises(ValueError) as exc_info:
        get_model_class("definitely_not_a_real_arch_xyz")
    msg = str(exc_info.value)
    # Informative message must name the unknown arch and list what *is*
    # registered, so the user can diagnose typos.
    assert "definitely_not_a_real_arch_xyz" in msg
    assert "Unknown architecture" in msg
    assert "Registered" in msg
    # Every currently-registered arch should be enumerated in the message.
    for known in REGISTRY:
        assert known in msg


def test_get_model_class_empty_string_raises_value_error(isolated_registry):
    with pytest.raises(ValueError):
        get_model_class("")


# ---------------------------------------------------------------------------
# @register decorator
# ---------------------------------------------------------------------------


def test_register_decorator_adds_class_and_returns_it(isolated_registry):
    class Dummy:
        arch = "dummy_arch_for_test"

    returned = register("dummy_arch_for_test")(Dummy)

    assert returned is Dummy, "decorator must return the class unchanged"
    assert isolated_registry["dummy_arch_for_test"] is Dummy
    assert get_model_class("dummy_arch_for_test") is Dummy


def test_register_decorator_usable_as_class_decorator_syntax(isolated_registry):
    @register("syntax_test_arch")
    class Foo:
        arch = "syntax_test_arch"

    assert get_model_class("syntax_test_arch") is Foo


def test_register_overwrites_existing_entry(isolated_registry):
    class First:
        arch = "overwrite_test"

    class Second:
        arch = "overwrite_test"

    register("overwrite_test")(First)
    assert get_model_class("overwrite_test") is First

    # Source has no guard against re-registration — second wins silently.
    register("overwrite_test")(Second)
    assert get_model_class("overwrite_test") is Second
    assert get_model_class("overwrite_test") is not First


def test_register_does_not_affect_unrelated_entries(isolated_registry):
    pre_gpt2 = REGISTRY.get("gpt2")

    class Sibling:
        arch = "sibling_arch"

    register("sibling_arch")(Sibling)

    assert REGISTRY.get("gpt2") is pre_gpt2
    assert get_model_class("gpt2") is GPT2LMHeadModel


def test_isolated_registry_fixture_actually_restores(isolated_registry):
    # Sanity check that pollution from the decorator tests above cannot
    # leak: at the start of this test the registry is back to baseline.
    assert "dummy_arch_for_test" not in REGISTRY
    assert "syntax_test_arch" not in REGISTRY
    assert "overwrite_test" not in REGISTRY
    assert "sibling_arch" not in REGISTRY
    assert "gpt2" in REGISTRY
