"""Tests for acervo.layers — Layer enum and NodeMeta dataclass."""

from acervo.layers import Layer, NodeMeta


class TestLayer:
    def test_layer_values(self):
        assert Layer.UNIVERSAL.value == 1
        assert Layer.PERSONAL.value == 2

    def test_layer_names(self):
        assert Layer.UNIVERSAL.name == "UNIVERSAL"
        assert Layer.PERSONAL.name == "PERSONAL"


class TestNodeMeta:
    def test_personal_factory(self):
        meta = NodeMeta.personal(owner="Sandy")
        assert meta.layer == Layer.PERSONAL
        assert meta.owner == "Sandy"
        assert meta.source == "user_assertion"
        assert meta.confidence_for_owner == 1.0
        assert meta.status == "complete"
        assert meta.pending_fields == []

    def test_universal_factory(self):
        meta = NodeMeta.universal()
        assert meta.layer == Layer.UNIVERSAL
        assert meta.owner is None
        assert meta.source == "world"
        assert meta.confidence_for_owner == 1.0
        assert meta.status == "complete"

    def test_incomplete_factory(self):
        meta = NodeMeta.incomplete(owner="Sandy", pending=["type"])
        assert meta.layer == Layer.PERSONAL
        assert meta.owner == "Sandy"
        assert meta.status == "incomplete"
        assert meta.pending_fields == ["type"]

    def test_to_dict(self):
        meta = NodeMeta.personal(owner="Sandy")
        d = meta.to_dict()
        assert d["layer"] == "PERSONAL"
        assert d["owner"] == "Sandy"
        assert d["source"] == "user_assertion"
        assert d["confidence_for_owner"] == 1.0
        assert d["status"] == "complete"
        assert d["pending_fields"] == []

    def test_from_dict_roundtrip(self):
        original = NodeMeta.personal(owner="Sandy")
        restored = NodeMeta.from_dict(original.to_dict())
        assert restored.layer == original.layer
        assert restored.owner == original.owner
        assert restored.source == original.source
        assert restored.status == original.status

    def test_from_dict_legacy_missing_fields(self):
        """Legacy nodes without owner should deserialize cleanly."""
        legacy = {"layer": "PERSONAL", "source": "user_assertion"}
        meta = NodeMeta.from_dict(legacy)
        assert meta.layer == Layer.PERSONAL
        assert meta.owner is None
        assert meta.status == "complete"

    def test_from_dict_unknown_layer(self):
        """Unknown layer names should default to PERSONAL."""
        meta = NodeMeta.from_dict({"layer": "NONEXISTENT"})
        assert meta.layer == Layer.PERSONAL


# ── User-requested tests (explicit constructor usage) ──


def test_node_meta_complete():
    meta = NodeMeta(
        layer=Layer.PERSONAL,
        owner="Sandy",
        source="user_assertion",
        confidence_for_owner=1.0,
        status="complete",
    )
    assert meta.layer == Layer.PERSONAL
    assert meta.pending_fields == []


def test_node_meta_incomplete():
    meta = NodeMeta(
        layer=Layer.PERSONAL,
        owner="Sandy",
        source="user_assertion",
        confidence_for_owner=1.0,
        status="incomplete",
        pending_fields=["tipo", "ubicacion"],
    )
    assert meta.status == "incomplete"
    assert "tipo" in meta.pending_fields


def test_universal_node_has_no_owner():
    meta = NodeMeta(
        layer=Layer.UNIVERSAL,
        owner=None,
        source="world",
        confidence_for_owner=1.0,
        status="complete",
    )
    assert meta.owner is None
    assert meta.layer == Layer.UNIVERSAL
