"""Tests for CUDA Green Context partition activation."""

from types import SimpleNamespace

import pytest

from closeloop_testbed.green_context import (
    GreenContextError,
    activate_green_context,
)


class FakeDriver:
    """Small driver stand-in with deterministic partition resources."""

    def __init__(self, group_counts=(14, 14, 14, 14), current=True):
        self.group_counts = group_counts
        self.keep_current = current
        self.current = None
        self.selected = None

    @staticmethod
    def _resource(sm_count):
        return SimpleNamespace(sm=SimpleNamespace(sm_count=sm_count))

    def initialize(self):
        """Record no state for fake initialization."""

    def device(self, index):
        """Return the input index as a fake device handle."""
        return index

    def device_sm_resource(self, _device):
        """Return a fake 56-SM device."""
        return self._resource(56)

    def split_sm_resource(self, _resource, _requested):
        """Return configured fake partition sizes."""
        return [self._resource(count) for count in self.group_counts]

    def create_green_context(self, _device, resource):
        """Retain the selected fake partition."""
        self.selected = resource
        return 101

    def context_from_green(self, _green_context):
        """Return a fake normal context handle."""
        return 202

    def set_current(self, context):
        """Set the fake current handle when enabled."""
        if self.keep_current:
            self.current = context

    def current_context(self):
        """Return the fake current handle."""
        return self.current

    def green_sm_count(self, _green_context):
        """Return the selected fake partition size."""
        return self.selected.sm.sm_count


def test_partition_selection_and_status_evidence():
    """The selected legal group is current and fully reported."""
    activation = activate_green_context(
        {"sm_count": 14, "partition_index": 1}, 0, FakeDriver()
    )
    assert activation.evidence == {
        "requested_sm_count": 14,
        "selected_partition_index": 1,
        "device_sm_count": 56,
        "available_partition_groups": 4,
        "actual_sm_count": 14,
        "current_context_active": True,
        "active": True,
        "error": None,
    }


def test_partition_index_and_actual_sm_failures_retain_evidence():
    """Invalid selection and driver rounding fail with partial status."""
    with pytest.raises(GreenContextError) as selected:
        activate_green_context(
            {"sm_count": 14, "partition_index": 4}, 0, FakeDriver()
        )
    assert selected.value.evidence["available_partition_groups"] == 4
    assert not selected.value.evidence["active"]
    assert "outside" in selected.value.evidence["error"]

    with pytest.raises(GreenContextError) as actual:
        activate_green_context(
            {"sm_count": 14, "partition_index": 0}, 0,
            FakeDriver((16, 16, 16))
        )
    assert actual.value.evidence["actual_sm_count"] == 16
    assert actual.value.evidence["current_context_active"]
    assert not actual.value.evidence["active"]
