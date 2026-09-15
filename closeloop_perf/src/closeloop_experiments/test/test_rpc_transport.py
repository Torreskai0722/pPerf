"""Tests for kernel-capsule stream framing."""

import json

import pytest

from closeloop_experiments.rpc_transport import receive_json_line


class FragmentedConnection:
    """Return predetermined fragments from a socket-like receiver."""

    def __init__(self, fragments):
        """Store fragments in receive order."""
        self.fragments = list(fragments)

    def recv(self, _size):
        """Return the next fragment or simulate end of stream."""
        if not self.fragments:
            return b""
        return self.fragments.pop(0)


def test_receive_json_line_reassembles_fragmented_large_message():
    """Do not parse a stream until its terminating newline arrives."""
    expected = {"payload": "x" * 300_000}
    encoded = (json.dumps(expected) + "\n").encode("utf-8")
    fragments = [
        encoded[offset:offset + 7919]
        for offset in range(0, len(encoded), 7919)
    ]
    assert receive_json_line(FragmentedConnection(fragments)) == expected


def test_receive_json_line_rejects_eof_before_delimiter():
    """Reject an incomplete frame even when its JSON prefix is valid."""
    with pytest.raises(ConnectionError, match="before newline"):
        receive_json_line(FragmentedConnection([b'{"status": "ok"}']))


def test_receive_json_line_rejects_size_limit():
    """Bound memory consumed by a peer that never finishes its frame."""
    with pytest.raises(ValueError, match="size limit"):
        receive_json_line(
            FragmentedConnection([b"{" + b"x" * 32]),
            max_message_bytes=8,
        )
