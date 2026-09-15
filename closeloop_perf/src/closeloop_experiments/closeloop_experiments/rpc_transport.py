"""Framed JSON transport shared by kernel-capsule Unix sockets."""

import json
from typing import Any, Dict


MAX_RPC_MESSAGE_BYTES = 64 * 1024 * 1024


def receive_json_line(
    connection: Any,
    max_message_bytes: int = MAX_RPC_MESSAGE_BYTES,
) -> Dict[str, Any]:
    """Receive one newline-terminated JSON object from a stream socket."""
    message = bytearray()
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            raise ConnectionError(
                "kernel capsule RPC connection closed before newline"
            )
        message.extend(chunk)
        newline = message.find(b"\n")
        if newline >= 0:
            if message[newline + 1:].strip():
                raise ValueError(
                    "kernel capsule RPC contains trailing data"
                )
            encoded = bytes(message[:newline])
            break
        if len(message) > max_message_bytes:
            raise ValueError("kernel capsule RPC exceeds size limit")
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("kernel capsule RPC root must be an object")
    return value
