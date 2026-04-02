"""
SHM Writer — Creates binary-compatible SHM files matching Rust's SharedDeviceState layout.

The binary format must match the #[repr(C)] struct SharedDeviceState
defined in crates/hipflex-internal/src/shared_memory/mod.rs.

Binary layout:
  devices:       16 * DeviceEntry (16 * 96 = 1536 bytes)
  device_count:  u32 (4 bytes)
  _pad:          4 bytes alignment
  last_heartbeat: u64 (8 bytes)
  _padding:      512 bytes

Total file size: 2064 bytes

DeviceEntry (96 bytes):
  uuid:        [u8; 64]  — null-terminated UTF-8 string
  device_info: SharedDeviceInfo (24 bytes)
  is_active:   u32
  _pad:        4 bytes (implicit from repr(C) alignment)

SharedDeviceInfo (24 bytes):
  mem_limit:           u64  (offset +0)
  pod_memory_used:     u64  (offset +8)
  effective_mem_limit: u64  (offset +16)  — 0 = not yet computed
"""

import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Constants matching Rust definitions ──

MAX_DEVICES = 16
MAX_UUID_LEN = 64
# DeviceEntry = UUID(64) + SharedDeviceInfo(24) + IsActive(4) + pad(4)
RUST_DEVICE_ENTRY_SIZE = 96

# SharedDeviceInfo size (all fields)
RUST_DEVICE_INFO_SIZE = 24

# Padding at end of SharedDeviceState
RUST_STATE_PADDING_SIZE = 512

# Total file size
# devices(16 * 96) + device_count(4) + pad(4) + last_heartbeat(8) + padding(512)
RUST_SHARED_DEVICE_STATE_TOTAL_SIZE = (
    MAX_DEVICES * RUST_DEVICE_ENTRY_SIZE + 4 + 4 + 8 + RUST_STATE_PADDING_SIZE
)

assert RUST_SHARED_DEVICE_STATE_TOTAL_SIZE == 2064, (
    f"Total size mismatch: {RUST_SHARED_DEVICE_STATE_TOTAL_SIZE} != 2064"
)

# ── Offsets within the file ──

# Offset of devices array within the file
DEVICES_OFFSET = 0

# Offset of device_count within the file
DEVICE_COUNT_OFFSET = MAX_DEVICES * RUST_DEVICE_ENTRY_SIZE

# Offset of last_heartbeat within the file
# device_count(4) + pad(4) = 8 bytes between device_count and last_heartbeat
HEARTBEAT_OFFSET = DEVICE_COUNT_OFFSET + 4 + 4


@dataclass
class DeviceSpec:
    """Specification for a single device in the SHM file."""

    uuid: str
    mem_limit: int  # bytes
    pod_memory_used: int = 0
    is_active: bool = True
    device_idx: Optional[int] = None  # If None, auto-assigned sequentially


def _pack_uuid(uuid_str: str) -> bytes:
    """Pack a UUID string into a 64-byte null-terminated buffer."""
    encoded = uuid_str.encode("utf-8")
    if len(encoded) >= MAX_UUID_LEN:
        encoded = encoded[: MAX_UUID_LEN - 1]
    return encoded.ljust(MAX_UUID_LEN, b"\x00")


def _pack_device_info(spec: DeviceSpec) -> bytes:
    """Pack SharedDeviceInfo into 24 bytes.

    Layout (all little-endian):
      u64 mem_limit
      u64 pod_memory_used
      u64 effective_mem_limit  (0 = not yet computed)
    """
    return struct.pack(
        "<Q Q Q",
        spec.mem_limit,
        spec.pod_memory_used,
        0,  # effective_mem_limit (0 = not yet computed, limiter updates at runtime)
    )


def _pack_device_entry(spec: DeviceSpec) -> bytes:
    """Pack a DeviceEntry into 96 bytes.

    Layout:
      [u8; 64]         uuid
      SharedDeviceInfo  device_info (24 bytes)
      u32              is_active
      [u8; 4]          padding (repr(C) alignment)
    """
    uuid_bytes = _pack_uuid(spec.uuid)
    info_bytes = _pack_device_info(spec)
    is_active = 1 if spec.is_active else 0
    tail = struct.pack("<I", is_active) + b"\x00" * 4  # is_active + padding

    entry = uuid_bytes + info_bytes + tail
    assert len(entry) == RUST_DEVICE_ENTRY_SIZE, (
        f"DeviceEntry size {len(entry)} != {RUST_DEVICE_ENTRY_SIZE}"
    )
    return entry


def create_shm_file(path: str, devices: List[DeviceSpec]) -> None:
    """Create a SHM file matching the Rust SharedDeviceState binary layout.

    Args:
        path: File path to write (this becomes the 'shm' file the crate mmaps).
        devices: List of device specifications. Max 16 devices.

    The file is created with size RUST_SHARED_DEVICE_STATE_TOTAL_SIZE (2064 bytes).
    All regions not explicitly written (padding) are zero-filled.
    """
    if len(devices) > MAX_DEVICES:
        raise ValueError(f"Too many devices: {len(devices)} > {MAX_DEVICES}")

    # Start with a zero-filled buffer
    buf = bytearray(RUST_SHARED_DEVICE_STATE_TOTAL_SIZE)

    # Write device entries
    for i, spec in enumerate(devices):
        idx = spec.device_idx if spec.device_idx is not None else i
        if idx >= MAX_DEVICES:
            raise ValueError(f"Device index {idx} >= {MAX_DEVICES}")
        offset = DEVICES_OFFSET + idx * RUST_DEVICE_ENTRY_SIZE
        entry_bytes = _pack_device_entry(spec)
        buf[offset : offset + RUST_DEVICE_ENTRY_SIZE] = entry_bytes

    # Write device_count (u32)
    struct.pack_into("<I", buf, DEVICE_COUNT_OFFSET, len(devices))

    # Write last_heartbeat (u64) — current unix timestamp
    heartbeat = int(time.time())
    struct.pack_into("<Q", buf, HEARTBEAT_OFFSET, heartbeat)

    # Write to file
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf)


def update_heartbeat(path: str, timestamp: Optional[int] = None) -> None:
    """Update the last_heartbeat field in an existing SHM file.

    Args:
        path: Path to the SHM file.
        timestamp: Unix timestamp to write. Defaults to current time.
    """
    if timestamp is None:
        timestamp = int(time.time())

    with open(path, "r+b") as f:
        f.seek(HEARTBEAT_OFFSET)
        f.write(struct.pack("<Q", timestamp))


def read_heartbeat(path: str) -> int:
    """Read the last_heartbeat value from an existing SHM file.

    Returns:
        The heartbeat timestamp as an integer.
    """
    with open(path, "rb") as f:
        f.seek(HEARTBEAT_OFFSET)
        return struct.unpack("<Q", f.read(8))[0]


def read_pod_memory_used(path: str, device_idx: int) -> int:
    """Read pod_memory_used for a specific device from the SHM file.

    Args:
        path: Path to the SHM file.
        device_idx: Index of the device (0-15).

    Returns:
        The pod_memory_used value in bytes.
    """
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    # pod_memory_used is at offset 8 within SharedDeviceInfo
    # SharedDeviceInfo starts at offset 64 within DeviceEntry (after uuid)
    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_SIZE
    info_offset = device_offset + MAX_UUID_LEN  # skip uuid
    pod_mem_offset = info_offset + 8  # mem_limit(8) = 8

    with open(path, "rb") as f:
        f.seek(pod_mem_offset)
        return struct.unpack("<Q", f.read(8))[0]


def read_mem_limit(path: str, device_idx: int) -> int:
    """Read mem_limit for a specific device from the SHM file.

    Args:
        path: Path to the SHM file.
        device_idx: Index of the device (0-15).

    Returns:
        The mem_limit value in bytes.
    """
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_SIZE
    info_offset = device_offset + MAX_UUID_LEN
    mem_limit_offset = info_offset  # mem_limit is first field

    with open(path, "rb") as f:
        f.seek(mem_limit_offset)
        return struct.unpack("<Q", f.read(8))[0]


def read_device_count(path: str) -> int:
    """Read the device_count from the SHM file."""
    with open(path, "rb") as f:
        f.seek(DEVICE_COUNT_OFFSET)
        return struct.unpack("<I", f.read(4))[0]


def read_device_uuid(path: str, device_idx: int) -> str:
    """Read the UUID for a specific device from the SHM file."""
    if device_idx >= MAX_DEVICES:
        raise ValueError(f"Device index {device_idx} >= {MAX_DEVICES}")

    device_offset = DEVICES_OFFSET + device_idx * RUST_DEVICE_ENTRY_SIZE

    with open(path, "rb") as f:
        f.seek(device_offset)
        uuid_bytes = f.read(MAX_UUID_LEN)

    # Find null terminator
    null_pos = uuid_bytes.find(b"\x00")
    if null_pos >= 0:
        uuid_bytes = uuid_bytes[:null_pos]
    return uuid_bytes.decode("utf-8")
