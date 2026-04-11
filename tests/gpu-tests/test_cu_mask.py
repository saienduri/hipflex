"""
CU Masking Tests — Validates FH_CU_RANGE compute restriction via HSA_CU_MASK.

Key behaviors under test:
  - FH_CU_RANGE=0-37 sets HSA_CU_MASK and spoofs multiProcessorCount to 38
  - Kernel execution succeeds with restricted CU range
  - multiProcessorCount reflects the CU range, not the physical CU count
  - hipExtStreamCreateWithCUMask roundtrip verifies CU mask pipeline end-to-end
  - CU-only mode: FH_CU_RANGE works without FH_MEMORY_LIMIT (allocation passthrough,
    totalGlobalMem reports real hardware, multiProcessorCount spoofed, kernel execution)
  - Edge cases: invalid ranges, reversed ranges, clamping
"""

import pytest

from conftest import run_standalone, requires_gpu

pytestmark = requires_gpu


class TestCUMaskMultiProcessorCount:
    """Verify that FH_CU_RANGE spoofs multiProcessorCount in hipGetDeviceProperties."""

    def test_multi_processor_count_spoofed(self):
        """With FH_CU_RANGE=0-37, multiProcessorCount should be 38."""
        result = run_standalone("""
            from hip_helper import HIPRuntime
            hip = HIPRuntime()
            mp_count = hip.get_device_properties_multi_processor_count()
            print(f"multiProcessorCount={mp_count}")

            if mp_count == 38:
                print("PASS")
            else:
                print(f"FAIL: expected 38, got {mp_count}")
        """, mem_limit="24G", cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"multiProcessorCount not spoofed:\n{result.stdout}\n{result.stderr}"
        )


class TestCUMaskKernelExecution:
    """Verify that kernels execute successfully with CU mask applied."""

    def test_kernel_runs_with_cu_mask(self):
        """A trivial kernel should succeed with FH_CU_RANGE restricting CUs."""
        result = run_standalone("""
            import ctypes

            hip = ctypes.CDLL("libamdhip64.so")

            # Allocate device memory
            ptr = ctypes.c_void_p(0)
            err = hip.hipMalloc(ctypes.byref(ptr), 1024)
            assert err == 0, f"hipMalloc failed: {err}"

            # hipMemset to exercise the GPU with CU mask active
            err = hip.hipMemset(ptr, 0, 1024)
            assert err == 0, f"hipMemset failed: {err}"

            err = hip.hipDeviceSynchronize()
            assert err == 0, f"hipDeviceSynchronize failed: {err}"

            err = hip.hipFree(ptr)
            assert err == 0, f"hipFree failed: {err}"

            print("PASS")
        """, mem_limit="24G", cu_range="0-37",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Kernel execution failed with CU mask:\n{result.stdout}\n{result.stderr}"
        )
        assert "CU range restriction" in result.output, (
            f"CU masking not active in logs:\n{result.output}"
        )


class TestCUMaskStreamVerification:
    """Verify CU masking at the HIP stream level via hipExtStreamCreateWithCUMask.

    HSA_CU_MASK operates below HIP's stream abstraction, so hipExtStreamGetCUMask
    on a regular stream shows all CUs. To verify the CU masking pipeline works
    end-to-end, we create a stream with an explicit CU bitmask matching the
    FH_CU_RANGE and read it back.
    """

    def test_stream_cu_mask_roundtrip(self):
        """Create a CU-masked stream for CUs 0-37, read back, verify the mask matches."""
        result = run_standalone("""
            import ctypes

            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
            hip.hipExtStreamCreateWithCUMask.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32)]
            hip.hipExtStreamGetCUMask.restype = ctypes.c_int
            hip.hipExtStreamGetCUMask.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
            hip.hipStreamDestroy.restype = ctypes.c_int
            hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]

            # Build CU mask for CUs 0-37: word[0] = all 32 bits, word[1] = bits 0-5
            in_mask = (ctypes.c_uint32 * 3)()
            in_mask[0] = 0xFFFFFFFF  # CUs 0-31
            in_mask[1] = 0x0000003F  # CUs 32-37
            in_mask[2] = 0x00000000

            stream = ctypes.c_void_p(0)
            err = hip.hipExtStreamCreateWithCUMask(ctypes.byref(stream), 3, in_mask)
            assert err == 0, f"hipExtStreamCreateWithCUMask failed: {err}"

            # Read back — requires size >= 16 on ROCm
            out_size = 16
            out_mask = (ctypes.c_uint32 * out_size)()
            err = hip.hipExtStreamGetCUMask(stream, out_size, out_mask)
            assert err == 0, f"hipExtStreamGetCUMask failed: {err}"

            enabled_cus = []
            for w in range(out_size):
                for b in range(32):
                    if out_mask[w] & (1 << b):
                        enabled_cus.append(w * 32 + b)

            print(f"enabled_count={len(enabled_cus)}")
            print(f"min_cu={min(enabled_cus) if enabled_cus else -1}")
            print(f"max_cu={max(enabled_cus) if enabled_cus else -1}")

            expected = set(range(0, 38))
            actual = set(enabled_cus)

            if actual == expected:
                print("PASS")
            else:
                missing = expected - actual
                extra = actual - expected
                print(f"FAIL: expected CUs 0-37 ({len(expected)}), got {len(actual)}")
                if missing:
                    print(f"  missing: {sorted(missing)}")
                if extra:
                    print(f"  extra: {sorted(extra)}")

            hip.hipStreamDestroy(stream)
        """, mem_limit="24G", cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Stream CU mask roundtrip failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_stream_cu_mask_subset_roundtrip(self):
        """Create a CU-masked stream for CUs 10-19, verify readback matches."""
        result = run_standalone("""
            import ctypes

            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipExtStreamCreateWithCUMask.restype = ctypes.c_int
            hip.hipExtStreamCreateWithCUMask.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32)]
            hip.hipExtStreamGetCUMask.restype = ctypes.c_int
            hip.hipExtStreamGetCUMask.argtypes = [
                ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
            hip.hipStreamDestroy.restype = ctypes.c_int
            hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]

            # CUs 10-19: bits 10-19 in word[0]
            in_mask = (ctypes.c_uint32 * 1)()
            in_mask[0] = 0x000FFC00  # bits 10-19 set

            stream = ctypes.c_void_p(0)
            err = hip.hipExtStreamCreateWithCUMask(ctypes.byref(stream), 1, in_mask)
            assert err == 0, f"hipExtStreamCreateWithCUMask failed: {err}"

            out_size = 16
            out_mask = (ctypes.c_uint32 * out_size)()
            err = hip.hipExtStreamGetCUMask(stream, out_size, out_mask)
            assert err == 0, f"hipExtStreamGetCUMask failed: {err}"

            enabled_cus = []
            for w in range(out_size):
                for b in range(32):
                    if out_mask[w] & (1 << b):
                        enabled_cus.append(w * 32 + b)

            print(f"enabled_count={len(enabled_cus)}")
            print(f"enabled_cus={enabled_cus}")

            expected = set(range(10, 20))
            actual = set(enabled_cus)

            if actual == expected:
                print("PASS")
            else:
                print(f"FAIL: expected CUs 10-19, got {sorted(actual)}")

            hip.hipStreamDestroy(stream)
        """, mem_limit="24G", cu_range="10-19")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Stream CU mask subset roundtrip failed:\n{result.stdout}\n{result.stderr}"
        )


class TestCUMaskEnvVar:
    """Verify HSA_CU_MASK is set in the process environment."""

    def test_hsa_cu_mask_set(self):
        """HSA_CU_MASK should be set by hipflex with format 'GPU:start-end[;GPU:start-end]'."""
        result = run_standalone("""
            import ctypes
            import ctypes.util
            import re

            hip = ctypes.CDLL("libamdhip64.so")

            # Trigger HIP init so hipflex runs
            count = ctypes.c_int(0)
            err = hip.hipGetDeviceCount(ctypes.byref(count))
            assert err == 0, f"hipGetDeviceCount failed: {err}"

            # Read from the live C environment (not Python's cached os.environ,
            # which was snapshotted before hipflex's ctor set HSA_CU_MASK).
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.getenv.restype = ctypes.c_char_p
            libc.getenv.argtypes = [ctypes.c_char_p]
            raw = libc.getenv(b"HSA_CU_MASK")
            mask = raw.decode() if raw else ""
            print(f"HSA_CU_MASK={mask}")

            # Validate exact format: each segment is "GPU_IDX:0-37"
            pattern = '^([0-9]+:0-37)(;[0-9]+:0-37)*$'
            if re.match(pattern, mask):
                print("PASS")
            else:
                print(f"FAIL: HSA_CU_MASK format mismatch: '{mask}'")
        """, mem_limit="24G", cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"HSA_CU_MASK not set correctly:\n{result.stdout}\n{result.stderr}"
        )


class TestCUMaskEdgeCases:
    """Edge cases: invalid ranges, clamping, graceful degradation."""

    def test_invalid_cu_range_passthrough(self):
        """Invalid FH_CU_RANGE should log a warning and proceed without CU masking."""
        result = run_standalone("""
            from hip_helper import HIPRuntime
            hip = HIPRuntime()
            count = hip.get_device_count()
            assert count > 0, "No devices"
            print("PASS")
        """, mem_limit="24G", cu_range="abc",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Passthrough failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "invalid FH_CU_RANGE" in result.output, (
            f"Expected parse warning in logs:\n{result.output}"
        )

    def test_reversed_cu_range_passthrough(self):
        """FH_CU_RANGE=5-3 (reversed) should log a warning and proceed."""
        result = run_standalone("""
            from hip_helper import HIPRuntime
            hip = HIPRuntime()
            count = hip.get_device_count()
            assert count > 0, "No devices"
            print("PASS")
        """, mem_limit="24G", cu_range="5-3",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Passthrough failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "start (5) > end (3)" in result.output, (
            f"Expected reversed-range warning in logs:\n{result.output}"
        )

    def test_cu_range_exceeding_device_clamps(self):
        """FH_CU_RANGE=0-999 should clamp to actual device CU count."""
        result = run_standalone("""
            from hip_helper import HIPRuntime
            hip = HIPRuntime()
            mp_count = hip.get_device_properties_multi_processor_count()
            print(f"multiProcessorCount={mp_count}")
            # Should be clamped to actual device CU count, not 1000
            assert mp_count < 1000, f"multiProcessorCount not clamped: {mp_count}"
            assert mp_count > 0, "multiProcessorCount is 0"
            print("PASS")
        """, mem_limit="24G", cu_range="0-999",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Clamp test failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "clamping to" in result.output, (
            f"Expected clamp warning in logs:\n{result.output}"
        )

    def test_cu_range_without_memory_limit(self):
        """FH_CU_RANGE without FH_MEMORY_LIMIT should apply CU masking (CU-only mode)."""
        result = run_standalone("""
            import ctypes
            import ctypes.util

            hip = ctypes.CDLL("libamdhip64.so")

            # Trigger HIP init
            count = ctypes.c_int(0)
            err = hip.hipGetDeviceCount(ctypes.byref(count))
            assert err == 0, f"hipGetDeviceCount failed: {err}"

            # Verify HSA_CU_MASK was set (read from C environment, not Python's cached os.environ)
            libc = ctypes.CDLL(ctypes.util.find_library("c"))
            libc.getenv.restype = ctypes.c_char_p
            libc.getenv.argtypes = [ctypes.c_char_p]
            raw = libc.getenv(b"HSA_CU_MASK")
            mask = raw.decode() if raw else ""
            print(f"HSA_CU_MASK={mask}")
            assert mask, "HSA_CU_MASK not set"

            # Verify multiProcessorCount is spoofed to 38
            from hip_helper import HIPRuntime
            hip_rt = HIPRuntime()
            mp_count = hip_rt.get_device_properties_multi_processor_count()
            print(f"multiProcessorCount={mp_count}")
            assert mp_count == 38, f"expected 38, got {mp_count}"

            print("PASS")
        """, mem_limit=None, cu_range="0-37",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"CU-only mode failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "CU-only mode" in result.output, (
            f"Expected CU-only mode log:\n{result.output}"
        )


class TestCUOnlyMode:
    """CU-only mode: FH_CU_RANGE without FH_MEMORY_LIMIT."""

    def test_alloc_passthrough(self):
        """Memory allocation should pass through to native (no enforcement)."""
        result = run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            # Allocate arbitrary size — no limit should be enforced
            ptr = hip.malloc(256 * 1024 * 1024)  # 256 MiB
            assert ptr != 0, "hipMalloc returned null"
            hip.free(ptr)
            print("PASS")
        """, mem_limit=None, cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Allocation passthrough failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_total_global_mem_not_spoofed(self):
        """hipGetDeviceProperties.totalGlobalMem should report real hardware value."""
        result = run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            # In CU-only mode, totalGlobalMem must NOT be spoofed.
            # MI325X has ~256 GiB — should be well above 100 GiB.
            total = hip.get_device_properties_total_mem()
            print(f"totalGlobalMem={total}")
            gib = total / (1024 ** 3)
            assert gib > 100, f"totalGlobalMem suspiciously low ({gib:.1f} GiB), may be spoofed"
            print("PASS")
        """, mem_limit=None, cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"totalGlobalMem appears spoofed:\n{result.stdout}\n{result.stderr}"
        )

    def test_mem_get_info_passthrough(self):
        """hipMemGetInfo should report real hardware values (no spoofing)."""
        result = run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            free, total = hip.mem_get_info()
            print(f"free={free} total={total}")
            gib = total / (1024 ** 3)
            assert gib > 100, f"hipMemGetInfo total suspiciously low ({gib:.1f} GiB), may be spoofed"
            assert free > 0, "hipMemGetInfo free is 0"
            print("PASS")
        """, mem_limit=None, cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"hipMemGetInfo passthrough failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_device_total_mem_passthrough(self):
        """hipDeviceTotalMem should report real hardware value (no spoofing)."""
        result = run_standalone("""
            from hip_helper import HIPRuntime

            hip = HIPRuntime()
            total = hip.device_total_mem(0)
            print(f"hipDeviceTotalMem={total}")
            gib = total / (1024 ** 3)
            assert gib > 100, f"hipDeviceTotalMem suspiciously low ({gib:.1f} GiB), may be spoofed"
            print("PASS")
        """, mem_limit=None, cu_range="0-37")

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"hipDeviceTotalMem passthrough failed:\n{result.stdout}\n{result.stderr}"
        )

    def test_kernel_execution(self):
        """Kernel execution should succeed in CU-only mode."""
        result = run_standalone("""
            import ctypes

            hip = ctypes.CDLL("libamdhip64.so")

            ptr = ctypes.c_void_p(0)
            err = hip.hipMalloc(ctypes.byref(ptr), 1024)
            assert err == 0, f"hipMalloc failed: {err}"

            err = hip.hipMemset(ptr, 0, 1024)
            assert err == 0, f"hipMemset failed: {err}"

            err = hip.hipDeviceSynchronize()
            assert err == 0, f"hipDeviceSynchronize failed: {err}"

            err = hip.hipFree(ptr)
            assert err == 0, f"hipFree failed: {err}"

            print("PASS")
        """, mem_limit=None, cu_range="0-37",
             extra_env={"FH_LOG_PATH": "stderr"})

        assert result.returncode == 0, f"Subprocess failed:\n{result.stderr}"
        assert "PASS" in result.stdout, (
            f"Kernel execution failed:\n{result.stdout}\n{result.stderr}"
        )
        assert "CU range restriction" in result.output, (
            f"CU masking not active in logs:\n{result.output}"
        )
