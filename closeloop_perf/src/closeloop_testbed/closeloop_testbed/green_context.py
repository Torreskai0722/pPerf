"""Minimal CUDA Driver API binding for per-process Green Contexts."""

import ctypes
from dataclasses import dataclass


CU_DEV_RESOURCE_TYPE_SM = 1
CU_GREEN_CTX_DEFAULT_STREAM = 1


class GreenContextError(RuntimeError):
    """Raised when a requested Green Context cannot be activated."""

    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}


class _SmResource(ctypes.Structure):
    _fields_ = [("sm_count", ctypes.c_uint)]


class _ResourceValue(ctypes.Union):
    _fields_ = [("sm", _SmResource), ("_oversize", ctypes.c_ubyte * 48)]


class _DeviceResource(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("type", ctypes.c_int),
        ("_internal_padding", ctypes.c_ubyte * 92),
        ("value", _ResourceValue),
    ]


class CudaDriver:
    """Expose only the CUDA calls needed to create one Green Context."""

    def __init__(self, library=None):
        self.library = library or ctypes.CDLL("libcuda.so.1")
        resource_pointer = ctypes.POINTER(_DeviceResource)
        signatures = {
            "cuInit": ([ctypes.c_uint], ctypes.c_int),
            "cuDeviceGet": ([ctypes.POINTER(ctypes.c_int), ctypes.c_int],
                            ctypes.c_int),
            "cuDeviceGetDevResource": (
                [ctypes.c_int, resource_pointer, ctypes.c_int], ctypes.c_int
            ),
            "cuDevSmResourceSplitByCount": (
                [resource_pointer, ctypes.POINTER(ctypes.c_uint),
                 resource_pointer, resource_pointer, ctypes.c_uint,
                 ctypes.c_uint], ctypes.c_int
            ),
            "cuDevResourceGenerateDesc": (
                [ctypes.POINTER(ctypes.c_void_p), resource_pointer,
                 ctypes.c_uint], ctypes.c_int
            ),
            "cuGreenCtxCreate": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                 ctypes.c_int, ctypes.c_uint], ctypes.c_int
            ),
            "cuCtxFromGreenCtx": (
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p],
                ctypes.c_int
            ),
            "cuCtxSetCurrent": ([ctypes.c_void_p], ctypes.c_int),
            "cuCtxGetCurrent": (
                [ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int
            ),
            "cuGreenCtxGetDevResource": (
                [ctypes.c_void_p, resource_pointer, ctypes.c_int],
                ctypes.c_int
            ),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.library, name)
            function.argtypes = arguments
            function.restype = result

    def _check(self, result, operation):
        if result:
            raise GreenContextError(f"{operation} failed with CUDA error {result}")

    def initialize(self):
        """Initialize the CUDA driver."""
        self._check(self.library.cuInit(0), "cuInit")

    def device(self, index):
        """Return the configured CUDA device handle."""
        device = ctypes.c_int()
        self._check(self.library.cuDeviceGet(ctypes.byref(device), index),
                    "cuDeviceGet")
        return device.value

    def device_sm_resource(self, device):
        """Return all SM resources on a device."""
        resource = _DeviceResource()
        self._check(
            self.library.cuDeviceGetDevResource(
                device, ctypes.byref(resource), CU_DEV_RESOURCE_TYPE_SM
            ),
            "cuDeviceGetDevResource",
        )
        return resource

    def split_sm_resource(self, resource, sm_count):
        """Return every legal group having at least ``sm_count`` SMs."""
        count = ctypes.c_uint()
        self._check(
            self.library.cuDevSmResourceSplitByCount(
                None, ctypes.byref(count), ctypes.byref(resource), None, 0,
                sm_count
            ),
            "cuDevSmResourceSplitByCount(query)",
        )
        groups = (_DeviceResource * count.value)()
        capacity = ctypes.c_uint(count.value)
        self._check(
            self.library.cuDevSmResourceSplitByCount(
                groups, ctypes.byref(capacity), ctypes.byref(resource), None,
                0, sm_count
            ),
            "cuDevSmResourceSplitByCount",
        )
        return [groups[index] for index in range(capacity.value)]

    def create_green_context(self, device, resource):
        """Create a Green Context from one selected resource group."""
        descriptor = ctypes.c_void_p()
        self._check(
            self.library.cuDevResourceGenerateDesc(
                ctypes.byref(descriptor), ctypes.byref(resource), 1
            ),
            "cuDevResourceGenerateDesc",
        )
        green_context = ctypes.c_void_p()
        self._check(
            self.library.cuGreenCtxCreate(
                ctypes.byref(green_context), descriptor, device,
                CU_GREEN_CTX_DEFAULT_STREAM
            ),
            "cuGreenCtxCreate",
        )
        return green_context.value

    def context_from_green(self, green_context):
        """Convert a Green Context to its set-current handle."""
        context = ctypes.c_void_p()
        self._check(
            self.library.cuCtxFromGreenCtx(
                ctypes.byref(context), green_context
            ),
            "cuCtxFromGreenCtx",
        )
        return context.value

    def set_current(self, context):
        """Make a context current on the calling thread."""
        self._check(self.library.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    def current_context(self):
        """Return the context current on the calling thread."""
        context = ctypes.c_void_p()
        self._check(self.library.cuCtxGetCurrent(ctypes.byref(context)),
                    "cuCtxGetCurrent")
        return context.value

    def green_sm_count(self, green_context):
        """Return the actual SM count provisioned to a Green Context."""
        resource = _DeviceResource()
        self._check(
            self.library.cuGreenCtxGetDevResource(
                green_context, ctypes.byref(resource),
                CU_DEV_RESOURCE_TYPE_SM
            ),
            "cuGreenCtxGetDevResource",
        )
        return resource.sm.sm_count


@dataclass(frozen=True)
class GreenContextActivation:
    """Keep the driver and handles alive with their status evidence."""

    driver: object
    green_context: object
    context: object
    evidence: dict


def activate_green_context(configuration, gpu_index, driver=None):
    """Create, verify, and activate one configured Green Context."""
    requested = int(configuration["sm_count"])
    selected = int(configuration["partition_index"])
    evidence = {
        "requested_sm_count": requested,
        "selected_partition_index": selected,
        "active": False,
    }
    driver = driver or CudaDriver()
    try:
        driver.initialize()
        device = driver.device(int(gpu_index))
        resource = driver.device_sm_resource(device)
        evidence["device_sm_count"] = int(resource.sm.sm_count)
        groups = driver.split_sm_resource(resource, requested)
        evidence["available_partition_groups"] = len(groups)
        if selected >= len(groups):
            raise GreenContextError(
                f"partition_index {selected} is outside {len(groups)} groups"
            )
        green_context = driver.create_green_context(device, groups[selected])
        context = driver.context_from_green(green_context)
        driver.set_current(context)
        actual = int(driver.green_sm_count(green_context))
        current = driver.current_context() == context
        evidence.update({
            "actual_sm_count": actual,
            "current_context_active": current,
        })
        if actual != requested:
            raise GreenContextError(
                f"requested {requested} SMs but Green Context has {actual}"
            )
        if not current:
            raise GreenContextError("Green Context is not current")
        evidence["active"] = True
        evidence["error"] = None
        return GreenContextActivation(
            driver, green_context, context, evidence
        )
    except Exception as exc:
        evidence["error"] = str(exc)
        raise GreenContextError(str(exc), evidence) from exc
