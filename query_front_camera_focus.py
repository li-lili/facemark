"""Read-only camera-1 focus capabilities. No video graph, frames, or setters.

Uses IAMCameraControl.GetRange and Get only. A reported manual capability is
not proof that a future manual setting will persist across mode changes.
"""

import ctypes as C
from datetime import datetime
import json

import benchmark_front_camera as bench
import query_front_camera_modes as dshow


def inspect_focus(source, ole):
    # IAMCameraControl: IUnknown slots 0..2; GetRange=3; Set=4; Get=5.
    control = dshow.query_interface(source, "c6e13370-30ac-11d0-a18c-00a0c9118956")
    if not control:
        return {"interface_supported": False}
    try:
        focus_property = 6  # CameraControl_Focus
        minimum, maximum, step, default, capabilities = [dshow.LONG() for _ in range(5)]
        hr_range = dshow.method(
            control, 3, dshow.LONG, [dshow.LONG] + [dshow.PTR] * 5,
            focus_property, *(C.byref(v) for v in
                              (minimum, maximum, step, default, capabilities)))
        value, flags = dshow.LONG(), dshow.LONG()
        hr_get = dshow.method(control, 5, dshow.LONG,
                             [dshow.LONG, dshow.PTR, dshow.PTR],
                             focus_property, C.byref(value), C.byref(flags))
        result = {
            "interface_supported": True,
            "get_range_hresult": f"0x{hr_range & 0xffffffff:08X}",
            "get_hresult": f"0x{hr_get & 0xffffffff:08X}",
        }
        if hr_range >= 0:
            result.update({
                "minimum": minimum.value, "maximum": maximum.value,
                "step": step.value, "default": default.value,
                "capability_flags": capabilities.value,
                "supports_auto": bool(capabilities.value & 1),
                "supports_manual": bool(capabilities.value & 2),
            })
        if hr_get >= 0:
            result.update({
                "current_value": value.value, "current_flags": flags.value,
                "currently_auto": bool(flags.value & 1),
                "currently_manual": bool(flags.value & 2),
            })
        return result
    finally:
        dshow.release(control)


def main():
    before = bench.protected_hashes()
    report = dshow.query_camera_one(filter_inspector=inspect_focus)
    report["focus"] = report.pop("filter_inspection")
    report.pop("modes", None)
    report["method"] = "Read-only IAMCameraControl.GetRange/Get; no Set, SetFormat, graph Run or capture"
    report["queried_at"] = datetime.now().astimezone().isoformat()
    after = bench.protected_hashes()
    report["protected_files_changed"] = [
        key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    destination = bench.ROOT / "benchmark_results" / (
        "camera1_focus_capabilities_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPORT", destination)


if __name__ == "__main__":
    main()
