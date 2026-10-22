"""Read-only DirectShow device enumeration and camera-1 IAMStreamConfig query.

No graph is run, no SetFormat or camera-control setters are called. Only the
index-1 filter is bound; other devices are enumerated as metadata, not opened.
Uses Windows ctypes COM directly, with no additional package dependencies.
"""

import ctypes as C
import json
import uuid
from datetime import datetime

import benchmark_front_camera as bench

PTR = C.c_void_p
LONG = C.c_int32
ULONG = C.c_uint32


class GUID(C.Structure):
    _fields_ = [("bytes", C.c_ubyte * 16)]

    def __init__(self, text):
        super().__init__()
        C.memmove(C.addressof(self), uuid.UUID(text).bytes_le, 16)

    def text(self):
        return str(uuid.UUID(bytes_le=bytes(self.bytes)))


class MediaType(C.Structure):
    _fields_ = [("major", GUID), ("subtype", GUID), ("fixed", LONG),
                ("temporal", LONG), ("sample_size", ULONG), ("format_type", GUID),
                ("unknown", PTR), ("format_size", ULONG), ("format_data", PTR)]


class Size(C.Structure):
    _fields_ = [("cx", LONG), ("cy", LONG)]


class VideoCaps(C.Structure):
    _fields_ = [
        ("guid", GUID), ("standard", ULONG), ("input_size", Size),
        ("min_crop", Size), ("max_crop", Size),
        ("crop_gran_x", LONG), ("crop_gran_y", LONG),
        ("crop_align_x", LONG), ("crop_align_y", LONG),
        ("min_output", Size), ("max_output", Size),
        ("out_gran_x", LONG), ("out_gran_y", LONG),
        ("stretch_x", LONG), ("stretch_y", LONG),
        ("shrink_x", LONG), ("shrink_y", LONG),
        ("min_interval", C.c_int64), ("max_interval", C.c_int64),
        ("min_bitrate", LONG), ("max_bitrate", LONG),
    ]


def method(ptr, slot, restype, types, *args):
    table = C.cast(ptr, C.POINTER(C.POINTER(PTR))).contents
    return C.WINFUNCTYPE(restype, PTR, *types)(table[slot])(ptr, *args)


def checked(hr, context):
    if hr < 0:
        raise OSError(f"{context}: HRESULT 0x{hr & 0xffffffff:08X}")
    return hr


def release(ptr):
    if ptr:
        method(ptr, 2, ULONG, [])


def query_interface(ptr, iid):
    result = PTR()
    guid = GUID(iid)
    hr = method(ptr, 0, LONG, [PTR, PTR], C.byref(guid), C.byref(result))
    return result if hr >= 0 else None


def property_text(moniker, name):
    bag = PTR()
    guid = GUID("55272a00-42cb-11ce-8135-00aa004bb851")
    hr = method(moniker, 9, LONG, [PTR, PTR, PTR, PTR],
                None, None, C.byref(guid), C.byref(bag))
    if hr < 0:
        return None
    value = (C.c_uint64 * 3)()  # VARIANT is 24 bytes on Win64, BSTR at offset 8.
    try:
        hr = method(bag, 3, LONG, [C.c_wchar_p, PTR, PTR], name, value, None)
        if hr >= 0 and C.c_uint16.from_buffer(value).value == 8:
            address = PTR.from_buffer(value, 8).value
            return C.wstring_at(address) if address else ""
        return None
    finally:
        C.OleDLL("oleaut32").VariantClear(C.byref(value))
        release(bag)


def free_media(ole, pointer):
    if pointer:
        media = C.cast(pointer, C.POINTER(MediaType)).contents
        if media.format_data:
            ole.CoTaskMemFree(media.format_data)
        release(media.unknown)
        ole.CoTaskMemFree(pointer)


def filter_modes(filter_ptr, ole):
    enum = PTR()
    checked(method(filter_ptr, 10, LONG, [PTR], C.byref(enum)), "EnumPins")
    rows = []
    try:
        while True:
            pin = PTR()
            fetched = ULONG()
            if method(enum, 3, LONG, [ULONG, PTR, PTR], 1,
                      C.byref(pin), C.byref(fetched)) != 0:
                break
            config = None
            try:
                direction = LONG()
                checked(method(pin, 9, LONG, [PTR], C.byref(direction)), "QueryDirection")
                if direction.value != 1:
                    continue
                pin_id_ptr = PTR()
                checked(method(pin, 10, LONG, [PTR], C.byref(pin_id_ptr)), "QueryId")
                try:
                    pin_id = C.wstring_at(pin_id_ptr)
                finally:
                    ole.CoTaskMemFree(pin_id_ptr)
                config = query_interface(pin, "c6e13340-30ac-11d0-a18c-00a0c9118956")
                if not config:
                    continue
                count, size = LONG(), LONG()
                checked(method(config, 5, LONG, [PTR, PTR],
                               C.byref(count), C.byref(size)), "GetNumberOfCapabilities")
                if not 0 < size.value < 65536 or not 0 <= count.value <= 10000:
                    raise RuntimeError("Unexpected capability buffer dimensions")
                for index in range(count.value):
                    media_ptr = PTR()
                    buffer = C.create_string_buffer(max(size.value, C.sizeof(VideoCaps)))
                    hr = method(config, 6, LONG, [LONG, PTR, PTR],
                                index, C.byref(media_ptr), buffer)
                    try:
                        checked(hr, "GetStreamCaps")
                        media = C.cast(media_ptr, C.POINTER(MediaType)).contents
                        format_type = media.format_type.text()
                        bitmap_offset = {
                            "05589f80-c356-11ce-bf01-00aa0055595a": 48,
                            "f72a76a0-eb0a-11d0-ace4-0000c0cc16ba": 72,
                        }.get(format_type)
                        if bitmap_offset is None or media.format_size < bitmap_offset + 40:
                            continue
                        data = C.string_at(media.format_data, media.format_size)
                        import struct
                        width, height = struct.unpack_from("<ii", data, bitmap_offset + 4)
                        compression = struct.unpack_from("<I", data, bitmap_offset + 16)[0]
                        interval = struct.unpack_from("<q", data, 40)[0]
                        caps = C.cast(buffer, C.POINTER(VideoCaps)).contents
                        row = {
                            "pin_id": pin_id,
                            "capability_index": index, "width": width, "height": abs(height),
                            "fourcc": "".join(chr(compression >> (8 * i) & 255) for i in range(4)),
                            "subtype": media.subtype.text(),
                            "default_interval_100ns": interval,
                            "default_fps": 1e7 / interval if interval > 0 else None,
                            "caps_buffer_size": size.value,
                        }
                        if size.value >= C.sizeof(VideoCaps):
                            row.update({
                                "min_interval_100ns": caps.min_interval,
                                "max_interval_100ns": caps.max_interval,
                                "min_fps": 1e7 / caps.max_interval if caps.max_interval > 0 else None,
                                "max_fps": 1e7 / caps.min_interval if caps.min_interval > 0 else None,
                            })
                        rows.append(row)
                    finally:
                        free_media(ole, media_ptr)
            finally:
                release(config)
                release(pin)
    finally:
        release(enum)
    return rows


def query_camera_one(filter_inspector=None):
    bench.preflight()
    ole = C.OleDLL("ole32")
    ole.CoInitializeEx.argtypes = [PTR, ULONG]
    ole.CoInitializeEx.restype = LONG
    ole.CoCreateInstance.argtypes = [PTR, PTR, ULONG, PTR, PTR]
    ole.CoCreateInstance.restype = LONG
    ole.CoTaskMemFree.argtypes = [PTR]
    checked(ole.CoInitializeEx(None, 2), "CoInitializeEx")
    dev_enum, monikers = PTR(), PTR()
    result = {"camera_index": 1, "devices": [], "modes": [],
              "method": "Read-only IAMStreamConfig.GetStreamCaps; no SetFormat, no running graph"}
    try:
        clsid = GUID("62be5d10-60eb-11d0-bd3b-00a0c911ce86")
        iid = GUID("29840822-5b84-11d0-bd3b-00a0c911ce86")
        checked(ole.CoCreateInstance(C.byref(clsid), None, 1, C.byref(iid),
                                    C.byref(dev_enum)), "SystemDeviceEnum")
        category = GUID("860bb310-5d01-11d0-bd3b-00a0c911ce86")
        hr = method(dev_enum, 3, LONG, [PTR, PTR, ULONG],
                    C.byref(category), C.byref(monikers), 0)
        checked(hr, "CreateClassEnumerator")
        if hr != 0:
            raise RuntimeError("No DirectShow video input devices")
        index = 0
        while True:
            moniker = PTR()
            fetched = ULONG()
            if method(monikers, 3, LONG, [ULONG, PTR, PTR], 1,
                      C.byref(moniker), C.byref(fetched)) != 0:
                break
            try:
                device = {"index": index, "name": property_text(moniker, "FriendlyName"),
                          "device_path": property_text(moniker, "DevicePath")}
                result["devices"].append(device)
                if index == 1:
                    source = PTR()
                    iid = GUID("56a86895-0ad4-11ce-b03a-0020af0ba770")
                    checked(method(moniker, 8, LONG, [PTR, PTR, PTR, PTR],
                                   None, None, C.byref(iid), C.byref(source)), "Bind camera 1")
                    try:
                        result["selected_device"] = device
                        if filter_inspector is None:
                            result["modes"] = filter_modes(source, ole)
                        else:
                            result["filter_inspection"] = filter_inspector(source, ole)
                    finally:
                        release(source)
                index += 1
            finally:
                release(moniker)
        if "selected_device" not in result:
            raise RuntimeError("DirectShow camera index 1 does not exist")
        return result
    finally:
        release(monikers)
        release(dev_enum)
        ole.CoUninitialize()


if __name__ == "__main__":
    before = bench.protected_hashes()
    report = query_camera_one()
    report["queried_at"] = datetime.now().astimezone().isoformat()
    after = bench.protected_hashes()
    report["protected_files_changed"] = [k for k in before if before[k] != after.get(k)]
    destination = bench.ROOT / "benchmark_results" / "camera1_supported_modes.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"devices": report["devices"], "modes_1080p": [
        row for row in report["modes"] if (row["width"], row["height"]) == (1920, 1080)
    ], "protected_files_changed": report["protected_files_changed"]},
        ensure_ascii=False, indent=2))
    print("REPORT", destination)
