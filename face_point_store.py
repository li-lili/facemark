"""face_point.json 模板与基线存储层。

该模块只负责 JSON 读写、旧文件迁移和模板分支管理；视觉算法和阈值判断留在调用方。
"""

import copy
import json
import os
from typing import Any, Optional

import numpy as np


DIR = os.path.dirname(os.path.abspath(__file__))
FACE_POINT_FILE = os.path.join(DIR, "face_point.json")
LEGACY_FACE_POINTS_FILE = os.path.join(DIR, "face_points.json")
FACE_POINTS_FILE = FACE_POINT_FILE
EYELID_BASELINE_FILE = os.path.join(DIR, "eyelid_baseline.json")
EYEBROW_BASELINE_FILE = os.path.join(DIR, "eyebrow_baseline.json")
EYEBROW_BASELINE_MALE_FILE = os.path.join(DIR, "eyebrow_baseline_male.json")
EYEBROW_BASELINE_FEMALE_FILE = os.path.join(DIR, "eyebrow_baseline_female.json")
MOUTH_BASELINE_FILE = os.path.join(DIR, "mouth_baseline.json")
LOWER_LIP_BASELINE_FILE = os.path.join(DIR, "lower_lip_baseline.json")
UPPER_LIP_BASELINE_FILE = os.path.join(DIR, "upper_lip_baseline.json")
MOUTH_CORNERS_BASELINE_FILE = os.path.join(DIR, "mouth_corners_baseline.json")
HEAD_POSITION_BASELINE_FILE = os.path.join(DIR, "head_position_baseline.json")
BASELINE_SECTIONS = (
    "eyelid",
    "eyebrow",
    "mouth",
    "lower_lip",
    "upper_lip",
    "mouth_corners",
    "head_position",
)
DEFAULT_TEMPLATE_NAME = "default"
DEFAULT_FRAME_WIDTH = 1920
DEFAULT_FRAME_HEIGHT = 1080


def _empty_template_data() -> dict[str, dict[str, Any]]:
    return {section: {} for section in BASELINE_SECTIONS}


def _default_face_point_data() -> dict[str, Any]:
    return {
        "version": 1,
        "active_template": DEFAULT_TEMPLATE_NAME,
        "common": {},
        "templates": {DEFAULT_TEMPLATE_NAME: _empty_template_data()},
    }


def _read_json_file(path: str) -> Optional[dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as json_file:
            return json.load(json_file)
    except Exception as exc:
        print(f"[FacePoint] Read failed {path}: {exc}")
        return None


def _write_json_file(path: str, data: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2, ensure_ascii=False)


def _normalize_face_point_data(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    normalized = _default_face_point_data()
    normalized.update({key: value for key, value in data.items() if key in ("version", "active_template", "common")})
    if not isinstance(normalized.get("common"), dict):
        normalized["common"] = {}

    src_templates = data.get("templates")
    if not isinstance(src_templates, dict):
        src_templates = data.get("gender", {})
        if data.get("active_gender") and not data.get("active_template"):
            normalized["active_template"] = str(data["active_gender"])
    if not isinstance(src_templates, dict):
        src_templates = {}

    normalized["templates"] = {}
    for template_name, template_data in src_templates.items():
        if not template_name:
            continue
        name = str(template_name)
        normalized["templates"][name] = _empty_template_data()
        if isinstance(template_data, dict):
            for section in BASELINE_SECTIONS:
                value = template_data.get(section, {})
                normalized["templates"][name][section] = value if isinstance(value, dict) else {}
    if not normalized["templates"]:
        normalized["templates"][DEFAULT_TEMPLATE_NAME] = _empty_template_data()
    if normalized["active_template"] not in normalized["templates"]:
        normalized["active_template"] = next(iter(normalized["templates"]))
    return normalized


def _legacy_face_common() -> dict[str, Any]:
    data = _read_json_file(LEGACY_FACE_POINTS_FILE)
    if not data:
        return {}
    try:
        common = {
            "nose": list(data["nose"]),
            "iris_left": list(data["iris_left"]),
            "iris_right": list(data["iris_right"]),
        }
        common["eye_line_y"] = float(
            data.get(
                "eye_line_y",
                (float(data["iris_left"][1]) + float(data["iris_right"][1])) / 2.0,
            )
        )
        return common
    except Exception as exc:
        print(f"[FacePoint] Legacy common migration failed: {exc}")
        return {}


def _legacy_section(section: str, path: str) -> dict[str, Any]:
    data = _read_json_file(path)
    if not data:
        return {}
    try:
        if section == "eyelid":
            return {"left_ear": float(data["left_ear"]), "right_ear": float(data["right_ear"])}
        if section == "eyebrow":
            required = ("left_brow_iris_gap", "right_brow_iris_gap", "left_slope", "right_slope")
            if not all(key in data for key in required):
                return {}
            return {
                "left_slope": float(data["left_slope"]),
                "right_slope": float(data["right_slope"]),
                "slope_symmetry": float(data.get("slope_symmetry", 0.0)),
                "left_brow_iris_gap": float(data["left_brow_iris_gap"]),
                "right_brow_iris_gap": float(data["right_brow_iris_gap"]),
            }
        if section == "mouth":
            return {"mar": float(data["mar"])}
        if section == "lower_lip":
            out: dict[str, Any] = {}
            if "llr" in data:
                out["llr"] = float(data["llr"])
            if "side_lower_tip" in data:
                out["side_lower_tip"] = list(data["side_lower_tip"])
            if "side_roi" in data:
                out["side_roi"] = list(data["side_roi"])
            return out
        if section == "upper_lip":
            out = {}
            if "ulr" in data:
                out["ulr"] = float(data["ulr"])
            if "side_upper_tip" in data:
                out["side_upper_tip"] = list(data["side_upper_tip"])
            if "side_roi" in data:
                out["side_roi"] = list(data["side_roi"])
            return out
        if section == "mouth_corners":
            return {
                "nose": list(data["nose"]),
                "corner_left": list(data["corner_left"]),
                "corner_right": list(data["corner_right"]),
            }
        if section == "head_position":
            return {
                "nose_px": list(data["nose_px"]),
                "eye_left": list(data["eye_left"]),
                "eye_right": list(data["eye_right"]),
                "frame_width": data.get("frame_width", DEFAULT_FRAME_WIDTH),
                "frame_height": data.get("frame_height", DEFAULT_FRAME_HEIGHT),
                "eye_distance": data.get("eye_distance", 0.0),
            }
    except Exception as exc:
        print(f"[FacePoint] Legacy {section} migration failed: {exc}")
    return {}


def _migrate_face_point_data() -> dict[str, Any]:
    data = _default_face_point_data()
    data["common"] = _legacy_face_common()
    default_template = data["active_template"]

    legacy_sections = {
        "eyelid": EYELID_BASELINE_FILE,
        "mouth": MOUTH_BASELINE_FILE,
        "lower_lip": LOWER_LIP_BASELINE_FILE,
        "upper_lip": UPPER_LIP_BASELINE_FILE,
        "mouth_corners": MOUTH_CORNERS_BASELINE_FILE,
        "head_position": HEAD_POSITION_BASELINE_FILE,
    }
    for section, path in legacy_sections.items():
        data["templates"][default_template][section] = _legacy_section(section, path)

    male_brow = _legacy_section("eyebrow", EYEBROW_BASELINE_MALE_FILE)
    female_brow = _legacy_section("eyebrow", EYEBROW_BASELINE_FEMALE_FILE)
    active_brow = _legacy_section("eyebrow", EYEBROW_BASELINE_FILE)
    if male_brow or active_brow:
        data["templates"][DEFAULT_TEMPLATE_NAME]["eyebrow"] = male_brow or active_brow
    if female_brow:
        data["templates"].setdefault("template_2", _empty_template_data())
        data["templates"]["template_2"]["eyebrow"] = female_brow
    return data


def load_face_point_data() -> dict[str, Any]:
    data = _read_json_file(FACE_POINT_FILE)
    if data is None:
        data = _migrate_face_point_data()
        if data["common"] or any(
            data["templates"][template][section]
            for template in data["templates"]
            for section in BASELINE_SECTIONS
        ):
            save_face_point_data(data)
            print(f"[FacePoint] Migrated legacy baselines -> {FACE_POINT_FILE}")
    return _normalize_face_point_data(data)


def save_face_point_data(data: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_face_point_data(data)
    _write_json_file(FACE_POINT_FILE, normalized)
    return normalized


def get_template_names() -> list[str]:
    return list(load_face_point_data().get("templates", {}).keys())


def get_active_template() -> str:
    return load_face_point_data().get("active_template", DEFAULT_TEMPLATE_NAME)


def set_active_template(template_name: str) -> bool:
    normalized_name = str(template_name).strip()
    if not normalized_name:
        raise ValueError("Template name cannot be empty")
    data = load_face_point_data()
    data["templates"].setdefault(normalized_name, _empty_template_data())
    data["active_template"] = normalized_name
    save_face_point_data(data)
    return True


def save_current_as_template(template_name: str) -> bool:
    normalized_name = str(template_name).strip()
    if not normalized_name:
        raise ValueError("Template name cannot be empty")
    data = load_face_point_data()
    active = data["active_template"]
    source = data["templates"].get(active, _empty_template_data())
    data["templates"][normalized_name] = copy.deepcopy(source)
    data["active_template"] = normalized_name
    save_face_point_data(data)
    return True


def _load_common_baseline() -> Optional[dict[str, Any]]:
    common = load_face_point_data().get("common", {})
    if not common:
        return None
    try:
        return {
            "nose": tuple(common["nose"]),
            "iris_left": tuple(common["iris_left"]),
            "iris_right": tuple(common["iris_right"]),
            "eye_line_y": float(
                common.get(
                    "eye_line_y",
                    (float(common["iris_left"][1]) + float(common["iris_right"][1])) / 2.0,
                )
            ),
        }
    except Exception as exc:
        print(f"[FacePoint] Common baseline invalid: {exc}")
        return None


def _save_common_baseline(common: dict[str, Any]) -> None:
    data = load_face_point_data()
    data["common"] = common
    save_face_point_data(data)


def _load_template_section(section: str, template_name: Optional[str] = None) -> Optional[dict[str, Any]]:
    if template_name is None:
        template_name = get_active_template()
    data = load_face_point_data()
    value = data["templates"].get(template_name, {}).get(section, {})
    return value.copy() if isinstance(value, dict) and value else None


def _save_template_section(section: str, value: dict[str, Any], template_name: Optional[str] = None) -> None:
    if template_name is None:
        template_name = get_active_template()
    data = load_face_point_data()
    data["templates"].setdefault(template_name, _empty_template_data())
    data["templates"][template_name][section] = value
    save_face_point_data(data)


def template_section_exists(template_name: str, section: str) -> bool:
    value = _load_template_section(section, template_name=template_name)
    return bool(value)


def save_baseline(nose_pos: tuple[float, float], left_iris_pos: tuple[float, float],
                  right_iris_pos: tuple[float, float], filepath: Optional[str] = None) -> None:
    eye_line_y = (left_iris_pos[1] + right_iris_pos[1]) / 2.0
    data = {
        "nose": list(nose_pos),
        "iris_left": list(left_iris_pos),
        "iris_right": list(right_iris_pos),
        "eye_line_y": eye_line_y,
    }
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_common_baseline(data)
    print(f"[Baseline] Saved: nose={nose_pos} L={left_iris_pos} R={right_iris_pos} eyeY={eye_line_y:.2f}")


def load_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    if filepath is None:
        return _load_common_baseline()
    try:
        data = _read_json_file(filepath)
        if not data:
            return None
        baseline = {
            "nose": tuple(data["nose"]),
            "iris_left": tuple(data["iris_left"]),
            "iris_right": tuple(data["iris_right"]),
        }
        if "eye_line_y" in data:
            baseline["eye_line_y"] = float(data["eye_line_y"])
        else:
            baseline["eye_line_y"] = (data["iris_left"][1] + data["iris_right"][1]) / 2.0
            print(f"[Baseline] eye_line_y auto-derived: {baseline['eye_line_y']:.2f}")
        return baseline
    except Exception as exc:
        print(f"[Baseline] Load failed: {exc}")
        return None


def save_eyelid_baseline(left_ear: float, right_ear: float, filepath: Optional[str] = None) -> None:
    data = {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
    }
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("eyelid", data)
    print(f"[Eyelid Baseline] Saved: L_EAR={left_ear:.4f} R_EAR={right_ear:.4f}")


def load_eyelid_baseline(filepath: Optional[str] = None) -> Optional[dict[str, float]]:
    try:
        data = _load_template_section("eyelid") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        return {
            "left_ear": float(data["left_ear"]),
            "right_ear": float(data["right_ear"]),
        }
    except Exception as exc:
        print(f"[Eyelid Baseline] Load failed: {exc}")
        return None


def save_eyebrow_baseline(metrics: dict[str, Any], filepath: Optional[str] = None) -> None:
    data = {}
    for key in (
        "left_slope",
        "right_slope",
        "slope_symmetry",
        "left_brow_iris_gap",
        "right_brow_iris_gap",
    ):
        if key in metrics:
            data[key] = float(metrics[key])
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("eyebrow", data)
    print(
        "[Eyebrow Baseline] Saved: "
        f"L_S={metrics['left_slope']:.4f} R_S={metrics['right_slope']:.4f} | "
        f"L_BIG={metrics['left_brow_iris_gap']:.4f} "
        f"R_BIG={metrics['right_brow_iris_gap']:.4f}"
    )


def load_eyebrow_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        data = _load_template_section("eyebrow") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        has_required = all(
            key in data
            for key in (
                "left_brow_iris_gap",
                "right_brow_iris_gap",
                "left_slope",
                "right_slope",
            )
        )
        if not has_required:
            print("[Eyebrow Baseline] Missing BIG/slope fields; ignoring old eyebrow baseline.")
            return None
        return {
            "left_slope": float(data["left_slope"]),
            "right_slope": float(data["right_slope"]),
            "slope_symmetry": float(data.get("slope_symmetry", 0.0)),
            "left_brow_iris_gap": float(data["left_brow_iris_gap"]),
            "right_brow_iris_gap": float(data["right_brow_iris_gap"]),
            "has_brow_iris_gap": True,
        }
    except Exception as exc:
        print(f"[Eyebrow Baseline] Load failed: {exc}")
        return None


def save_mouth_baseline(mar: float, filepath: Optional[str] = None) -> None:
    data = {"mar": float(mar)}
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("mouth", data)
    print(f"[Mouth Baseline] Saved: MAR={mar:.4f}")


def load_mouth_baseline(filepath: Optional[str] = None) -> Optional[dict[str, float]]:
    try:
        data = _load_template_section("mouth") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        return {"mar": float(data["mar"])}
    except Exception as exc:
        print(f"[Mouth Baseline] Load failed: {exc}")
        return None


def save_lower_lip_baseline(llr: float, filepath: Optional[str] = None,
                            side_lower_tip: Optional[tuple[int, int]] = None,
                            side_roi: Optional[list[int]] = None) -> None:
    data = {"llr": float(llr)}
    if side_lower_tip is not None:
        data["side_lower_tip"] = [int(side_lower_tip[0]), int(side_lower_tip[1])]
    if side_roi is not None:
        data["side_roi"] = [int(value) for value in side_roi]
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("lower_lip", data)
    side_msg = f", side_lower_tip={data['side_lower_tip']}" if "side_lower_tip" in data else ""
    print(f"[LowerLip Baseline] Saved: LLR={llr:.4f}{side_msg}")


def save_lower_lip_side_roi(side_roi: list[int], filepath: Optional[str] = None,
                            side_lower_tip: Optional[tuple[int, int]] = None) -> None:
    data: dict[str, Any] = {}
    current = _load_template_section("lower_lip") if filepath is None else _read_json_file(filepath)
    if isinstance(current, dict):
        data.update(current)

    data["side_roi"] = [int(value) for value in side_roi]
    if side_lower_tip is not None:
        data["side_lower_tip"] = [int(side_lower_tip[0]), int(side_lower_tip[1])]

    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("lower_lip", data)
    print(f"[LowerLip Baseline] Saved side ROI: {data['side_roi']}")


def load_lower_lip_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        data = _load_template_section("lower_lip") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        out: dict[str, Any] = {}
        if "llr" in data:
            out["llr"] = float(data["llr"])
        if "side_lower_tip" in data:
            out["side_lower_tip"] = list(data["side_lower_tip"])
        if "side_roi" in data:
            out["side_roi"] = list(data["side_roi"])
        if not out:
            return None
        return out
    except Exception as exc:
        print(f"[LowerLip Baseline] Load failed: {exc}")
        return None


def save_upper_lip_baseline(ulr: float, filepath: Optional[str] = None,
                            side_upper_tip: Optional[tuple[int, int]] = None,
                            side_roi: Optional[list[int]] = None) -> None:
    data = {"ulr": float(ulr)}
    if side_upper_tip is not None:
        data["side_upper_tip"] = [int(side_upper_tip[0]), int(side_upper_tip[1])]
    if side_roi is not None:
        data["side_roi"] = [int(value) for value in side_roi]
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("upper_lip", data)
    side_msg = f", side_upper_tip={data['side_upper_tip']}" if "side_upper_tip" in data else ""
    print(f"[UpperLip Baseline] Saved: ULR={ulr:.4f}{side_msg}")


def save_upper_lip_side_roi(side_roi: list[int], filepath: Optional[str] = None,
                            side_upper_tip: Optional[tuple[int, int]] = None) -> None:
    data: dict[str, Any] = {}
    current = _load_template_section("upper_lip") if filepath is None else _read_json_file(filepath)
    if isinstance(current, dict):
        data.update(current)

    data["side_roi"] = [int(value) for value in side_roi]
    if side_upper_tip is not None:
        data["side_upper_tip"] = [int(side_upper_tip[0]), int(side_upper_tip[1])]

    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("upper_lip", data)
    print(f"[UpperLip Baseline] Saved side ROI: {data['side_roi']}")


def load_upper_lip_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        data = _load_template_section("upper_lip") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        out: dict[str, Any] = {}
        if "ulr" in data:
            out["ulr"] = float(data["ulr"])
        if "side_upper_tip" in data:
            out["side_upper_tip"] = list(data["side_upper_tip"])
        if "side_roi" in data:
            out["side_roi"] = list(data["side_roi"])
        if not out:
            return None
        return out
    except Exception as exc:
        print(f"[UpperLip Baseline] Load failed: {exc}")
        return None


def save_mouth_corners_baseline(nose_pos: tuple[float, float], left_corner_pos: tuple[float, float],
                                right_corner_pos: tuple[float, float],
                                filepath: Optional[str] = None) -> None:
    data = {
        "nose": list(nose_pos),
        "corner_left": list(left_corner_pos),
        "corner_right": list(right_corner_pos),
    }
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("mouth_corners", data)
    print(f"[MouthCorners Baseline] Saved: nose={nose_pos} L_corner={left_corner_pos} R_corner={right_corner_pos}")


def load_mouth_corners_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        data = _load_template_section("mouth_corners") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        return {
            "nose": tuple(data["nose"]),
            "corner_left": tuple(data["corner_left"]),
            "corner_right": tuple(data["corner_right"]),
        }
    except Exception as exc:
        print(f"[MouthCorners Baseline] Load failed: {exc}")
        return None


def save_head_position_baseline(nose_px: tuple[float, float], eye_left_px: tuple[float, float],
                                eye_right_px: tuple[float, float], frame_width: int,
                                frame_height: int, filepath: Optional[str] = None) -> None:
    eye_dist = np.linalg.norm(np.array(eye_right_px, dtype=np.float64) - np.array(eye_left_px, dtype=np.float64))
    data = {
        "nose_px": list(nose_px),
        "eye_left": list(eye_left_px),
        "eye_right": list(eye_right_px),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "eye_distance": round(float(eye_dist), 1),
    }
    if filepath is not None:
        _write_json_file(filepath, data)
    else:
        _save_template_section("head_position", data)
    print(
        f"[HeadPosition Baseline] Saved: nose={nose_px} L_eye={eye_left_px} "
        f"R_eye={eye_right_px} eye_dist={eye_dist:.1f}"
    )


def load_head_position_baseline(filepath: Optional[str] = None) -> Optional[dict[str, Any]]:
    try:
        data = _load_template_section("head_position") if filepath is None else _read_json_file(filepath)
        if not data:
            return None
        return {
            "nose_px": tuple(data["nose_px"]),
            "eye_left": tuple(data["eye_left"]),
            "eye_right": tuple(data["eye_right"]),
            "frame_width": data.get("frame_width", DEFAULT_FRAME_WIDTH),
            "frame_height": data.get("frame_height", DEFAULT_FRAME_HEIGHT),
            "eye_distance": data.get("eye_distance", 0.0),
        }
    except Exception as exc:
        print(f"[HeadPosition Baseline] Load failed: {exc}")
        return None
