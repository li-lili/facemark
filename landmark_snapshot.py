"""One-shot exports of published detection results; no camera/model dependencies."""

import copy
import json
import threading
import time
from datetime import datetime
from pathlib import Path


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def mediapipe_points(face, smoothed, width, height):
    """Keep model coordinates separately from the EMA coordinates used by key 3."""
    points = []
    for index, lm in enumerate(face):
        raw = {name: float(getattr(lm, name)) for name in ("x", "y", "z")}
        point = {
            "id": index,
            "raw_normalized": raw,
            "raw_pixel": {"x": raw["x"] * width, "y": raw["y"] * height},
            "smoothed_pixel": {"x": float(smoothed[index][0]),
                               "y": float(smoothed[index][1])},
        }
        for name in ("visibility", "presence"):
            value = getattr(lm, name, None)
            if value is not None:
                point[name] = float(value)
        points.append(point)
    return points


class LandmarkSnapshots:
    # Export freshness only; does not change detection or calibration tolerances.
    MAX_AGE_SECONDS = 2.0

    def __init__(self, directory=None, logger=print):
        self.directory = Path(directory or Path(__file__).parent / "landmark_snapshots")
        self.logger = logger
        self._lock = threading.Lock()
        self._sources = {}
        self._writer = None

    def publish(self, source, data):
        # Producers publish only a completed frame. Never retain mutable model arrays.
        snapshot = copy.deepcopy(data)
        with self._lock:
            self._sources[source] = (time.monotonic(), snapshot)

    def invalidate(self, source, reason):
        self.publish(source, {"status": reason, "point_count": 0})

    def request_save(self, logger=None):
        logger = logger if logger is not None else self.logger
        requested_at = timestamp()
        with self._lock:
            if self._writer is not None and self._writer.is_alive():
                logger("[Landmarks][ERROR] 上一次保存尚未完成，请稍后再按 4。 路径=未生成 front=not_sampled side=not_sampled")
                return False
            sources = {}
            now = time.monotonic()
            for name in ("front", "side"):
                item = self._sources.get(name)
                if item is None:
                    sources[name] = {"status": "not_started_or_unavailable", "point_count": 0}
                    continue
                published, data = item
                age = now - published
                if age > self.MAX_AGE_SECONDS:
                    sources[name] = {"status": "stale", "point_count": 0,
                                     "age_seconds": age}
                else:
                    sources[name] = copy.deepcopy(data)
                    sources[name]["age_seconds"] = age
            count = sum(s.get("point_count", 0) for s in sources.values())
            if not count:
                logger(f"[Landmarks][ERROR] 路径=未生成 无有效检测结果，未保存：{sources}")
                return False
            payload = {"schema_version": 1, "saved_at": requested_at,
                       "capture_mode": "latest_available_not_synchronized",
                       "point_count": count, "sources": sources}
            self._writer = threading.Thread(target=self._write, args=(payload, logger),
                                            name="landmark-json-writer", daemon=False)
            self._writer.start()
        return True

    def _write(self, payload, logger=None):
        logger = logger if logger is not None else self.logger
        counts = ", ".join(f"{name}={data.get('point_count', 0)} ({data['status']})"
                           for name, data in payload["sources"].items())
        path = None
        created = False
        try:
            # Reject NaN/Infinity before creating any file.
            text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda value: value.tolist())
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromisoformat(payload["saved_at"]).strftime("%Y-%m-%d_%H-%M-%S_%f")
            base = f"face_landmarks_{stamp}"
            suffix = 0
            while True:
                path = self.directory / f"{base}{'_' + str(suffix) if suffix else ''}.json"
                try:
                    stream = path.open("x", encoding="utf-8")
                    created = True
                    break
                except FileExistsError:
                    suffix += 1
            with stream:
                stream.write(text)
            logger(f"[Landmarks][OK] 路径={path.resolve()} 时间={payload['saved_at']} "
                        f"特征点数量={payload['point_count']} [{counts}]")
        except Exception as exc:
            if created and path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass
            logger(f"[Landmarks][ERROR] 保存失败：{exc} 路径={path or '未生成'} "
                   f"时间={payload['saved_at']} [{counts}]")
