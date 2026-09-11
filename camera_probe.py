"""Probe local cameras and preview the ones that can return frames.

Usage:
    python camera_probe.py
    python camera_probe.py --max-index 15
    python camera_probe.py --backend msmf
    python camera_probe.py --no-preview
"""

import argparse
import time

import cv2


BACKENDS = {
    "auto": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


def open_camera(index: int, backend: int):
    if backend:
        return cv2.VideoCapture(index, backend)
    return cv2.VideoCapture(index)


def probe_camera(index: int, backend: int, warmup_frames: int):
    cap = open_camera(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    ok = False
    frame = None
    for _ in range(max(1, warmup_frames)):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.03)

    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    return {
        "index": index,
        "cap": cap,
        "frame": frame,
        "width": int(width),
        "height": int(height),
        "fps": float(fps) if fps else 0.0,
    }


def preview_cameras(cameras):
    if not cameras:
        return

    print("\nPreview controls: q/ESC=quit, n=next camera")
    current = 0
    while 0 <= current < len(cameras):
        cam = cameras[current]
        cap = cam["cap"]
        win = f"Camera {cam['index']} ({current + 1}/{len(cameras)})"
        print(f"Previewing camera {cam['index']} - press n for next, q to quit")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = cam["frame"]

            label = f"Camera {cam['index']}  {cam['width']}x{cam['height']}  q=quit n=next"
            cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow(win, frame)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyWindow(win)
                return
            if key == ord("n"):
                cv2.destroyWindow(win)
                current += 1
                break


def main():
    parser = argparse.ArgumentParser(description="Probe available camera indexes.")
    parser.add_argument("--max-index", type=int, default=9,
                        help="highest camera index to test, inclusive")
    parser.add_argument("--backend", choices=BACKENDS.keys(), default="dshow",
                        help="OpenCV backend to use")
    parser.add_argument("--warmup-frames", type=int, default=8,
                        help="frames to try before declaring a camera unavailable")
    parser.add_argument("--no-preview", action="store_true",
                        help="only print results, do not open preview windows")
    args = parser.parse_args()

    backend = BACKENDS[args.backend]
    cameras = []
    print(f"Testing camera indexes 0..{args.max_index} with backend={args.backend}")

    for index in range(args.max_index + 1):
        cam = probe_camera(index, backend, args.warmup_frames)
        if cam is None:
            print(f"[--] camera {index}: unavailable")
            continue
        cameras.append(cam)
        fps_text = f"{cam['fps']:.1f}" if cam["fps"] > 0 else "unknown"
        print(f"[OK] camera {index}: {cam['width']}x{cam['height']} fps={fps_text}")

    indexes = [cam["index"] for cam in cameras]
    print(f"\nAvailable cameras: {indexes}")
    print(f"Count: {len(cameras)}")

    try:
        if not args.no_preview:
            preview_cameras(cameras)
    finally:
        for cam in cameras:
            cam["cap"].release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
