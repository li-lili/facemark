"""
摄像头测试工具 — 快速验证摄像头是否正常
=========================================
支持多摄像头切换（按 1/2/3... 切换）
按 Q 退出

用法: python camera_test.py             # 扫描并打开第一个可用摄像头
      python camera_test.py 1            # 打开摄像头1
      python camera_test.py 0 1          # 同时打开两个摄像头, 按键切换
"""

import sys
import cv2
import numpy as np


def scan_cameras(max_idx=10):
    """扫描可用摄像头，返回可用索引列表"""
    available = []
    print("=" * 40)
    print("  扫描可用摄像头...")
    print("=" * 40)
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append(i)
            print(f"  Camera [{i}]  OK  {w}x{h}")
            cap.release()
        else:
            print(f"  Camera [{i}]  -")
    print(f"\n  可用: {available}")
    print("=" * 40)
    return available


def get_exp_status(cap):
    """获取曝光/增益等状态"""
    ae = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    ae_mode = "自动" if ae == 3.0 or ae == 1.0 else "手动"
    return ae_mode, exp, gain, brightness, contrast


def set_exp(cap, val):
    """设置手动曝光值"""
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动
    cap.set(cv2.CAP_PROP_EXPOSURE, val)
    print(f"  Exposure set to {val:.0f}")


def set_auto_exp(cap, enable=True):
    """切换自动/手动曝光"""
    val = 3.0 if enable else 0.25
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
    print(f"  Auto exposure: {'ON' if enable else 'OFF'}")


def main():
    # 先扫描所有摄像头
    scan_cameras()

    # 解析参数：指定要打开的摄像头索引
    indices = [0]  # 默认
    for arg in sys.argv[1:]:
        try:
            indices.append(int(arg))
        except ValueError:
            pass
    indices = sorted(set(indices))

    caps = {}
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            # 设置 1920x1080
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            caps[idx] = cap
            res_note = " ✓ 1920x1080" if (w, h) == (1920, 1080) else f" (实际: {w}x{h})"
            print(f"  [Camera {idx}] OK  {w}x{h}  FPS={fps:.1f}{res_note}")

            # 读取初始曝光
            ae, exp, gain, bri, con = get_exp_status(cap)
            print(f"       Exposure: {'AUTO' if ae == '自动' else 'MANUAL'}  exp={exp:.0f}  gain={gain:.0f}")
        else:
            print(f"  [Camera {idx}] FAILED to open")

    if not caps:
        print("ERROR: 没有可用的摄像头")
        sys.exit(1)

    current_idx = min(caps.keys())
    print(f"\n当前: Camera {current_idx}")
    print("按键: [Q]退出  [1/2/3...]切换  [S]截图  [F]全屏  [R]分辨率")
    print("      [A]自动曝光开关  [+/-]曝光  [[/]]增益")

    win_name = "Camera Test"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        cap = caps[current_idx]
        ok, frame = cap.read()
        if not ok:
            print(f"Camera {current_idx} 读取失败")
            break

        h, w = frame.shape[:2]
        ae_mode, exp, gain, bri, con = get_exp_status(cap)

        # 叠加信息
        info_lines = [
            f"Camera {current_idx}  {w}x{h}",
            f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}",
            f"Exp: {ae_mode} ({exp:.0f})  Gain: {gain:.0f}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, 30 + i * 24),
                        0, 0.5, (0, 255, 255), 1)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"capture_cam{current_idx}_{ts}.png"
            cv2.imwrite(fname, frame)
            print(f"  Saved: {fname}")
        elif key == ord('f'):
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        elif key == ord('r'):
            print(f"\n--- Camera {current_idx} 分辨率支持 ---")
            test_res = [
                (1920, 1080), (1280, 720), (960, 540),
                (848, 480), (800, 600), (640, 480),
                (640, 360), (480, 640), (480, 360),
                (432, 368), (352, 288), (320, 240),
            ]
            for tw, th in test_res:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, tw)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, th)
                rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if rw == tw and rh == th:
                    print(f"  {tw}x{th}  ✓")
                else:
                    print(f"  {tw}x{th}  -  (实际: {rw}x{rh})")
            print()
        elif key == ord('a'):
            # 切换自动/手动曝光
            ae, _, _, _, _ = get_exp_status(cap)
            set_auto_exp(cap, ae != "自动")
        elif key == ord('+') or key == ord('='):
            # 增加曝光（更亮）
            set_auto_exp(cap, False)
            _, exp, _, _, _ = get_exp_status(cap)
            set_exp(cap, exp + 1)
        elif key == ord('-') or key == ord('_'):
            # 减少曝光（更暗）
            set_auto_exp(cap, False)
            _, exp, _, _, _ = get_exp_status(cap)
            set_exp(cap, exp - 1)
        elif key == ord(']'):
            # 增加增益
            g = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, g + 5)
            print(f"  Gain set to {g+5:.0f}")
        elif key == ord('['):
            # 减少增益
            g = cap.get(cv2.CAP_PROP_GAIN)
            cap.set(cv2.CAP_PROP_GAIN, g - 5)
            print(f"  Gain set to {g-5:.0f}")
        elif ord('0') <= key <= ord('9'):
            idx = key - ord('0')
            if idx in caps:
                current_idx = idx
                print(f"  Switched to Camera {current_idx}")
            elif idx in indices:
                current_idx = idx
                print(f"  Switched to Camera {current_idx}")

    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
