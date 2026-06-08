import cv2

def check_camera_supported_resolutions(cap_index=0):
    cap = cv2.VideoCapture(cap_index)
    
    # 常用分辨率列表（你可以自己加）
    resolutions = [
        (1920, 1080),
        (1280, 720),
        (1024, 768),
        (960, 540),
        (800, 600),
        (640, 480),
        (320, 240),
    ]

    print("摄像头支持的分辨率 & 帧率：\n")
    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        real_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if int(real_w) == w and int(real_h) == h:
            print(f"✅ 支持 {w}x{h}  |  FPS: {fps:.1f}")
        else:
            print(f"❌ 不支持 {w}x{h}")

    cap.release()

if __name__ == "__main__":
    check_camera_supported_resolutions(0)