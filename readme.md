# FaceMark — 眼睛舵机自动调整系统

基于 **MediaPipe + EllSeg 虹膜检测** 的人脸仿真机器人眼部舵机自动标定工具。通过摄像头实时检测虹膜位置，自动调整舵机（A8-A13）使眼球居中，替代人工手动调参。

---

## 整体架构

```
┌─────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  摄像头/USB  │───▶│ EllSeg 虹膜偏移检测   │───▶│  舵机调整策略    │
│  (1920x1080) │    │ (MediaPipe + EllSeg) │    │ (坐标下降/手动)  │
└─────────────┘    └──────────────────────┘    └────────┬────────┘
                                                        │ 串口 UART
                                                        ▼
                                                 ┌─────────────────┐
                                                 │  舵机 A8-A13     │
                                                 │  (眼球+眼皮)     │
                                                 └─────────────────┘
```

**工作流程：**
1. 摄像头采集人脸 → MediaPipe 提取 478 点人脸网格 → 裁剪眼部区域
2. EllSeg 在眼部区域分割虹膜/瞳孔椭圆 → 与基线比较得到像素偏移
3. 将偏移转换为舵机调整方向 → 每次移动 ±1° → 等待稳定 → 循环直到偏移 ≤ 2px

---

## 环境准备

### 依赖
- Python 3.11+
- **摄像头**（建议支持 1920×1080 @ 60fps+ 的 USB 摄像头）
- **串口舵机控制器**（默认 COM5，115200 bps）

### 安装

```bash
pip install torch opencv-python numpy mediapipe pyyaml pyserial PySide6
```

EllSeg 模型和 MediaPipe 模型会在**首次运行时自动下载**，位于:
- `models/face_landmarker.task` — MediaPipe 人脸关键点模型
- `EllSeg/weights/all.git_ok` — EllSeg 虹膜分割权重

---

## 快速开始

### 1. 建立虹膜基线（必须）

先让脸保持**中性表情、眼球注视正前方**，运行基线标定：

```bash
python ellseg_scorer.py
```

- 画面稳定后，按 **`[S]`** 保存当前鼻子 + 双眼虹膜中心为基线
- 成功后会生成 `face_points.json`
- 按 **`[Q]`** 退出

> 也可用 `ellseg_demo.py`（轻量版）完成同样操作。

### 2. 自动调整眼球舵机

确认 `face_points.json` 存在后：

```bash
python eye_auto_tuner.py
```

**核心参数（可命令行覆盖）：**

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--port` | COM5 | 串口端口 |
| `--baud` | 115200 | 波特率 |
| `--ratio` | 2.0 | 像素→角度换算系数 (°/px) |
| `--max-iter` | 50 | 最大迭代次数 |
| `--wait` | 1.0 | 每次移动后等待秒数 |

**调整流程（每轮迭代）：**
1. 打开 MediaPipe + EllSeg 检测
2. 采集 60 帧 → 去掉最高、最低各 10 帧 → 对中间 40 帧求平均偏移
3. 关闭检测（释放 GPU）
4. 若双眼平均偏移均 ≤ 2px → **通过**，保存结果
5. 否则按偏移方向移动眼球舵机 ±1° → 等待 1s → 进入下一轮

**按键：** `[Q]` / `[Esc]` 随时停止

### 3. 输出文件

| 文件 | 说明 |
|------|------|
| `face_points.json` | 虹膜基线（鼻子 + 左右虹膜中心坐标） |
| `best_eye_config.json` | 最优舵机角度结果 |
| `eye_angles_best.json` | 导出的角度配置（可直接用于舵机设置） |

---

## 舵机通道定义

| 通道 | 名称 | 功能 |
|------|------|------|
| A8 | 左眼皮 | 上下开合 |
| A9 | 右眼皮 | 上下开合 |
| A10 | 左眼球垂直 | 上下转动 |
| A11 | 右眼球垂直 | 上下转动（物理反向） |
| A12 | 左眼球水平 | 内外转动 |
| A13 | 右眼球水平 | 内外转动（物理反向） |

`auto_adjust()` **仅调整 A10-A13（眼球）**，眼皮单独手动调整。

---

## 文件结构

```
facemark/
├── eye_auto_tuner.py          # ★ 主入口：初始化硬件 + 引导式自动调整
├── ellseg_scorer.py           # EllSeg 虹膜检测器（摄像头 + 模型 + 基线偏移）
├── eye_scorer.py              # 纯几何眼睛评分（开放度 + 虹膜居中度）
├── eye_strategies.py          # 优化策略（坐标下降法）
├── eye_constants.py           # 常量 + TuningResult 数据类
├── Motor.py                   # 舵机控制器 FaceController
├── Communication.py           # UART 串口通信
├── utility.py                 # YAML 读写 + 校验 + 角度转换
├── expression_match.py        # 表情匹配度检测 + FaceEMA
│
├── manual_eye_calibrator.py   # 手动标定工具（可视化眼部关键点 + 评分）
├── eye_keypoint_monitor.py    # 眼部关键点抖动监测 + 模板记录
├── ellseg_demo.py             # EllSeg 独立 Demo（标定基线）
├── res.py                     # 摄像头分辨率检测工具
├── test_servo_demo.py         # 舵机测试 Demo
│
├── 29_servo_config(13).yaml   # 舵机配置文件（角度范围 + 初始位置）
├── eyelid_tuner_config.json   # 眼皮调优参数配置
│
├── models/                    # MediaPipe 模型
├── EllSeg/                    # EllSeg 虹膜分割模型
├── UI/                        # Qt GUI 界面
├── build.bat                  # PyInstaller 打包脚本
└── readme.md
```


## 已知问题 & 注意事项

1. **首次移动幅度大**：初始化时不会将舵机归位到 YAML 中的 `temp_deg`。如果舵机在上次关机时处于极端位置，首次发送命令会产生较大位移。可在 `initialize()` 中增加舵机归位步骤。

2. **检测功耗**：EllSeg 每次推理会占用 GPU，设计上每轮采集 60 帧后关检测再移舵机，避免 GPU 一直跑。

3. **光照敏感**：EllSeg 对眼部光照质量敏感，确保环境光线均匀，避免强烈阴影。

4. **摄像头分辨率**：建议使用 1920×1080 以获得足够的虹膜像素细节，低分辨率会导致检测精度下降。
