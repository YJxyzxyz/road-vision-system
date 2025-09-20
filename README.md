# Road Vision System

#### author:YJxyzxyz

<img src="https://github.com/YJxyzxyz/road-vision-system/blob/master/logo.png" width="350px">

------

# 总体模块路线图

1. **Module 1：视频采集与时间戳**（输入源、分辨率、帧率、精准时间）
2. **Module 2：图像预处理插件框架**（占位：去雾/去雨可独立开关与串接）
3. **Module 3：车辆检测接口**（统一返回格式，后端可替换：YOLO/TensorRT）
4. **Module 4：轻量跟踪（ID维持）**
5. **Module 5：单目测距（相机标定→几何测距）**
6. **Module 6：速度估计（帧时间差 + 平滑）**
7. **Module 7：可视化与告警**
8. **Module 8：性能与异步流水线（多线程/跳帧/配置热切换）**
9. **Module 9：Jetson Nano 部署替换层（相机/GStreamer，TensorRT推理）**

------

# Module 1（视频采集与时间戳）

## ✅ 目标

- 从摄像头/视频文件**稳定拿帧**
- 每帧附带**捕获时间戳**（秒，float），不要用“估计帧率”
- 可配置分辨率与帧率；统计**实际FPS**（滑动平均）

## 📁 目录&文件（最小骨架）

```
roadvision/
  configs/
    default.yaml
  src/
    __init__.py
    config.py
    io_video/
      __init__.py
      capture.py
      fps_meter.py
  main_preview.py
```

------

# Module 2.1（图像预处理插件框架）

## ✅ 目标

- 传统方法实现
- 在配置里开/关与调整参数，默认零侵入（不开也不影响主流程）
- 后面换成更强的去雾/去雨 CNN 也只需替换同名插件即可。
- 同时支持cpu和gpu两种方式

## 📁 目录&文件（最小骨架）

修改调参说明

**默认开链**（`preprocess.enabled: true`），画面对比度应更清晰（CLAHE），细雨噪点略弱（Median）。

关链：把 `enabled: false` 对比性能与视觉差异。

改 `CLAHEDehaze.params`：

- `space: "LAB"` 往往偏“自然”，`"YCrCb"`偏“锐”；
- `clip_limit` 
- `tile_grid` 越大块对比度自适应越细腻，但开销略增。

改 `MedianDerain.params.ksize` 为 5 看“去纹理”的力度（过大会糊）。

------

# Module 3（车辆检测接口）

## ✅ 目标

- 统一的检测输出 `Detection` 数据结构，后端可替换（Ultralytics/ONNX/TensorRT）。
- 在配置中控制模型路径、类别筛选、阈值。
- 结果直接传递给后续的跟踪、测距、可视化模块。

## ⚙️ 使用方法

- 默认 `configs/default.yaml` 中开启 YOLOv8n 检测：

  ```yaml
  detect:
    enabled: true
    model: "yolov8n.pt"
    classes_keep: [0, 2, 3, 5, 7]
  ```

- 若需要切换模型或设备，修改 `model`、`device`、`conf_thres` 等字段即可。

------

# Module 4 / 5 / 6（跟踪 + 测距 + 速度估计）

## ✅ 新增能力

- 内置 SORT 跟踪器，自动维持目标 ID，支持 Kalman 预测与 IoU 匹配。
- `Detection` 结构新增 `track_id`、`distance_m`、`speed_kmh` 字段，串联测距与速度估计。
- 在真实时间戳驱动下，根据地面坐标轨迹窗口计算目标瞬时速度。

## ⚙️ 快速上手

1. **启用跟踪器**（默认已开）：

   ```yaml
   tracking:
     enabled: true
     backend: "sort"
     max_staleness: 1.2
     min_hits: 3
     iou_threshold: 0.35
     speed_window: 0.8
   ```

2. **配置地面投影**（若需距离/速度）：

   ```yaml
   geometry:
     enabled: true
     projector:
       type: "homography"
       image_points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
       world_points: [[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]]
       origin: [0.0, 0.0]
       max_distance: 1000.0
   ```

   - `image_points` 为图像像素（通常取路面四角/车道线交点）。
   - `world_points` 对应实际地面坐标（单位：米），需与像素点一一对应。
   - `origin` 定义距离的起点（例如相机投影在地面的坐标）。

3. **结果查看**：主循环中会为每个目标叠加 `ID | 类别 | 置信度` 以及底部的 `距离 / 速度` 信息。

> 📌 若暂未完成标定，可仅开启跟踪器（`geometry.enabled: false`），系统会显示 ID 且保持距离/速度为 `None`。

------

# Module 7（可视化与告警）

## ✅ 当前实现

- `src/vis/draw.py` 提供统一的绘制函数，按类别自动配色，叠加跟踪 ID、距离、速度。
- `preview.compare` 支持横/纵向对比原始帧与处理帧，可在画面上方/下方显示 FPS。
- 录像输出（`preview.record`）可选，将对比画面导出到 MP4 文件。

## ⚙️ 配置入口

```yaml
vis:
  draw:
    det: true          # 控制是否绘制检测框
    thickness: 2
    font_scale: 0.6
```

------

# Module 8（性能与异步流水线）

## ✅ 当前实现

- `src/runtime/async_pipeline.AsyncPipeline` 提供**采集线程 + 推理线程**的异步组合，避免摄像头读取阻塞推理。
- 帧队列支持 `queue_size`、`drop_policy`、`max_latency_ms`，在性能不足时自动丢弃旧帧或跳帧保持实时性。
- 主循环读取 `AsyncPipeline` 输出，可在退出时看到“丢弃/跳过帧”统计，便于评估吞吐表现。

## 🔄 热更新

- 新增 `runtime.hot_reload` 轮询器，默认每 2 秒检测配置文件改动。触发后自动重建管线（相机、推理、渲染）。
- 启动命令支持 `--config` 指定自定义配置文件，例如：

  ```bash
  python main_preview.py --config configs/jetson.yaml
  ```

- 修改 `configs/default.yaml` 后（如切换模型阈值），无需重启窗口即可生效。

## ⚙️ 配置片段

```yaml
runtime:
  async:
    enabled: true          # 打开多线程采集/推理
    queue_size: 6          # 缓存的帧数
    drop_policy: "oldest"   # 队列满时丢弃最旧帧
    max_latency_ms: 200.0  # 帧延迟超过 200ms 视为过期
    result_timeout_ms: 300.0
  hot_reload:
    enabled: true
    interval_sec: 2.0
```

> ℹ️ 若想保留“严格逐帧”模式，可保持 `runtime.async.enabled: false`，系统仍按单线程顺序执行。

------

# Module 9（Jetson Nano 部署层）

## ✅ 当前实现

- `camera.backend: "gstreamer"` 自动拼接 `nvarguscamerasrc` 管线，或直接填写自定义 GStreamer 字符串，实现 Jetson CSI 摄像头零拷贝采集。
- 新增 TensorRT 检测后端（`detect.backend: "tensorrt"`），复用 Ultralytics 导出的 `.engine`，支持 `imgsz`、`warmup_runs`、`half` 等参数。
- 同一份配置可在 PC 与 Jetson 间切换，仅需调整 `runtime.async` 和 TensorRT 引擎路径。

## ⚙️ Jetson 摄像头示例

```yaml
camera:
  backend: "gstreamer"
  gstreamer:
    pipeline: ""  # 为空时自动生成 nvarguscamerasrc 管线
    sensor_id: 0
    capture_width: 1920
    capture_height: 1080
    display_width: 1280
    display_height: 720
    framerate: 30
    flip_method: 2
```

## ⚙️ TensorRT 检测示例

```yaml
detect:
  enabled: true
  backend: "tensorrt"
  engine: "weights/yolov8n.engine"
  imgsz: 640
  half: true
  warmup_runs: 3
  classes_keep: [2, 3, 5, 7]
```

- 使用 Ultralytics 导出引擎：`yolo export model=yolov8n.pt format=engine half=True device=0`。
- Jetson Nano 若只安装 TensorRT runtime，可在 `requirements.txt` 基础上额外安装 `ultralytics` 轻量包，或复制训练好的 `.engine`。

------

# 运行示例

```bash
python main_preview.py  # 读取默认配置

# 或者指定自定义配置，保持热更新
python main_preview.py --config configs/default.yaml
```

- 默认读取 `configs/default.yaml`，如启用 `runtime.hot_reload.enabled: true`，编辑文件后会自动重启管线。
- 保证已安装依赖：`pip install -r requirements.txt`（包含 `filterpy`、`ultralytics` 等）。
- 运行后窗口显示 RAW/PROC 对比，按 `q` 或 `Esc` 退出。

------

# 标定小贴士

1. 使用静态帧（截屏或单帧保存）标注地面上已知距离的点，例如停车线、车道线交点。
2. 在 `image_points` 中填入像素坐标（可用 OpenCV `imshow` + 鼠标获取或其他工具），在 `world_points` 填对应实际距离。
3. 建议起点 `origin` 设为摄像机投影点或路口附近的固定参考点，便于读数。
4. `max_distance` 可防止远处噪点造成极端速度值，可根据场景调整。

------

# TODO / 下一步计划

- Module 10：引入在线再训练/模型蒸馏接口，结合道路场景自适应。
- Module 11：融合多摄像头/多路 RTSP 管线的调度与同步告警。

