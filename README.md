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

## ✅ 目标

- **输入/推理/显示分离**：通过生产者-消费者队列把取帧、推理和渲染拆分到独立线程，降低相互阻塞的概率。
- **智能跳帧与限流**：在检测/跟踪跟不上时丢弃过期帧或降低处理频率，确保 UI 与录像依旧流畅。
- **热切换配置**：在不中断主循环的情况下更新阈值、模型或可视化参数，实现运行期的“软重启”。
- **性能监控**：采集各阶段耗时与队列长度，输出到日志/屏幕，便于快速定位瓶颈。

## 🧱 推荐架构

1. **CaptureThread**
   - 独立线程调用 `VideoSource.read()`，将 `(Frame, ts)` 放入 `queue.Queue(maxsize=N)`。
   - 如果队列已满且允许跳帧，优先丢弃最旧帧（`queue.get_nowait()` + `task_done()`）。
   - 提供 `stop_event`，在退出时优雅释放相机。

2. **InferWorker**
   - 另一个线程从输入队列取帧，完成预处理 + 检测 + 跟踪。
   - 结果 `Detection` 列表放入输出队列，必要时在此阶段完成几何投影。
   - 记录单帧耗时（preprocess/detect/track）并写入 `FPSMeter` 或自定义 profiler。

3. **Main/UI Thread**
   - 负责从输出队列取数据，调用 `draw_detections` 叠加结果并刷新窗口/录像。
   - 若开启录像，在此线程统一写入，避免多线程争抢编码器。

4. **配置热更新**
   - 监控 `configs/default.yaml` 的 `mtime` 或提供 REST/WebSocket 入口。
   - 一旦检测到变化，通过线程安全的 `ConfigStore` 更新阈值、告警规则。

## ⚙️ 配置建议

在 `configs/default.yaml` 新增 `runtime.async` 区块，示例：

```yaml
runtime:
  async:
    enabled: true
    capture:
      queue_size: 4
      drop_oldest: true
    infer:
      workers: 1          # Jetson 上通常维持 1，x86 可视情况增加
      profile_interval: 60  # 每 60 帧输出一次性能统计
    output:
      queue_size: 2
    hot_reload:
      enabled: true
      watch_interval: 2.0   # 秒
```

结合 `FPSMeter` 输出，可在日志中形成如下数据：

```
[Perf] capture=4.2ms preprocess=6.8ms detect=21.5ms track=3.1ms queue_in=1/4 queue_out=0/2
```

## 🧪 验证清单

- 压力测试：从 1080p @ 30 FPS 视频流读取，确认在 CPU/GPU 不同组合下不会积压超过 `queue_size`。
- 跳帧策略：刻意让检测线程睡眠，观察主窗口 FPS 是否仍稳定在可接受范围。
- 热更新：修改 `detect.conf_thres` 或 `vis.draw.det`，确保无需重启即可生效。

------

# Module 9（Jetson Nano 部署与 TensorRT）

## ✅ 目标

- 在 **Jetson Nano/Orin** 等 JetPack 平台上完成端到端部署，接入 CSI 或 USB 摄像头。
- 使用 **TensorRT** 替换默认的 PyTorch/ONNX 推理后端，满足实时性能。
- 将视频采集切换到 **GStreamer**，充分利用硬件 ISP 与 NVMM 零拷贝。
- 提供一套部署脚本（安装依赖、模型转换、性能调优）与故障排查指引。

## 🚀 部署步骤

1. **环境准备**
   - 依赖 JetPack ≥ 4.6，执行 `sudo apt update && sudo apt install python3-venv libopencv-dev gstreamer1.0-tools`。
   - 开启性能模式：`sudo nvpmodel -m 0 && sudo jetson_clocks`。

2. **模型转换**
   - 利用 Ultralytics 导出 ONNX：`yolo export model=yolov8n.pt format=onnx dynamic=True`。
   - 使用 `trtexec` 生成 engine：
     ```bash
     /usr/src/tensorrt/bin/trtexec \
       --onnx=yolov8n.onnx \
       --saveEngine=yolov8n.engine \
       --workspace=2048 --fp16
     ```
   - 在 `configs/default.yaml` 中设置：
     ```yaml
     detect:
       backend: "tensorrt"
       engine: "models/yolov8n.engine"
       device: "cuda:0"
     ```

3. **相机接入（GStreamer）**
   - CSI 摄像头：
     ```yaml
     camera:
       backend: "gstreamer"
       source: "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! appsink"
     ```
   - USB 摄像头：可使用 `v4l2src device=/dev/video0 ! ... ! appsink` 管线，必要时加 `videoscale`、`videorate` 限制分辨率与帧率。

4. **运行与调优**
   - 首次启动加 `--warmup` 预热引擎，避免初次推理的缓存开销。
   - 在 Module 8 的异步模式下测试 `queue_size=2`、`drop_oldest=true`，防止 Nano 内存压力。
   - 如需进一步提速，可在配置中开启 INT8 引擎（需校准数据集）。

## 🧰 故障排查

- **GStreamer 打不开**：检查插件是否安装（`gst-inspect-1.0 nvarguscamerasrc`），确认摄像头权限。
- **TensorRT engine 不兼容**：确保在同一 JetPack 版本下生成；修改 `max_workspace_size` 以适配内存。
- **内存不足/显存溢出**：减小 `preprocess` 链条或启用 Module 8 的跳帧策略。
- **性能未达标**：
  - 确认 `nvpmodel`/`jetson_clocks` 已启用。
  - 降低模型尺寸（`yolov8n` → `yolov8n-int8`）。
  - 通过 `tegrastats` 观察 CPU/GPU 占用，按需调整线程数与分辨率。

## 📦 建议的部署结构

```
roadvision/
  models/
    yolov8n.engine
  scripts/
    install_jetson.sh      # 安装依赖
    export_trt.sh          # 模型转换
    run_jetson.sh          # 带 GStreamer/async 的启动脚本
```

------

# 运行示例

```bash
python main_preview.py
```

- 默认读取 `configs/default.yaml`。
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

- Module 10：边缘侧事件缓存 + 云端同步（违规截图、结构化数据回传接口）。
- Module 11：违规行为识别（压线、逆行等）与规则引擎可视化配置。

