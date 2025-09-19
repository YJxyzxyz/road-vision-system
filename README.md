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

- Module 8：引入多线程采集与推理、异步队列，提高实时性。
- Module 9：补充 Jetson / TensorRT 推理路径与硬件接口。

