# Road Vision System

#### author:YJxyzxyz

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

## 📁 目录&文件（最小骨架）

修改调参说明

**默认开链**（`preprocess.enabled: true`），画面对比度应更清晰（CLAHE），细雨噪点略弱（Median）。

关链：把 `enabled: false` 对比性能与视觉差异。

改 `CLAHEDehaze.params`：

- `space: "LAB"` 往往偏“自然”，`"YCrCb"`偏“锐”；
- `clip_limit` 
- `tile_grid` 越大块对比度自适应越细腻，但开销略增。

改 `MedianDerain.params.ksize` 为 5 看“去纹理”的力度（过大会糊）。

