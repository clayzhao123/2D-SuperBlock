# Superblock v1.2

这个版本聚焦三个目标：

1. 训练记录可落盘（checkpoint + CSV）；
2. 训练过程可视化（dashboard + 本地交互 UI）；
3. 支持改环境（可视单元格）后继续运行。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) 命令行训练（支持断点与记录）

```bash
python -m superblock.train
```

默认会持续产出：

- `artifacts/train.ckpt`：模型 + buffer + 历史数据 + 可视单元格
- `artifacts/dashboard.html`：可视化曲线（score / MSE / 信任图）
- `artifacts/metrics.csv`：逐天训练记录（可直接 Excel/Pandas 打开）

### 断点恢复

```bash
python -m superblock.train --resume --max-days 300
```

### 改可视单元格后继续

```bash
python -m superblock.train \
  --resume \
  --override-visible-cells \
  --visible-cells 19:19,20:19,21:19
```

### 仅白天运行（不更新模型）

```bash
python -m superblock.train \
  --resume \
  --daytime-only \
  --override-visible-cells \
  --visible-cells 19:19,20:19,21:19
```

## 2) 交互可视化 UI（像素风黑白）

```bash
python -m superblock.ui
```

UI 结构：

- 左侧：环境网格 + superblock + 最近 5 天轨迹叠加（更老轨迹自动丢弃）；
- 右侧上：参数面板
  - 可视单元格坐标（支持多个）
  - superblock 数量（v1.3 预留，仅参数入口）
  - superblock 初始位置（4 点坐标）
  - 训练天数（默认 100）
  - 每天采样步数（默认 100）
- 右侧中：`RUN` 按钮（点击后会二次确认）；
- 右侧下：实时指标 + “信任图”（横轴天数，纵轴当天命中可视单元格次数）。

UI 训练时也会写入：

- `artifacts/ui_train.ckpt`
- `artifacts/ui_dashboard.html`
- `artifacts/ui_metrics.csv`

## 评分

当前默认：

- `score = 0.5*exp(-k*mse) + 0.2*exact_acc + 0.3*near_acc`

并输出：

- `pos_mse`
- `exact_acc`
- `near_acc`
- `visible_hits`（信任图原始统计）

## Roadmap

- [x] v1.0：基础环境 + 日夜训练
- [x] v1.1：dashboard + resume + 可视单元格配置
- [x] v1.2：训练记录 CSV + 交互 UI + 轨迹叠加 + 信任图
- [ ] v1.3：多 superblock 交互（已预留参数入口）

## 测试

```bash
pytest -q
```

## FAQ

### 为什么 superblock 是 4 个单元格？
当前环境是一个 **1x1 面积的刚体方块**，用 4 个顶点（也就是 4 个网格单元角点）表示。
训练输入/动作编码也固定按 4 个 leg 设计（`leg_id` 为 0~3，状态是 8 维坐标），所以 v1.2 里 superblock 数量字段只是为 v1.3 预留。

### 为什么信任图看起来“没有输出”？
信任图实际画的是每天 `visible_hits`（命中可视单元格的次数）。
如果你把可视单元格设为 `19:19`，但 superblock 多数时间在左下区域活动，那么 100 步里可能一直是 0，图上会是一条贴底线。
v1.2 已在 UI 中加了提示文本：当全部为 0 会显示“当前可视单元格未被命中”。
可以把可视单元格改到初始区域附近（如 `2:2,3:2`）验证曲线变化。
