# Superblock (Pure Python RL Sandbox)

`Superblock` 是一个纯 Python 的强化学习实验项目（不依赖 PyTorch/TensorFlow，当前实现也不依赖 `numpy`）。

当前包含 4 层能力：

1. `motion`: Forward Model（`state + action -> next_state`）
2. `forage`: 觅食策略（含饥饿机制）
3. `evade`: 躲避 superhacker（含草丛掩护）
4. `survival`: 高层元策略 RL（食物与威胁同时存在）

---

## 1. 环境与安装

- Python: `>=3.10`
- 推荐使用虚拟环境

```bash
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install "[dev]"
```

如果你只想安装测试工具：

```bash
python -m pip install pytest
```

---

## 2. 训练入口

### 2.1 Motion

```bash
python -m superblock.train
```

默认输出：

- `artifacts/train.ckpt`
- `artifacts/dashboard.html`

### 2.2 Forage

```bash
python -m superblock.forage_train \
  --days 100 \
  --steps-per-day 100 \
  --food-count 3 \
  --hunger-interval 25 \
  --hunger-death-steps 75 \
  --vision-radius 3 \
  --motion-checkpoint-path artifacts/train.ckpt
```

默认输出：

- `artifacts/forage.ckpt`
- `artifacts/forage_metrics.csv`
- `artifacts/forage_dashboard.html`

### 2.3 Evade

```bash
python -m superblock.evade_train \
  --days 100 \
  --steps-per-day 100 \
  --grass-area 10 \
  --grass-count 3 \
  --food-count 3 \
  --hunger-interval 25 \
  --hunger-death-steps 75 \
  --vision-radius 3 \
  --motion-checkpoint-path artifacts/train.ckpt
```

默认输出：

- `artifacts/evade_metrics.csv`
- `artifacts/evade_dashboard.html`

### 2.4 Survival

```bash
python -m superblock.survival_train
```

默认输出：

- `artifacts/survival_meta.ckpt`
- `artifacts/survival_metrics.csv`
- `artifacts/survival_compare.csv`
- `artifacts/survival_dashboard.html`
- `artifacts/survival_runs/<run_id>/run.json`

---

## 3. UI 入口

### 3.1 旧 UI（motion/forage/evade）

```bash
python -m superblock.ui
```

### 3.2 Survival UI

```bash
python -m superblock.survival_ui
```

Survival UI 主要能力：

- 参数可视化编辑
- 训练曲线实时更新
- 环境网格实时渲染
- 草丛渲染颜色已调整为浅灰色，便于区分食物/威胁

---

## 4. Survival 参数优化（新）

### 4.1 UI 交互

在 `superblock.survival_ui` 中：

- `optimization_direction`: 选择优化方向
- `optimization_runs`: 设置本次方向要跑的训练次数 `n`
- `resume_unfinished_only`: 仅恢复当前方向尚未完成的 run
- 右侧中部显示 `ETA`
- `STOP FOR NOW`: 请求安全中止；已完成 run 会完整保留并进入 dashboard

### 4.2 内置优化方向（5 条）

1. `meta_lr_stability`
2. `low_level_unfreeze_balance`
3. `predator_pressure_robustness`
4. `resource_hunger_balance`
5. `perception_capacity_tradeoff`

### 4.3 Dashboard 分层结构

默认优化 dashboard：

- `artifacts/survival_optimization_dashboard.html`

默认优化状态文件：

- `artifacts/survival_optimization_status.json`

页面结构：

1. **Live Optimization Monitor**：当前批次进度、run 状态、参数表
2. **Direction Overview**：每个优化方向的总体效果（含多条 meta survival curve）
3. **Direction Subtables**：方向下每次训练结果（迭代参数高亮）
4. **Run Details**：单次训练详情（曲线、比较、参数）

归档 run 会写入优化元信息：

- `optimization.direction`
- `optimization.run_index` / `optimization.total_runs`
- `optimization.iterated_params`
- `optimization.iterated_values`

---

## 5. 回归测试

运行全量测试：

```bash
py -3 -m pytest -q
```

当前仓库测试基线（本次迭代后）：

- `47 passed`

---

## 6. 架构图

![Superblock Architecture](artifacts/project_architecture.svg)
