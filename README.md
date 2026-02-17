# Superblock v1.3

纯 Python（无新增第三方依赖）的 `Superblock` 训练与行为实验项目。

当前包含两条**独立**能力链路：

1. **运动能力训练（ForwardModel）**：学习 `state_t + action -> state_{t+1}`。
2. **觅食能力训练（Foraging）**：在饥饿约束下，用“好奇心探索 + 食物记忆导航”提高觅食成功率。

---

## 功能总览

- 40x40 网格中的 `Superblock` 刚体运动（平移 + 90°离散转向）
- 前向运动模型 `ForwardModel`（纯 Python MLP）
- 白天采样 / 夜间训练流程（运动训练）
- Dashboard + Checkpoint + CSV 指标落盘
- **Curiosity 探索策略**（使用已训练 ForwardModel 预测下一步）
- **Foraging 环境**（食物、饥饿触发、死亡窗口、成功率统计）
- `pytest` 覆盖核心环境与渲染规则，以及觅食规则

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1) 运动能力训练（原有主训练）

运行：

```bash
python -m superblock.train
```

默认产物：

- `artifacts/train.ckpt`：模型权重 + buffer + history + visible_cells
- `artifacts/dashboard.html`：训练监控（score/MSE）

常用参数：

```bash
python -m superblock.train --help
```

断点恢复：

```bash
python -m superblock.train --resume --max-days 300
```

修改可视单元格继续训练：

```bash
python -m superblock.train \
  --resume \
  --override-visible-cells \
  --visible-cells 19:19,20:19,21:19
```

仅白天采样（不更新模型）：

```bash
python -m superblock.train \
  --resume \
  --daytime-only
```

---

## 2) 好奇心探索（利用已训练运动模型，不重新训练）

运行：

```bash
python -m superblock.explore \
  --checkpoint-path artifacts/train.ckpt \
  --steps 500 \
  --seed 42
```

说明：

- 从 `train.ckpt` 中只加载 `ForwardModel` 权重；
- 基于访问记忆（中心格访问计数）计算新颖度；
- 在 8 个候选动作中选最高分动作（含 invalid 惩罚 + epsilon 探索）；
- 输出覆盖率，并写 `artifacts/curiosity_report.txt`。

---

## 3) 觅食训练（独立入口，不与 train.py 混写）

运行：

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

默认产物：

- `artifacts/forage.ckpt`
- `artifacts/forage_metrics.csv`
- `artifacts/forage_dashboard.html`

规则摘要：

- 每 `hunger_interval` 步触发一次饥饿（计为一次觅食尝试）。
- 进入饥饿后，若 `hunger_death_steps` 内未成功吃到食物则死亡，当天提前结束。
- 吃到食物后计一次成功并重置饥饿状态。
- 非饥饿时以 Curiosity 探索；饥饿且已有食物记忆时朝最近记忆食物导航。

---

## 4) 交互 UI（运动训练）

```bash
python -m superblock.ui
```

> 当前仓库保留原运动训练 UI；觅食训练为独立 CLI 入口（`superblock.forage_train`）。

---

## 测试

```bash
pytest -q
```

---

## 项目结构

```text
superblock/
  __init__.py
  env.py
  render.py
  buffer.py
  models.py
  monitor.py
  train.py
  ui.py
  policy_curiosity.py
  explore.py
  forage_env.py
  forage_agent.py
  forage_train.py
  utils.py
tests/
  test_env.py
  test_render.py
  test_forage_env.py
```
