# Superblock V1 Prototype

一个可复现（固定随机种子）的本地 CPU 原型，实现了：

- 40x40 网格中的 `Superblock` 刚体运动（平移 + 90°离散转向）
- 基于观察边的视野渲染（近大远小透视效果）
- “白天采样 / 晚上理解训练”循环
- 训练过程可视化（HTML Dashboard）
- 自动断点保存与恢复训练
- `pytest` 验证核心几何与渲染规则

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行训练（含可视化）

```bash
python -m superblock.train
```

训练时会持续更新：

- `artifacts/dashboard.html`：可视化监控页面（分数曲线 / MSE 曲线 / 最近日志表）
- `artifacts/train.ckpt`：断点文件（模型 + buffer + 历史指标 + 可视单元格配置）

每晚打印：

- `day_idx`
- `invalid_move_ratio`
- `pos_mse`
- `exact_acc`
- `near_acc`
- `score`

## 断点恢复

中断后继续：

```bash
python -m superblock.train --resume --max-days 300
```

说明：

- 会从 `--checkpoint-path` 中恢复模型/数据，继续 day_idx。
- 默认也会恢复上次的 `visible_cells` 配置。

## 可配置可视单元格

例如设置 3 个可视单元格：

```bash
python -m superblock.train --visible-cells 19:19,20:19,20:20
```

如果你已经有 checkpoint，想在恢复时改成**新环境**（新增/修改可视单元格）：

```bash
python -m superblock.train \
  --resume \
  --override-visible-cells \
  --visible-cells 19:19,20:19,21:19
```

## 仅白天运行（不做夜间训练）

你提到“训练结束后更改环境，再让它白天在新环境运行”，可以用：

```bash
python -m superblock.train \
  --resume \
  --override-visible-cells \
  --visible-cells 19:19,20:19,21:19 \
  --daytime-only \
  --max-days 260
```

这会继续采样白天数据并更新可视化，但不会更新模型参数（`epochs=0`）。

## 评分为什么会卡在 0.495 附近

旧评分：

- `score = 0.5 * exp(-k * pos_mse) + 0.5 * exact_acc`

当 `pos_mse≈1.8e-4, k=50` 时第一项约 `0.495`；而 `exact_acc` 常接近 0（离散全维完全命中太苛刻），总分就会看起来“卡住”。

现在默认评分是：

- `score = w_mse*exp(-k*mse) + w_exact*exact_acc + w_near*near_acc`

并新增 `near_acc`（±1 格内视作近似命中），可更真实反映学习进展。

## 路线图（Roadmap）

- [x] **V1 原型**：规则环境、渲染、前向模型、日夜循环
- [x] **V1.1**：评分增强（`near_acc`）+ 可视化 dashboard + 断点恢复
- [ ] **V1.2**：训练曲线 CSV 导出、误差分解、采样策略优化
- [ ] **V1.3**：模型升级（更强 MLP / 基线对比 / 模型存取）
- [ ] **V2**：图像特征分支、多任务目标、实验管理面板

## 运行测试

```bash
pytest -q
```

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
  utils.py
tests/
  test_env.py
  test_render.py
```
