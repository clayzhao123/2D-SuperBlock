# Superblock V1 Prototype

一个可复现（固定随机种子）的本地 CPU 原型，实现了：

- 40x40 网格中的 `Superblock` 刚体运动（平移 + 90°离散转向）
- 基于观察边的视野渲染（近大远小透视效果）
- “白天采样 / 晚上理解训练”循环
- `pytest` 验证核心几何与渲染规则

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行训练

```bash
python -m superblock.train
```

每晚打印：

- `day_idx`
- `invalid_move_ratio`
- `pos_mse`
- `exact_acc`
- `score`

可选参数示例：

```bash
python -m superblock.train --seed 42 --steps-per-day 100 --max-days 200
```

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
  train.py
  utils.py
tests/
  test_env.py
  test_render.py
```
