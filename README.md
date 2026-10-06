# SuperBlock

纯 Python 的二维网格学习实验台，支持运动预测、觅食与躲避；运行不依赖 NumPy、PyTorch 或 TensorFlow。GUI 使用 Tkinter。

**当前状态：实验原型。** Survival 元策略、Survival UI 和自动参数优化尚未实现。规则策略与 Q-learning 的能力边界见 [实现状态](docs/status.md)。

## 安装

需要 Python 3.10 或更新版本。下载/克隆本仓库，进入根目录。

Windows PowerShell：

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

不需要测试工具时，最后一条换成 `python -m pip install -e .`。原 `python -m pip install -r requirements.txt` 也会安装项目与测试依赖。

GUI 若提示缺少 Tkinter，需要安装当前 Python 发行版的 Tk 支持；Ubuntu 系统 Python 通常使用 python3-tk。命令行训练不需要显示器。

## 先运行短演示

```bash
python -m superblock.demo
```

演示依次执行少量 motion、qlearn forage、evade 步骤，生成三个独立实验目录。终端打印各目录和总览路径，用浏览器打开 master_dashboard.html 查看结果。

这验证执行链路与输出，**不代表模型已经学会觅食或躲避**。

## 自己运行实验

先训练运动模型：

```bash
python -m superblock.experiment motion --run-id motion-baseline -- --max-days 10 --steps-per-day 100 --epochs-per-night 10
```

用它的 checkpoint 运行 Q-learning 觅食：

```bash
python -m superblock.experiment forage --run-id forage-baseline -- --motion-checkpoint-path artifacts/runs/motion-baseline/train.ckpt --policy qlearn --days 10
```

再运行躲避：

```bash
python -m superblock.experiment evade -- --motion-checkpoint-path artifacts/runs/motion-baseline/train.ckpt --days 10
```

`--` 前是实验目录选项，后是原训练器参数。不填写 --run-id 会自动生成名字；已有同名目录会报错，不覆盖结果。参数、输入文件指纹和运行状态随结果保存。详见 [实验管理](docs/experiments.md)。

## 原有入口

| 入口 | 用途 |
|---|---|
| python -m superblock.train | 运动预测；默认生成 artifacts/train.ckpt |
| python -m superblock.forage_train | 觅食；依赖运动 checkpoint，默认 heuristic |
| python -m superblock.evade_train | 躲避 + 觅食；依赖运动 checkpoint |
| python -m superblock.explore | 探索覆盖度 |
| python -m superblock.ui | 原有 motion/forage/evade GUI |
| python -m superblock.demo | 短演示 |
| python -m superblock.experiment | 独立实验归档 |

各训练器完整参数用 --help 查询，例如 `python -m superblock.forage_train --help`。

原训练器沿用固定 artifacts 输出路径，日常实验推荐归档入口。motion 续训继续使用原入口的 --resume；归档入口只创建新实验。

## 修改与检查

- [文件导航](docs/architecture.md)：一个需求主要改哪些文件。
- [实现状态与下一步](docs/status.md)：已实现、占位和计划。
- [实验与结果管理](docs/experiments.md)：参数、输出、恢复和比较。

```bash
python -m compileall -q superblock
python -m pytest -q
python -m superblock.demo
```

GitHub Actions 在 Python 3.10 / 3.12 上执行检查，并从仓库外运行安装后的 wheel。测试数量以实际结果为准。

## 使用边界

当前觅食策略包含规则导航和可读取环境内部状态的基线，不能直接解释为严格有限视野学习。用多个 seed 和明确指标评估效果。

Checkpoint 使用 pickle，只加载可信的本项目输出。运行数据位于 artifacts，已被 Git 忽略，需要自行备份。

许可证：[MIT](LICENSE)。
