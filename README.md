# SuperBlock：让一个小方块探索、觅食和躲避

这是一个二维网格中的“方块生命体”实验。你可以把它理解为：把一个小方块放进地图，给它移动能力，再观察它能否找到食物、避免饿死，并在敌人出现时寻找草丛躲藏。

**项目目的：** 逐步研究“知道行动的后果 → 主动探索 → 寻找食物 → 应对威胁”。当前是可运行的研究原型，尚未完成自主生存智能体。运行只用 Python 标准库；测试使用 pytest，窗口界面使用 Tkinter。

## 先看懂：目前在做什么

| 部分 | 通俗解释 | 当前实际做法 |
|---|---|---|
| 运动预测 motion | 猜测“向右走一步后，会到哪里” | 小型神经网络学习当前坐标、动作与下一步坐标的关系 |
| 好奇心探索 curiosity | 优先去较少去过的位置 | 结合运动预测和访问次数，给候选动作打分 |
| 觅食 forage | 饿了以后去找食物，超时没吃到就死亡 | 可选择规则导航或表格 Q-learning |
| 躲避 evade | 看见敌人时尝试去草丛，同时还要吃饭 | 规则切换、草丛记忆和探索；目前没有单独训练避敌神经网络 |
| 高层生存 survival | 自己学会何时觅食、何时逃跑 | 计划；此前 README 中有介绍，但没有对应实现 |

“白天”“夜晚”“一天”是模拟中的采样与训练轮次，不是现实时间。默认每轮最多 100 步；死亡可能让这一轮提前结束。

## 第一版、后续版本，以及两种方法

历史版本和算法方法需要分别理解。本说明根据 Git 提交与现有源码归纳，**后续阶段没有正式、独立的 V2 发布号**；下面的 A/B 是阅读分类。

### 版本演变

| 阶段 | 当时的思路 | 现在如何找到 |
|---|---|---|
| 原始 V1，2026-02-16 | 四顶点表示方块刚体；随机行动收集样本，晚上训练运动预测模型，并渲染视野 | [V1 固定源码](https://github.com/clayzhao123/2D-SuperBlock/tree/7dc56343ea44b250b4bf4efe262d073338e34238) |
| 能力扩展，2026-02-17～18 | 在运动模型上增加好奇心、食物记忆、饥饿规则；随后加入可选 Q-learning | 当前 agents/ 中的两种觅食方法及共享探索 |
| 单格与躲避，2026-02-19～20 | 默认身体改成单个位置；增加敌人、草丛，再把觅食与躲避组合 | 当前 env.py、forage_env.py、evade_env.py 与训练入口 |

当前代码沿用了早期模型的一些维度与动作编码，因此“当前单格版本”不能直接当成“原始四顶点版本”运行。详细的历史证据、原始思路变化和未合并分支见 [版本沿革](docs/history.md)。

### 方法 A：运动模型辅助探索 + 规则觅食

先让运动模型学习“动作会带来什么位置变化”；探索时参考预测和访问记忆。饥饿后，由规则挑选接近食物的动作。

原始觅食实现用模型预测下一步来导航。后续为避免预测误差，**当前规则觅食默认用环境的真实下一步结果导航**，运动模型仍服务于好奇心探索。

这相当于“学习运动知识，再按规则行动”，并不意味着觅食决策本身已经通过奖励训练出来。默认选择：`--policy heuristic`。策略在 [agents/heuristic.py](superblock/agents/heuristic.py)，共享探索在 [agents/curiosity.py](superblock/agents/curiosity.py)。

### 方法 B：表格 Q-learning 觅食

把位置、饥饿状态、食物方向等压缩成状态，在一张 Q 表中记录“某个状态下，某个动作有多值得做”。吃到食物加分，死亡扣分，反复尝试后更新 Q 表。

这是通过奖励学习动作选择的强化学习方法。它复用同一套地图、食物和饥饿规则，选择：`--policy qlearn`。策略在 [agents/qlearn.py](superblock/agents/qlearn.py)。

两种方法都还能读取部分完整环境信息，尚不是严格的“只凭有限视野自主学习”。觅食入口也都会加载运动 checkpoint；B 的动作由 Q 表选择，这项文件依赖不表示它用运动模型训练 Q 表。更详细的区别见 [方法说明](docs/methods.md)。

## 目前效果怎么样

2026-10-07 对当前实现做了一次小规模运行检查，seed 为 7、42、2026。每个 seed 的 motion 运行 5 轮，每晚 3 个训练 epoch；两种觅食方法及 evade 各运行 10 轮，每轮最多 100 步，其余使用默认参数。

| 部分 | 本次观察 | 可以说明什么 |
|---|---|---|
| 运动预测 | 最后一轮坐标 MSE 为 0.0063～0.0099，近似准确率为 14%～29% | 短训练可以执行，但不足以确认预测可靠；精确准确率存在维度问题，见下文 |
| A：规则觅食 | 累计觅食成功率 79.2%～85.0%，各 10 轮均无饥饿死亡 | 在这组设置下规则导航能完成较多觅食尝试 |
| B：Q-learning | 累计觅食成功率 0%～9.1%，10 轮中死亡 9～10 次 | 在这次有限预算下未表现出稳定觅食能力，不能据此判定长训练的最终上限 |
| 躲避 + 觅食 | 完整存活 8～10 轮，即 80%～100% | 规则流程能运行；不表示已经学会通用避敌策略 |

这是训练期观察，**不是独立测试集上的成绩，也不是 A/B 效果提升的证明**。成功率是“成功吃到食物的尝试 / 饥饿触发的尝试”；尚未结束的饥饿尝试也进入分母。死亡会减少实际行动步数。规则和 Q-learning 的有效信息、学习状态、动作编码还需要进一步统一。

尤其需要先处理：当前单格状态只有 2 个坐标，默认运动模型仍输出 8 个值，精确准确率比较了不同长度列表，所以会一直为零；模型辅助探索还可能把多余输出当成身体坐标。默认单格动作中的腿编号和转向也有冗余。因此不应把旧 score 当成学习能力的最终结论。

完整逐 seed 数据、参数、复现步骤和优先修复事项见 [运行观察](docs/results.md)。原始实验数据默认留在本地 artifacts，不上传模型文件或大体积日志。

## 安装：从这里开始

需要 Python 3.10 或更新版本。克隆或下载本仓库，进入包含 README.md 的目录。

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

不运行测试时，可用 `python -m pip install -e .`。原 `python -m pip install -r requirements.txt` 也会安装项目和测试依赖。GUI 若缺少 Tkinter，需要安装当前 Python 发行版的 Tk 支持；命令行运行不需要显示器。

## 第一次运行：看短演示

```bash
python -m superblock.demo
```

它依次跑 motion、Q-learning forage 和 evade，生成三个独立实验目录。打开终端给出的 master_dashboard.html，查看面板。这个短演示验证流程与输出，不代表已经训练成功。

## 分别运行两种方法

先生成共享运动模型：

```bash
python -m superblock.experiment motion --run-id motion-example -- --max-days 5 --steps-per-day 100 --epochs-per-night 3
```

A：规则觅食：

```bash
python -m superblock.experiment forage --run-id forage-heuristic -- --motion-checkpoint-path artifacts/runs/motion-example/train.ckpt --policy heuristic --days 10 --steps-per-day 100
```

B：Q-learning 觅食：

```bash
python -m superblock.experiment forage --run-id forage-qlearn -- --motion-checkpoint-path artifacts/runs/motion-example/train.ckpt --policy qlearn --days 10 --steps-per-day 100
```

躲避 + 觅食：

```bash
python -m superblock.experiment evade --run-id evade-example -- --motion-checkpoint-path artifacts/runs/motion-example/train.ckpt --days 10 --steps-per-day 100
```

`--` 前是实验目录选项，后是训练器参数。同名实验会拒绝覆盖；重复运行请换名字或省略 --run-id。

## 结果在哪里、怎么看

每次实验位于 artifacts/runs/<实验名>/。checkpoint（.ckpt）是保存的模型或状态；CSV 是可用表格软件打开的指标；HTML 是用浏览器打开的面板。

| 文件 | 作用 |
|---|---|
| config.json | 保存实际参数、seed、可得的代码版本与输入文件指纹 |
| run.json | 保存运行、完成、失败或中断状态 |
| train.ckpt / forage.ckpt | 运动模型 / 觅食策略状态；evade 暂无独立策略 checkpoint |
| *_metrics.csv | 每轮结果，适合比较成功率与死因 |
| *dashboard.html | 阶段曲线与总览 |

觅食优先看成功率、死亡和成功后的耗时，不能只看运动 score；耗时为 -1 表示没有成功样本，不能解释为“很快”。evade 的存活率指撑过本轮步数预算，不是无限期生存。

## 我想改代码，该去哪里

| 想改什么 | 主要位置 |
|---|---|
| 地图、运动、食物、饥饿、敌人与草丛 | env.py、forage_env.py、evade_env.py |
| 方法 A：规则觅食 | agents/heuristic.py |
| 方法 B：Q-learning | agents/qlearn.py |
| 共享探索与运动模型 | agents/curiosity.py、models.py、buffer.py |
| 每轮如何执行与统计 | train.py、forage_train.py、evade_train.py |
| 实验归档 | experiment.py |
| CSV、曲线与总览 | reporting/、master_dashboard.py |
| 窗口操作 | ui.py |

以上代码路径位于 superblock/。原 forage_agent.py 和 policy_curiosity.py 保留为兼容入口，实际实现集中在 agents/。完整职责和兼容约定见 [文件导航](docs/architecture.md)。

原 CLI 和 GUI 仍可用：`python -m superblock.train`、`python -m superblock.forage_train`、`python -m superblock.evade_train`、`python -m superblock.explore`、`python -m superblock.ui`。完整参数用 --help 查询。

## 下一步优先做什么

1. 对齐单格 / 四顶点状态、运动模型维度与动作编码，修复指标含义。
2. 区分可见信息与完整环境状态，明确规则基线和有限视野实验。
3. 加入独立评估与更充分的多 seed 比较，再判断 Q-learning 是否有效。
4. 完善随机状态恢复、觅食/躲避续训，随后再设计 Survival 元策略。

功能状态以 [实现清单](docs/status.md) 为准，实验与续训规则见 [实验管理](docs/experiments.md)。

## 检查与使用边界

```bash
python -m compileall -q superblock scripts
python -m pytest -q
python -m superblock.demo
```

GitHub Actions 在 Python 3.10 / 3.12 上运行测试及安装包演示。2026-10-06 的维护基线为 55 项测试通过；本次改动的结果以 PR Checks 为准。测试通过说明已覆盖的行为能工作，不能代替学习效果评估；GUI 窗口交互需要手工验收。

checkpoint 使用 pickle，只加载可信输出。artifacts 已被 Git 忽略，需要保留实验时自行备份整个目录。

许可证：[MIT](LICENSE)。
