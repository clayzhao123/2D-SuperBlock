# 实验与结果管理

## 一次运行，一个目录

```bash
python -m superblock.experiment motion -- --max-days 2 --steps-per-day 16 --epochs-per-night 1
```

默认创建 artifacts/runs/<时间-stage-随机后缀>/。可自定义 --runs-dir 和 --run-id，已有名字拒绝覆盖。

| 文件 | 用途 |
|---|---|
| config.json | 原始/完整参数、Python/代码版本与输入 |
| run.json | running / completed / failed / interrupted、时间、文件与错误 |
| train.ckpt / forage.ckpt | 阶段 checkpoint；evade 当前只输出指标和面板 |
| input_motion.ckpt | 输入运动模型副本，config 保存 SHA-256 |
| *_metrics.csv、forage_attempts.csv | 指标与觅食尝试 |
| *dashboard.html | 阶段面板和总览 |

归档入口管理所有输出，不接受训练器自定义输出路径；用 --runs-dir 控制位置。若要直接控制每个文件，使用原训练器。

缺失输入或训练失败会保留 failed 记录并返回异常。强制杀进程/断电可能留下 running，需要人工核查。

从 Git checkout 运行会记录 commit 和 dirty；安装包没有 Git 元数据时记录 null，不猜测版本。复现实验尽量使用已提交代码。

## motion 续训

归档入口创建新实验；续训使用原入口：

```bash
python -m superblock.train --resume --checkpoint-path artifacts/runs/<已有motion实验>/train.ckpt --dashboard-path artifacts/runs/<已有motion实验>/dashboard.html --master-dashboard-path artifacts/runs/<已有motion实验>/master_dashboard.html --max-days 20
```

--max-days 是总目标天数。直接续训会更新训练文件，但不更新 experiment 原始 config/run 元数据；比较实验优先新建归档运行。目前没有统一的 forage/evade 续训入口。

## 比较与备份

相同预算、环境与观测条件下，用多个 seed 比较。觅食看成功率、耗时和死亡；无成功的耗时为 -1。躲避看存活与死因，不只看曲线。规则与 Q-learning 分开标注。

本轮没有新增跨实验统计器，也不承诺效果提升。artifacts/checkpoint 不进入 Git，保留时备份完整实验目录，不能只留截图。

## 修改流程

运行 README 的 syntax / pytest / demo，再手工验收受影响的 CLI 或 GUI。一次 PR 处理一类问题，同步更新 status；未执行的验收写“未验证”。
