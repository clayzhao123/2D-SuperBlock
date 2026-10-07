# 架构与文件导航

本轮只拆出明确独立的职责，没有改变运动规则、奖励或学习参数。

| 想改什么 | 首先查看 | 验证 |
|---|---|---|
| 网格运动与渲染 | env.py、utils.py、render.py | test_env、test_render |
| 食物与饥饿 | forage_env.py | test_forage_env |
| 方法 A：规则觅食 | agents/heuristic.py | test_forage_policy |
| Q-learning | agents/qlearn.py | test_maintenance、test_forage_train |
| 共享模型辅助探索 | agents/curiosity.py | test_forage_policy、demo |
| 运动模型 | models.py、buffer.py | test_train |
| 训练循环与指标 | train.py、forage_train.py、evade_train.py | 对应 test_*_train |
| CSV 与阶段面板 | reporting/forage.py、evade.py | test_forage_train、demo |
| 共享 SVG | reporting/charts.py | test_monitor、demo |
| 总览与链接 | master_dashboard.py、reporting/summary.py | test_master_dashboard、test_maintenance |
| checkpoint | checkpoint.py、monitor.py | test_maintenance |
| 实验归档 | experiment.py | test_maintenance |
| GUI | ui.py | test_ui + 手工 GUI 验收 |

上述路径均位于 superblock/。

## 职责

环境提供状态与转移，策略选择动作，训练循环更新模型/策略并收集指标。reporting 输出报告，checkpoint 保存状态。experiment 复用训练器，不实现第二套算法。

agents/ 并列放置两种觅食方法 heuristic.py / qlearn.py 和共享探索 curiosity.py。历史版本留在 Git 的固定提交中，避免复制出两套逐渐分叉的环境和报表。小规模观察脚本在 scripts/inspect_methods.py；结果说明在 docs/results.md。

## 兼容约定

- 保留原 CLI 和 GUI 入口。
- QLearnForagePolicy 在 agents/qlearn.py，旧 forage_train 导入位置仍可用。
- forage_agent.py / policy_curiosity.py 保留旧类和函数的导入，实际实现分别在 agents/heuristic.py / curiosity.py。这些兼容文件不放第二份策略逻辑。
- monitor.load_checkpoint / save_checkpoint 的调用和 payload 格式兼容。
- Namespace 没有 master_dashboard_path 时仍使用 artifacts/master_dashboard.html。
- 新实验总览只读取本目录和明确输入 checkpoint，链接相对总览生成。

## 后续拆分

需要改 GUI 时再拆 ui.py，环境文件暂时保留原路径。先让职责可定位，不为了形式引入框架或更多层级。
