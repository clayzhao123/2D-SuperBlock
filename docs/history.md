# 历史版本与两条方法的来源

本页核对 Git 提交、PR 描述与固定版本源码。V1 是原始 README 的名称；“后续版本”是对功能演变的归纳，不创建一个原本不存在的正式 V2 标签，也不把任意两个旧分支当成两套完整项目。

## 原始 V1：先学习运动后果

2026-02-16 的 [V1 提交 7dc5634](https://github.com/clayzhao123/2D-SuperBlock/tree/7dc56343ea44b250b4bf4efe262d073338e34238) 和 [PR #1](https://github.com/clayzhao123/2D-SuperBlock/pull/1) 实现：

- 四个顶点表示方块刚体，坐标组成 8 维状态；平移改变位置，转向通过顶点标签重排表达。
- 白天随机行动，把“当前状态、动作、下一状态”放进 ReplayBuffer。
- 晚上用监督学习训练 ForwardModel 预测下一状态。
- 渲染观察边的视野图像，但模型输入来自坐标与动作，未实现“只看图像学会移动”。

因此第一版核心是运动预测，不是奖励驱动的觅食强化学习。要看原貌，打开上面的固定源码链接；当前 main 不再默认使用四顶点身体。

## 后续 A：用模型辅助探索，按规则找食物

2026-02-17 的 [提交 f866329](https://github.com/clayzhao123/2D-SuperBlock/commit/f8663290cfc1ccff08c7e97a55f5e42b59cb1288) 和 [PR #6](https://github.com/clayzhao123/2D-SuperBlock/pull/6) 加入好奇心、食物记忆、饥饿和独立觅食循环。

当时的 ForagePolicy 用模型预测每个动作后的坐标，选取更接近记忆食物的动作。后续加入真实 peek_step 导航，当前默认不再用模型预测食物导航，以减少模型误差的影响。共享好奇心探索仍会使用模型。

2026-02-18 的 [PR #11](https://github.com/clayzhao123/2D-SuperBlock/pull/11) 修正占据格、食物距离和 overlap 摄食的一致性；[PR #13](https://github.com/clayzhao123/2D-SuperBlock/pull/13) 清理消失食物的记忆。

这些属于方法 A 的演进，不是另一个完整的学习算法。

## 后续 B：可选的表格 Q-learning

2026-02-18 的 [提交 4008e06](https://github.com/clayzhao123/2D-SuperBlock/commit/4008e06be5155bac9baed90abdfceb5c80e16f64) 和 [PR #10](https://github.com/clayzhao123/2D-SuperBlock/pull/10) 加入 Q 表、奖励更新和 --policy 切换，默认仍为 heuristic。

它与 A 共享 ForageEnv，但通过奖励更新动作价值；因此本次把 heuristic.py 和 qlearn.py 放在同一个 agents/ 目录，curiosity.py 作为共享组件。没有复制两套地图、训练循环或报表。

## 身体与任务继续变化

2026-02-19 的 [提交 0f7a7bd](https://github.com/clayzhao123/2D-SuperBlock/commit/0f7a7bdc6c9c06b39ab5856bca3b3c373e5ff1b4) 将默认身体从四顶点改为单格，并加入敌人和草丛。[PR #21](https://github.com/clayzhao123/2D-SuperBlock/pull/21) 加入草丛导航与 checkpoint 兼容。

2026-02-20 的 [提交 646ac18](https://github.com/clayzhao123/2D-SuperBlock/commit/646ac1840a91914cc899b12439b20a1859d7b118) 扩充了 evade 的觅食、饥饿及死因统计，也修改了 README。

这不是“所有任务都变成 Q-learning”。当前 evade 没有更新独立的逃跑策略权重；即使传入 forage checkpoint，也只是解析运动模型路径，不会加载其中 Q 表来执行联合策略。

## README 曾超前于实现

646ac18 的 README 描述了 survival_train、survival_ui 和五个参数优化方向，源码没有对应入口。2026-10-06 核查 main 与当时其他 22 个分支也未找到这些实现。因此标注为未实现，保留为后续计划，不用说明文字证明代码存在。

[PR #4](https://github.com/clayzhao123/2D-SuperBlock/pull/4) 的四点说明与旧 UI 改动在本次核查时仍未合并；它是历史工作分支，不代表当前默认身体。旧分支保留历史，不进行批量删除。

## 本次整理的边界

保留历史来源、两种觅食方法和原入口；集中策略实现，解释当前效果。没有重新实现原始 V1，也没有补造缺失的 Survival 第二版。仓库提交中没有可供核验的完整历史训练结果，本次实测不能代表你过去在本机训练出的最佳表现。
