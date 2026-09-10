# 地物变化实施任务

父规格：[Issue #6](https://github.com/arthuryin1314/change_dection_backend/issues/6)。父规格已同步 2026-09-08 面积展示位置修订。

用户已批准拆分，确认实际影像主要同网格，因此网格统一排在后面。下表顺序为实施优先级，依赖列才是硬阻塞关系。

| 切片 | 已发布任务 | 硬依赖 |
| --- | --- | --- |
| 2 | [#9 两期结果齐备后计算并保存矩阵](https://github.com/arthuryin1314/change_dection_backend/issues/9) | #7、#8 |
| 4b | [#11 自动补算、状态查询与重试](https://github.com/arthuryin1314/change_dection_backend/issues/11) | #9、#7（明确列出流水线依赖） |
| 5 | [#12 识别与变化检测历史](https://github.com/arthuryin1314/change_dection_backend/issues/12) | #9、#8 |
| 3 | [#13 自动统一空间网格](https://github.com/arthuryin1314/change_dection_backend/issues/13) | #9 |

待办为 #9、#11、#12、#13；#7 与 #8 已核对为 Completed，不再调度。#9 的 #7、#8 依赖已满足，可进入后续计划；尚未自动启动实施。原 #10 的查找与复用职责已并入 #9，#10 因合并关闭，不表示功能实现完成。

第一个面向用户的可发布单元是 #7 + #8。#9 整卡依赖 #8 的只读解析与选择状态，矩阵计算核心仍只依赖 #7；后端像元地表面积算法与容差契约唯一归 #7，前端单位换算唯一归 #8。

## 共同边界

- 两期完整空间分类结果必须均已保存、可读才能计算矩阵；不要求单期面积列表先实现。
- 公共有效区是空间交集内两期有效掩膜的 AND，不是两期 NoData 的交集；交集为空不生成成功矩阵。
- #13 完成前，网格不一致必须明确拒绝，不得静默生成矩阵。
- #9 完整保存矩阵及计算上下文并调用 #8 的版本匹配查询、负责检测前校验；#12 提供单期识别和变化矩阵两类历史列表及详情。
- 后端面积为平方米，前端单位转换由 #8 统一提供；#9 直接复用 #8 的共用模型、前后影像状态和单位转换。
- 真实尺寸影像性能测量是 #7、#8、#9 的实施验收要求；发布任务不代表这些验证已经执行。

前端仓库为 [arthuryin1314/change_dection](https://github.com/arthuryin1314/change_dection)，后端仓库为 [arthuryin1314/change_dection_backend](https://github.com/arthuryin1314/change_dection_backend)。Issue 在后端仓库统一跟踪，两端分别修改、验证并关联对应任务。


## 已完成

- #7：GitHub 状态已核对为 Completed。
- #8：两端 feature-compute-area 已推送，后端 fe5c734、前端 ed8915b；已发布完成评论并以 Completed 关闭。
- #8 完成记录：https://github.com/arthuryin1314/change_dection_backend/issues/8#issuecomment-5587484253
- #9 及后续原生依赖保留，不删除依赖历史。


## Issue #9 实现完成、验收待补

#9 实现已完成，验收部分通过；同图页面与网格不兼容拒绝已验证，不同真实影像成功矩阵与性能/I/O 待 #13 网格对齐后补验。不按全部验收通过关闭。代码仅本地提交，未推送。

本次评论草稿为 issue9-partial-acceptance.md；用户直接授权后已发布并回读核验，GitHub 保持 OPEN，标题标注部分通过。评论：https://github.com/arthuryin1314/change_dection_backend/issues/9#issuecomment-5602011526 。后续待实现为 #11、#12、#13，#9 仅保留待补验，不重复实现、不自动解除原生依赖。


