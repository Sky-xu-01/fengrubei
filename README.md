# 实验室三维温度场预测项目（2026 最新版）

本项目用于构建实验室 CFD 场的深度学习代理模型，目标是用毫秒级推理替代高成本全量仿真。
当前版本为“6 物理场联合预测”，并在 3D UNet 中引入 SE3D 注意力模块，强化全局空间关系建模能力。

---

## 1. 建模版本概览

### 1.1 输入/输出定义

- 网格尺寸：`32 x 64 x 24`（`nz, ny, nx`）
- 时间策略：Time-as-Channel（时间步展平成通道）
- 输入历史步长：`input_steps=2`
- 预测步长：`pred_steps=1`（可扩展）

**每个输入时间步 20 通道：**

1) 动态通道（8）
- `T, U, V, W, K, NUT, Q_SRC, Global_Avg_T_Channel`

2) 静态/控制通道（12）
- `Wall_Dist, Inlet_Mask, Z_Coord, Time_Step, Cell_Zone_Mask`
- `Current_Time_Norm, Fluid_Avg_T, Solid_Avg_T`
- `AC_Vel_Set, Fan_Vel_Set, AC_Temp_Set, Fan_Temp_Set`

**输出通道：**

- 每个预测步输出 6 通道：`T, U, V, W, K, NUT`
- 输出形状：`(B, pred_steps * 6, Z, Y, X)`

### 1.2 模型结构更新

- 主干：`models/unet3d.py` 中的 `UNet3DTimeAsChannel`
- 新增：`SE3D`（3D Squeeze-and-Excitation）
- 插入位置：每个 `Down` 后 + `Bottleneck` 后
- 目的：增强网络对全局空间统计关系（如全局温度阈值效应、远程耦合特征）的建模能力

### 1.3 损失函数更新

`train.py` 使用加权多通道 MSE：

- 温度通道损失：`MSE(T)`
- 流场通道损失：`MSE(U,V,W,K,NUT)`
- 总损失：

$$
\mathcal{L}=0.7\cdot \text{MSE}_T + 0.3\cdot \text{MSE}_{flow} + \lambda_{smooth}\cdot \mathcal{L}_{smooth}
$$

其中 `lambda_smooth` 默认 `1e-4`。

---

## 2. 数据要求与字段说明

### 2.1 Fluent 导出文件要求

`convert_data.py` 默认读取目录：`fluent_output_2/`（文件名 `FFF*`）。

CSV/ASCII 必需列（由 `config.py -> csv_mapping` 管理）：

- 坐标：`x-coordinate`, `y-coordinate`, `z-coordinate`
- 物理场：`temperature`, `x-velocity`, `y-velocity`, `z-velocity`, `turb-kinetic-energy`, `viscosity-turb`
- 新增：`user-volumetric-energy-source`, `cell-zone`

### 2.2 预处理后 PKL 关键字段

- 动态场：`T/U/V/W/K/NUT/Q_SRC`
- 几何与掩码：`Z_Coord`, `Cell_Zone_Mask`, `Wall_Dist`, `Inlet_Mask`
- 坐标：`X/Y/Z`

`Cell_Zone_Mask` 用于区分流体/固体区域；`Global_Avg_T_Channel` 在 `data/dataset.py` 中在线计算（不是离线存盘字段）。

此外，`convert_data.py / convert_data_cpu.py / convert_data_gpu.py` 会在预处理阶段复刻 UDF 控制逻辑并写入以下字段：

- `Current_Time, Current_Time_Norm`（由文件名末尾步号提取）
- `Fluid_Avg_T, Solid_Avg_T`（按体积加权平均温度计算）
- `AC_Vel_Set, Fan_Vel_Set, AC_Temp_Set, Fan_Temp_Set`（按 UDF 阈值逻辑计算）

---

## 3. 一键流程（推荐）

### 3.1 环境准备

```bash
conda create -n labtemp-gpu python=3.10 -y
conda activate labtemp-gpu
pip install -r requirements.txt
```

如果需要 CUDA，请安装与本机驱动匹配的 CUDA 版 PyTorch。

### 3.2 数据处理（必须先做）

```bash
# 二选一：CPU 线性质量优先
python convert_data_cpu.py

# 或：GPU 最近邻速度优先
python convert_data_gpu.py

# 划分数据
python generate_splits.py --mode control_rolling --num_folds 4 --val_ratio 0.2 --active_fold 0

# 算归一化
python compute_stats.py
```

说明：每次重跑 `convert_data.py` 后，都建议重新执行 `generate_splits.py` 与 `compute_stats.py`，以保证新通道与归一化参数一致。

补充说明：

- `generate_splits.py --mode control_rolling`：按 `AC_Vel_Set/Fan_Vel_Set/AC_Temp_Set` 的控制状态切换点对齐切分，避免在同一控制段中间硬切。
- `--active_fold` 会把对应折自动写回默认 `train_files.txt / val_files.txt`，训练脚本无需改动。
- 工况切分报告见：`data_pkl/splits/control_splits_report.csv`。
- 控制切换节点见：`data_pkl/splits/control_change_points.csv`。
- `control_rolling` 产生的有效折数可能小于 `--num_folds`（当控制段较少或约束较强时），请以 `control_splits_report.csv` 为准。

### 3.3 训练

```bash
python train.py
```

常用训练参数：

```bash
# 从最新 checkpoint 继续训练（用于跨折经验累积）
python train.py --resume_latest

# 指定继续训练轮数（覆盖 config.py 的 num_epochs）
python train.py --resume_latest --epochs 10

# 指定 checkpoint 恢复
python train.py --resume checkpoints/best_model_epochXX_vallossXXXX.pt
```

建议使用“经验积累”方案：**control_rolling + 跨折续训**：

```bash
# fold0: 冷启动
python generate_splits.py --mode control_rolling --num_folds 4 --val_ratio 0.2 --active_fold 0
python compute_stats.py
python train.py
python predict.py
python visualize_results.py
python plot_training_curve.py

# fold1 及后续有效折: 在上一折最优权重上继续学习（累积不同工况经验）
python generate_splits.py --mode control_rolling --num_folds 4 --val_ratio 0.2 --active_fold 1
python compute_stats.py
python train.py --resume_latest
python predict.py
python visualize_results.py
python plot_training_curve.py

# 若 report 中还有 fold_2 / fold_3，按同样模式继续

# 全部折完成后，汇总温度 MSE 曲线
python plot_fold_summary.py
```

说明：该流程会在时间先后与控制工况切换点对齐前提下，让模型逐折吸收新工况并持续累积经验。

#### 3.3.1 当前工况训练计划表（基于 `control_splits_report.csv`）

> 下表基于当前数据自动生成的有效折（当前为 2 折）。若重跑切分，表中步号范围会变化，请以最新 `data_pkl/splits/control_splits_report.csv` 为准。

| Fold | 训练步号范围 | 验证步号范围 | 验证主工况 | 建议训练方式 | 建议 Epoch |
|---|---|---|---|---|---|
| 0 | 2 ~ 200 | 202 ~ 400 | `AC_Vel=5.0/Fan_Vel=5.0` 与 `AC_Vel=3.0/Fan_Vel=5.0` | 冷启动：`python train.py` | 20（默认） |
| 1 | 2 ~ 400 | 402 ~ 600 | `AC_Vel=5.0/Fan_Vel=5.0` 与 `AC_Vel=3.0/Fan_Vel=5.0` | 续训：`python train.py --resume_latest` | 8~15 |

执行模板（推荐）：

```bash
# fold0
python generate_splits.py --mode control_rolling --num_folds 4 --val_ratio 0.2 --active_fold 0
python compute_stats.py
python train.py
python predict.py
python visualize_results.py
python plot_training_curve.py

# fold1（经验累积）
python generate_splits.py --mode control_rolling --num_folds 4 --val_ratio 0.2 --active_fold 1
python compute_stats.py
python train.py --resume_latest --epochs 10
python predict.py
python visualize_results.py
python plot_training_curve.py

# 多折汇总
python plot_fold_summary.py
```

补充建议：

- 若 fold1 出现明显过拟合（`val_loss` 回升），把续训轮数从 10 下调到 6~8。
- 若 fold1 的 `T_MSE` 仍下降明显，可把续训轮数提高到 12~15。
- 每折训练后记录当前 best checkpoint 文件名，便于回溯“经验累积链条”。

产物：

- 最优权重：`checkpoints/best_model_epoch*_valloss*.pt`
- Loss 记录：`vis_results/train_val_loss.csv`（当前折）
- 每折 Loss 记录：`vis_results/folds/fold_{k}/train_val_loss.csv`
- 每折训练曲线：`vis_results/folds/fold_{k}/train_val_loss.png`
- 每折对比图：`vis_results/folds/fold_{k}/prediction_comparison.png`
- 每折指标：`vis_results/folds/fold_{k}/metrics.txt`（含 `T_MSE`）
- 温度 MSE 汇总：`vis_results/fold_t_mse_summary.csv`
- 温度 MSE 汇总曲线：`vis_results/fold_t_mse_summary.png`

### 3.4 推理与仿真

基础推理：

```bash
python predict.py
python predict.py --ac_temp 26.5
```

专家仿真（自回归）：

```bash
python expert_simulation.py
```

一键可视化（避免手动逐条执行）：

```bash
python run/3_predict_vis.py
```

该脚本会依次执行：`predict.py -> visualize_results.py -> plot_training_curve.py -> plot_fold_summary.py`。

说明：`expert_simulation.py` 已按最新 6 通道输出改造，自回归时会更新 `T/U/V/W/K/NUT` 全部字段，并保留温度边界覆盖能力。

---

## 4. 维护指南（长期使用必看）

### 4.1 每次新增数据后要做什么

1) 放入新 Fluent 导出文件到 `fluent_output_2/`
2) 重跑：`convert_data.py -> generate_splits.py -> compute_stats.py`
3) 再开始训练

### 4.2 配置维护建议

- 统一在 `config.py` 修改参数，不要在脚本中写死
- 改了通道定义后，必须同步检查：
  - `config.py`（`dynamic_channels/static_channels/in_channels_per_step/out_channels_per_step`）
  - `data/dataset.py`（通道拼接顺序）
  - `models/unet3d.py`（输出维度）
  - `train.py`（loss reshape/索引）
  - `convert_data*.py`（离线特征构造逻辑）

### 4.3 Checkpoint 管理

- 训练会持续生成最优模型快照
- 建议周期性清理过旧模型，保留 `val_loss` 最佳若干个
- 如需复现实验，请保存当次 `config.py` 与 `normalization_stats.json`

---

## 5. 调试指南（常见问题定位）

### 5.1 维度不匹配（最常见）

现象：训练报错 `Expected channel dim = pred_steps*6 ...`

排查顺序：

1) `config.py` 中 `in_channels_per_step/out_channels_per_step/pred_steps`
2) `data/dataset.py` 中每步通道数量是否恰好 20
3) `models/unet3d.py` 的 `OutConv(base_channels, pred_steps * 6)`

### 5.2 CUDA 不可用

现象：日志提示未检测到 CUDA

排查：

- 当前解释器是否是 `labtemp-gpu`
- `torch.cuda.is_available()` 是否为 True
- 驱动与 PyTorch CUDA 版本是否匹配

### 5.3 推理结果异常（全零/爆炸）

建议检查：

- `data_pkl/normalization_stats.json` 是否存在、字段是否齐全
- `compute_stats.py` 是否覆盖了当前使用字段
- 输入数据是否含大量 NaN（`convert_data.py` 中已做 nearest 回填）

### 5.4 快速验证工具

- `check_values.py`：检查 `prediction_result.npz` 中范围与形状
- `plot_training_curve.py`：绘制当前活动折的训练/验证 MSE 曲线
- `plot_fold_summary.py`：绘制多折温度 `T_MSE` 汇总曲线
- `visualize_results.py`：可视化预测/误差分布

指标口径说明：

- `metrics.txt` 中 `MSE/RMSE/MAE` 等价于温度通道指标（`T_MSE/T_RMSE/T_MAE`）。
- 同时提供 `U/V/W/K/NUT` 分通道指标，便于定位是哪一类物理量在拖高误差。

---

## 6. 文件清单：哪些必须，哪些可选

### 6.1 训练与推理“必须文件”

- `config.py`：唯一配置源
- `data/dataset.py`：数据加载、归一化、通道组织、`Global_Avg_T_Channel` 计算
- `data/__init__.py`：导出 `create_dataloaders`
- `models/unet3d.py`：UNet + SE3D 主模型
- `train.py`：训练入口
- `predict.py`：批量推理入口
- `convert_data.py`：Fluent 原始数据转网格化 PKL
- `generate_splits.py`：生成训练/验证文件列表
- `compute_stats.py`：计算归一化统计参数

### 6.2 功能增强/实验文件（可选）

- `python expert_simulation.py`：带 AC 控制的自回归仿真
- `python visualize_results.py`：结果可视化
- `python plot_training_curve.py`：训练曲线绘图
- `check_values.py`：结果数据体检
- `convert_results_to_csv.py`：结果导出 CSV
- `plot_fold_summary.py`：多折温度 MSE 汇总曲线

### 6.3 便捷启动脚本（可选）

- `run/1_data_process.py`
- `run/2_train.py`
- `run/3_predict_vis.py`

这些脚本只是封装调用主脚本，不是核心逻辑。

### 6.4 当前可视为“历史/非关键”文件

- `models/__init__.py` 内自带旧版 `CONFIG`（与根目录 `config.py` 不一致）
  - 保留仅为兼容导出，不建议作为配置来源
- `convert_ascii_to_excel.py`
  - 仅在你的原始输入不是 `FFF*` 可直接读格式时才需要
- `merge_data.py` 等数据整理脚本
  - 非当前主训练链路必需

---

## 7. 推荐标准工作流（团队协作）

1) 统一修改 `config.py`
2) 先跑小样本验证（少量文件 + 少 epoch）
3) 确认维度、loss、可视化正常后再全量训练
4) 每次提交实验结果时同时记录：
- 配置快照
- 最优 checkpoint 名称
- train/val 曲线
- 推理样例图

---

## 8. 版本说明（当前）

本 README 对应以下已落地更新：

- 6 通道联合预测（`T/U/V/W/K/NUT`）
- 加权多通道损失（`T:0.7, 其余:0.3`）
- `Q_SRC` 与 `Cell_Zone_Mask` 数据链路
- `Global_Avg_T_Channel` 在线构造
- UDF 控制特征离线复刻（`Current_Time_Norm/Fluid_Avg_T/Solid_Avg_T/AC_Vel_Set/Fan_Vel_Set/...`）
- UNet 引入 `SE3D`（Down 后 + Bottleneck）
- 专家仿真支持 6 通道自回归更新

如后续再改通道定义，请优先更新本 README 的“1.1 输入/输出定义”和“6. 文件清单”。

---

## 9. 科研课堂报告撰写参考

本节用于支持科研课堂中的“课题总结”“个人总结报告”“阶段汇报 PPT”等文字材料撰写。内容尽量与当前项目代码、数据流、训练流程和已有实验结果保持一致，避免在正式提交材料时出现“报告写法”和“项目实际实现”脱节的问题。

### 9.1 课题总结建议写法

若需撰写正式课题总结，建议采用以下结构：

#### 9.1.1 课题背景及意义

可围绕以下几点展开：

- 实验室温度场、速度场和热源分布耦合明显，传统 CFD 仿真精度高但计算开销大。
- 在空调调参、送风组织优化、实验室环境控制和数字孪生场景中，需要更快的三维场预测方法。
- 本项目尝试用深度学习代理模型替代部分全量瞬态仿真，以毫秒级推理支持快速评估。

建议写入的项目事实：

- 当前任务不是单纯预测温度，而是进行 6 物理场联合预测：`T/U/V/W/K/NUT`。
- 当前模型引入了 `SE3D` 注意力模块，用于增强对全局空间耦合关系的建模能力。
- 当前输入包含历史动态场、几何信息、边界掩码和 UDF 控制特征，属于“物理信息增强”的数据驱动建模思路。

#### 9.1.2 研究目的

建议写成“总体目标 + 具体目标”的形式。

总体目标可写为：

- 构建实验室三维温度场智能代理模型，实现对复杂通风与空调工况下室内热流场的快速预测。

具体目标建议包括：

- 建立从 Fluent 原始导出结果到规则三维训练样本的完整预处理流程。
- 设计适合三维场预测的神经网络模型，实现温度与流场的多通道联合输出。
- 将空调速度、风机速度、空调温度等控制特征纳入输入，提高工况泛化能力。
- 实现训练、验证、推理、可视化和自回归仿真的一体化流程。

#### 9.1.3 研究方案

建议按照“数据处理 -> 特征设计 -> 模型设计 -> 损失函数 -> 实验组织”来写。

可直接引用的项目要点如下：

- 网格尺寸：`32 x 64 x 24`。
- 时间组织方式：Time-as-Channel。
- 输入：`input_steps=2`，每步 20 通道。
- 输出：`pred_steps=1`，每步 6 通道。
- 动态输入包括：`T/U/V/W/K/NUT/Q_SRC/Global_Avg_T_Channel`。
- 静态与控制输入包括：`Wall_Dist/Inlet_Mask/Z_Coord/Time_Step/Cell_Zone_Mask/Current_Time_Norm/Fluid_Avg_T/Solid_Avg_T/AC_Vel_Set/Fan_Vel_Set/AC_Temp_Set/Fan_Temp_Set`。
- 主模型为 3D UNet，并在编码阶段与瓶颈层加入 `SE3D`。
- 训练损失为加权多通道 MSE：温度损失权重 0.7，其余流场量总权重 0.3，并叠加平滑正则。
- 数据切分采用 `control_rolling` 策略，按控制状态切换点对齐，避免同一控制段被不合理切断。

#### 9.1.4 主要研究结果

如果需要在报告中引用当前项目结果，可优先使用以下指标口径：

- `metrics.txt` 中的 `MSE/RMSE/MAE` 默认对应温度通道指标。
- 同时可引用 `U/V/W/K/NUT` 的分通道误差，说明模型具备多物理量联合建模能力。

当前已可直接写入报告的结果示例如下：

- `fold_0`：`T_MSE = 5.242461`，`T_RMSE = 2.289642`，`T_MAE = 1.631136`
- `fold_1`：`T_MSE = 13.725123`，`T_RMSE = 3.704743`，`T_MAE = 2.300975`

建议报告中采用“阶段性成果”表述：

- 模型已能够学习实验室三维温度场的主要分布规律。
- 模型在部分工况下达到较好预测精度，但不同验证折间仍存在一定误差波动。
- 结果表明深度学习代理建模路线可行，但跨工况泛化能力仍需继续提升。

#### 9.1.5 讨论与结论

推荐从以下三个角度展开：

- 优点：数据链路完整、输入特征丰富、模型结构稳定、具备多物理量联合预测能力。
- 不足：样本规模有限、工况覆盖仍不充分、当前主要为短步预测、长期滚动预测误差累积尚需研究。
- 结论：项目已完成从数据处理到预测分析的闭环实现，具备继续向控制优化、长时预测和数字孪生方向扩展的基础。

### 9.2 个人总结报告建议写法

若需撰写“个人总结报告”，推荐采用以下结构：

#### 9.2.1 在课题中承担的工作

建议不要只写“参与训练”或“参与调试”，而应从完整链路中选择自己真正参与的部分展开，例如：

- 原始 CFD 数据字段梳理与训练样本理解。
- 数据预处理流程的学习、测试和结果核对。
- 模型输入输出关系整理，理解 20 通道输入与 6 通道输出含义。
- 训练过程跟踪，包括观察 train loss、val loss 与 checkpoint 变化。
- 推理、可视化、指标整理和实验记录归档。
- README、汇报材料、结果总结等文档工作。

写作时建议体现两点：

- 自己做了什么。
- 自己为什么这样做，以及这些工作对项目推进有什么价值。

#### 9.2.2 对研究方案设计、实验结果的评价

可从下面几个方向展开：

- 方案设计上，项目并非“纯黑箱温度回归”，而是融合了物理量、几何信息和控制信息，思路较完整。
- 模型结构上，3D UNet 适合体数据任务，`SE3D` 进一步增强了全局空间特征表达。
- 实验结果上，已能较好预测主要温度分布趋势，但不同工况下仍有误差波动。
- 工程价值上，当前模型更适合作为高成本 CFD 的快速代理与预筛工具，而非直接完全替代高精度仿真。

#### 9.2.3 个人深入思考的问题

个人总结中建议体现“反思深度”，可参考以下方向：

- 数据驱动模型究竟学到的是物理规律还是统计相关性。
- 当模型效果不理想时，问题可能出在数据、特征、划分方式还是网络本身。
- 评价一个场预测模型时，是否只看全局 MSE 就足够，是否还应关注热点区域和关键工位的局部误差。
- 代理模型与传统 CFD 的关系应理解为替代、补充还是加速协同。

#### 9.2.4 对科研课堂的心得体会与建议

此部分建议结合“科研训练而非单次作业”的认识来写，例如：

- 对科研过程的认识从“重结果”转向“重问题、重过程、重复现”。
- 意识到数据处理、实验记录和结果解释与模型本身同样重要。
- 建议课程增加前期研究方法训练、阶段性汇报反馈和跨组交流机制。

### 9.3 报告写作时建议引用的文件产物

建议在写总结、答辩稿或阶段汇报时，优先以以下文件为依据：

- `config.py`：参数与通道定义的唯一配置源。
- `data/dataset.py`：输入特征构造、归一化和通道拼接逻辑。
- `models/unet3d.py`：模型结构与 `SE3D` 加入位置。
- `train.py`：损失函数、训练流程、best checkpoint 保存机制。
- `predict.py`：推理流程、反归一化与结果保存方式。
- `vis_results/metrics.txt`：当前活动折指标结果。
- `vis_results/folds/fold_0/metrics.txt` 与 `vis_results/folds/fold_1/metrics.txt`：折间指标对比。
- `vis_results/train_val_loss.csv` 与各折 `train_val_loss.csv`：训练过程证据。
- `data_pkl/splits/control_splits_report.csv`：训练集/验证集划分依据。
- `data_pkl/splits/control_change_points.csv`：控制段切换依据。

### 9.4 使用建议

若后续要继续提交“课题总结”“个人总结报告”“中期检查材料”或“答辩 PPT”，建议先同步检查以下内容是否和代码一致：

1) 当前通道数是否仍为输入 20、输出 6。
2) 当前模型是否仍为 `UNet3DTimeAsChannel + SE3D`。
3) 当前最优结果是否来自最新折次。
4) 当前图表和指标是否与 `vis_results/` 下文件一致。

这样可以避免常见问题：报告写的是旧版设定，而实验跑的是新版配置。

