# Anomaly-Aware Readout for Graph OOD Detection

## 0. 文档说明

本文档用于说明项目的研究目标、核心方法、代码结构与实验注意事项。

如果后续需要继续扩展代码，请尽量遵循本文档中给出的建模假设、损失设计和工程结构，尤其不要随意删除当前的联合优化逻辑。

## 1. 研究动机

在真实图网络中，例如金融交易网络或引用网络，异常节点可能在局部拓扑结构上看起来正常，但其节点属性或语义特征与周围上下文不一致。本文将这种现象视为 `topology-semantic mismatch`。

传统基于图卷积的 OOD 检测方法通常依赖邻域聚合来形成节点表示。在这种设定下，异常节点的局部异常信号可能被大量正常邻居稀释，进而影响节点级 OOD 评分表现。

## 2. 方法概述

当前的 `NodeAnomalyAwareModel` 基于联合优化的拓扑-语义对齐机制。

### 2.1 拓扑-语义双空间表示

对于图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中的节点 $i$，设其输入特征为 $\mathbf{x}_i$。模型分别构建拓扑表示 $\mathbf{h}_{topo}^{(i)}$ 与语义表示 $\mathbf{z}_{sem}^{(i)}$，并将两者投影到共享的对齐空间：

$$
\mathbf{h}_{topo}^{(i)} = \text{ReLU}(\text{GCNConv}(\mathbf{x}_i, \mathcal{E}))
$$

$$
\mathbf{z}_{topo}^{(i)} = \text{MLP}_{topo}(\mathbf{h}_{topo}^{(i)})
$$

$$
\mathbf{z}_{sem}^{(i)} = \text{MLP}_{sem}(\mathbf{x}_i)
$$

### 2.2 异常评分

在对齐空间中，正常节点的拓扑表示与语义表示应当保持一致，而 OOD 节点通常会表现出更大的表示偏差。因此，当前实现使用欧氏距离作为节点异常分数：

$$
s_i = || \mathbf{z}_{topo}^{(i)} - \mathbf{z}_{sem}^{(i)} ||_2
$$

### 2.3 联合优化目标

为避免表示塌陷并保持分类能力，训练目标由分类损失和对齐损失共同组成：

$$
\mathcal{L} = \mathcal{L}_{cls} + \alpha \cdot \mathcal{L}_{align}
$$

其中，$\mathcal{L}_{cls}$ 为基于有标签节点的交叉熵损失，$\mathcal{L}_{align}$ 为在全图范围内计算的 MSE 对齐损失：

$$
\mathcal{L}_{align} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} || \mathbf{z}_{topo}^{(i)} - \mathbf{z}_{sem}^{(i)} ||_2^2
$$

## 3. 代码结构

当前项目主要包含以下模块：

```text
Graph-level OOD Detection/
├── data/                  # PyG 下载的数据集，例如 Cora
├── logs/                  # 实验日志
├── models/                # 节点级模型定义
│   ├── __init__.py
│   ├── baselines.py       # StandardNodeGCN baseline
│   └── anomaly_aware.py   # NodeAnomalyAwareModel
├── utils/                 # 工具函数
│   ├── __init__.py
│   └── metrics.py         # AUROC / FPR95 计算
└── main.py                # 节点级实验主入口
```

## 4. 当前实验设定

- 数据集：`Cora`，包含 `2708` 个节点和 `1433` 维特征。
- OOD 构造方式：从测试集选取一部分节点，打乱其特征表示，同时保持图结构不变。
- 基线模型：`Standard GCN + Energy`
- 目标模型：`NodeAnomalyAwareModel`

当前记录的一个节点级实验结果如下：

- `Standard GCN + Energy`: `AUROC ~65.17%`, `FPR95 ~87.25%`
- `Anomaly-Aware`: `AUROC 76.43%`, `FPR95 78.75%`

这些数值应视为当前实现和特定实验设置下的参考结果，而不是固定结论。

## 5. 实现注意事项

### 5.1 避免表示塌陷

如果训练时只保留对齐损失

$$
\mathcal{L}_{align} = \text{MSE}(\mathbf{z}_{topo}, \mathbf{z}_{sem})
$$

而缺少分类约束，模型可能会退化到平凡解，导致节点分数缺乏判别性。

因此，当前实现中应保留分类头 `self.classifier = nn.Linear(align_dim, num_classes)`，并联合使用 $\mathcal{L}_{cls}$ 与 $\mathcal{L}_{align}$。

### 5.2 不要只在训练节点上做对齐

如果仅在训练掩码范围内计算对齐损失，例如：

```python
loss_align = MSELoss()(z_topo[data.train_mask], z_sem[data.train_mask])
```

那么对齐约束会局限在少量有标签节点上，容易导致语义投影层过拟合，进而影响测试阶段的异常评分稳定性。

当前实现采用全图对齐：

```python
loss_align = nn.MSELoss()(z_topo, z_sem)
```

## 6. 使用建议

如果后续要扩展本项目，例如替换数据集、增加可视化或调整网络结构，建议优先检查以下几点：

- 是否仍然保留了分类损失与对齐损失的联合训练方式。
- 异常分数的定义是否仍然与拓扑-语义表示偏差一致。
- 新增改动是否影响了当前节点级与图级实验的可复现性。

如果需要把这份文档作为后续协作上下文，可以直接说明：

> 请先阅读 `Project_Context.md`，再基于现有方法实现或修改相关功能。
