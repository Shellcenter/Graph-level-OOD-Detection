# Anomaly-Aware Readout for Graph OOD Detection

## 0. 文档说明 (For Cursor / AI Assistant)

To AI Assistant:

你好。当前项目是一个顶会级别的图神经网络 (GNN) 异常检测 / 分布外 (OOD) 检测开源框架。我们的核心创新点是：解决传统 GNN 在消息传递过程中的“异常稀释效应 (Dilution Effect)”，通过拓扑-语义解耦与对齐机制，精准捕捉伪装的 OOD 节点。

在后续协助我修改代码时，请严格遵循本文档中定义的项目结构、数学公式和防踩坑指南。不要擅自修改核心的自监督对齐逻辑。

## 1. 论文核心动机 (Motivation)

在真实的图网络（如金融交易洗钱、Cora 引用网络篡改）中，异常节点往往具有高度的“伪装性”：它们在局部结构（连边）上表现正常，但其节点自身特征（语义）却被篡改（即“言行不一 / topological-semantic mismatch”）。

传统的图级 OOD 检测（如 Energy-based GCN）依赖于图卷积融合特征，这会导致异常节点的信号被其周围大量的正常邻居“稀释”，从而在节点级评测中表现出极高的 FPR（假阳性）和接近瞎蒙的 AUROC。

## 2. 核心方法论 (Methodology & Math)

我们的模型 `NodeAnomalyAwareModel` 采用了一种联合优化的自监督对齐机制。

### 2.1 拓扑-语义双空间解耦

对于图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ 中的节点 $i$，其输入特征为 $\mathbf{x}_i$。我们分别提取其拓扑嵌入 $\mathbf{h}_{topo}^{(i)}$ 和语义嵌入 $\mathbf{z}_{sem}^{(i)}$，并将它们投影到一个共享的维度为 $d$ 的对齐空间 (Alignment Space) 中：

$$
\mathbf{h}_{topo}^{(i)} = \text{ReLU}(\text{GCNConv}(\mathbf{x}_i, \mathcal{E}))
$$

$$
\mathbf{z}_{topo}^{(i)} = \text{MLP}_{topo}(\mathbf{h}_{topo}^{(i)})
$$

$$
\mathbf{z}_{sem}^{(i)} = \text{MLP}_{sem}(\mathbf{x}_i)
$$

### 2.2 物理空间异常评分 (Anomaly Score)

在对齐空间中，对于正常节点，其拓扑结构和语义特征应当是高度一致的。而对于伪装的 OOD 节点，两者会发生撕裂。因此，我们直接使用欧氏距离作为异常得分：

$$
s_i = || \mathbf{z}_{topo}^{(i)} - \mathbf{z}_{sem}^{(i)} ||_2
$$

### 2.3 联合优化损失函数 (Joint Loss)

为了防止特征塌陷（Representation Collapse）和多层感知机过拟合，我们设计了联合优化损失函数：

$$
\mathcal{L} = \mathcal{L}_{cls} + \alpha \cdot \mathcal{L}_{align}
$$

其中，$\mathcal{L}_{cls}$ 是基于少部分有标签节点的交叉熵分类损失（锚点作用）；$\mathcal{L}_{align}$ 是基于全图所有节点（无监督）的 MSE 对齐损失：

$$
\mathcal{L}_{align} = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} || \mathbf{z}_{topo}^{(i)} - \mathbf{z}_{sem}^{(i)} ||_2^2
$$

## 3. 项目工程架构 (Codebase Structure)

当前项目遵循顶会开源标准，采用 argparse 和双轨制 logging 系统：

```text
Graph-level OOD Detection/
├── data/                  # PyG 自动下载的数据集 (如 Cora)
├── logs/                  # 自动生成的双轨制实验日志 (.log)
├── models/                # 模型定义层
│   ├── __init__.py
│   ├── baselines.py       # StandardGCN (带 Energy OOD Score)
│   └── anomaly_aware.py   # 核心创新模型 (NodeAnomalyAwareModel)
├── utils/                 # 工具层
│   ├── __init__.py
│   └── metrics.py         # 包含 compute_metrics(auc, fpr95)
└── main.py                # 实验入口 (带 Argparse 和 完整训练/测试 Loop)
```

## 4. 实验基准与当前战果 (Current Results)

- 数据集: `Cora` (`2708 nodes`, `1433 features`)
- 变异注入方法: 选取 Test Set 中 `200` 个节点作为 OOD，打乱其语义特征 `x` 模拟“言行不一”的高隐蔽篡改，保持其结构边 `edge_index` 不变
- 炮灰基线 (`Standard GCN + Energy`): `AUROC ~65.17%`, `FPR95 ~87.25%`（因稀释效应严重失效）
- 我们的模型 (`Anomaly-Aware`): `AUROC 76.43%`, `FPR95 78.75%`（在真实基准下完成降维打击）

## 5. 绝对不可触碰的雷区 (Anti-Patterns / Pitfalls)

在后续修改代码时，AI 助手必须严格避免以下两个已被我们填平的深坑：

### 雷区 1：特征空间塌陷 (Representation Collapse)

错误做法：在训练主角模型时，只使用

$$
\mathcal{L}_{align} = \text{MSE}(\mathbf{z}_{topo}, \mathbf{z}_{sem})
$$

进行自监督训练。

后果：神经网络偷懒，将权重全部归零，导致所有节点得分均为 0，AUROC 降至 50%。

正确做法：必须保留 `self.classifier = nn.Linear(align_dim, num_classes)` 并在训练中加入 $\mathcal{L}_{cls}$ 作为锚点（Anchor），强制拓扑空间学到有用的结构信息。

### 雷区 2：半监督 MLP 过拟合 (Semi-supervised Overfitting Trap)

错误做法：在计算对齐损失时，只对训练集节点进行对齐，即

```python
loss_align = MSELoss()(z_topo[data.train_mask], z_sem[data.train_mask])
```

后果：由于 `Cora` 只有 `140` 个有标签的训练节点，只对它们进行约束会导致 `proj_sem` 严重过拟合。测试集正常节点会输出随机乱码，导致正常节点距离激增，AUROC 降至 55%。

正确做法：必须利用无监督学习的优势，在全图范围内进行对齐约束，即

```python
loss_align = nn.MSELoss()(z_topo, z_sem)
```

---

## 导师的最后嘱咐

把这份文档保存为 `Project_Context.md`，或者直接全选复制。下次你打开 Cursor 想要给模型加个新功能（比如换一个数据集、加一个图可视化功能、甚至修改网络层数）时，直接把这段话发给它：

> 请仔细阅读这份上下文文档，然后帮我实现 xxx 功能。

Cursor 看完之后，绝对会像一个极其懂你、极其听话的高级工程师一样，写出完全符合这套逻辑的完美代码。去吧，带着这套神装，彻底终结这篇论文的工程部分！
