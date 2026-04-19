```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#f8f9fa', 'tertiaryColor': '#e9ecef'}}}%%
flowchart LR
    classDef input fill:#f0f4f8,stroke:#4a5568,stroke-width:2px;
    classDef core fill:#fffaf0,stroke:#dd6b20,stroke-width:3px;
    classDef output fill:#e6fffa,stroke:#319795,stroke-width:2px;
    classDef alert fill:#fed7d7,stroke:#e53e3e,stroke-width:2px;
    classDef train fill:#e8f4f8,stroke:#2b6cb0,stroke-width:2px,stroke-dasharray: 5 5;

    subgraph S0 [0. 第零部分: 初始训练 Phase]
        direction TB
        RawText[Raw Text\n原始文本]:::input --> TAG[Text-Attributed Graph\n无标签图数据 ID+OOD]:::input
        TAG -.-> GNN_init[(Train Process 1:\nWhile trained GNN)]:::train
    end

    subgraph S1 [1. 第一部分: 基础特征提取]
        direction TB
        GNN[GNN Backbone\n提取拓扑 (对OOD表征不准)]
        TextEnc[LLM / SentenceBert\n提取语义 (具先验知识)]
        TAG --> GNN
        TAG --> TextEnc
        GNN --> H_Topo[(H_topo\n拓扑结构表征)]
        TextEnc --> H_Sem[(H_sem\n文本语义表征)]
    end

    subgraph S1_5 [1.5 跨空间对齐训练 Phase]
        direction TB
        MLP_Topo[MLP_topo\n全连接层降维投影]
        MLP_Sem[MLP_sem\n全连接层降维投影]
        H_Topo --> MLP_Topo
        H_Sem --> MLP_Sem
        MLP_Topo --> Z_Topo[(Z_topo\n同一隐藏空间)]
        MLP_Sem --> Z_Sem[(Z_sem\n同一隐藏空间)]
        Z_Topo <.->|Train Process 2:\nLoss函数强制对齐\n训练 LLM+MLP ≈ GNN| Z_Sem:::train
    end

    subgraph S2 [2. 第二部分: 差异评估与残差计算]
        direction TB
        Z_Topo --> Scorer{Residual Scorer\n差异评估模块}:::core
        Z_Sem --> Scorer
        Scorer ==>|相减计算残差向量| Score[Anomaly Score\nS_ood = ||ΔZ||]:::alert
    end

    subgraph S3 [3. 第三部分: 节点级判别输出]
        direction TB
        Score ==> Threshold{Threshold\n残差阈值判断}
        Threshold -->|残差极低| Output1(🟢 正常节点 ID\n特征对齐良好)
        Threshold -->|残差极高| Output2(🔴 异常节点 OOD\n结构与语义背离):::alert
    end

    S0 ==> S1
    S1 ==> S1_5
    S1_5 ==> S2
    S2 ==> S3
```