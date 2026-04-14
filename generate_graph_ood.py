import os
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# =====================================================================
# [导师配置区]：请务必修改这三个核心参数！
# =====================================================================
PROXY_PORT = "9674"  # 替换为你的真实代理端口 (如 7890, 10808)
API_KEY = "AIzaSyBwB5Vft5rqw8l87bl0e3KmUteacVqsY_A"
DOMAIN_THEME = "Financial Transaction Network"  # 图的宏观背景，可改为"分子化学"或"计算机网络"

# 强制网络接管 (防止网络黑洞)
os.environ['http_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['https_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# =====================================================================
# 模块 1: 角色感知的拓扑骨架生成 (Skeleton Generator)
# =====================================================================
def create_graph_skeleton(num_nodes=5):
    """
    生成一个具有中心-边缘结构的星型/无标度拓扑图。
    这里为了演示清晰，生成一个经典的 5 节点星型网络：
    Node 0 是绝对的中心 (Degree=4)，其余是边缘节点 (Degree=1)
    """
    G = nx.star_graph(num_nodes - 1)

    # 提取节点角色 (按度数)
    node_roles = {}
    for node, degree in G.degree():
        if degree > 1:
            node_roles[node] = f"Central Hub (Degree: {degree})"
        else:
            node_roles[node] = f"Edge Node (Degree: {degree})"

    # 转换为 PyG 的 edge_index (2 x E 张量)
    edges = list(G.edges())
    # 无向图需要双向边
    edges.extend([(v, u) for u, v in edges])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, node_roles


# =====================================================================
# 模块 2: 基于 Gemini 的拓扑-语义解耦生成 (Semantic Injector)
# =====================================================================
def generate_node_semantics(client, node_roles, is_ood=False):
    """
    核心逻辑：角色驱动的语义注入。
    如果是 ID 图：所有节点都生成正常的背景文本。
    如果是 Hard OOD 图：中心节点正常，但悄悄篡改边缘节点的语义！
    """
    # 构建严格的系统提示词
    condition = "OUT-OF-DISTRIBUTION (Hard OOD)" if is_ood else "IN-DISTRIBUTION (Normal ID)"
    ood_instruction = ""
    if is_ood:
        ood_instruction = "CRITICAL: Secretly inject severely conflicting anomalies (e.g., money laundering, crypto-scam, illegal dark web routing) ONLY into 'Node 1'. Keep ALL other nodes (including Node 0, 2, 3, 4) absolutely normal. Node 1 is the sole anomaly."
    prompt = f"""
    You are an expert Graph Data Synthesizer.
    Task: Generate node descriptions for a {DOMAIN_THEME}.
    Condition: This graph must be {condition}.

    Node Roles provided below:
    {json.dumps(node_roles, indent=2)}

    Rules:
    1. STRICT TOPOLOGY: You must generate EXACTLY one text description for each node ID provided.
    2. ROLE-DRIVEN: The 'Central Hub' should describe macro-level, core structural concepts. The 'Edge Nodes' should describe specific, granular behaviors.
    3. {ood_instruction}
    4. OUTPUT FORMAT: Respond ONLY with a valid JSON dictionary mapping node IDs (as strings) to their text descriptions. No markdown blocks, no other text.
    """

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.7,
    )

    print(f"\n[Generation] Sending request to Gemini 3.1 Flash ({'OOD' if is_ood else 'ID'} sample)...")
    response = client.models.generate_content(
        model='gemini-3.1-flash-lite-preview',  # ⚡ 启用 3.1 世代的极速轻量引擎
        contents=prompt,
        config=config,
    )
    print("[Generation] Response received successfully.")

    # 解析 JSON
    try:
        text_data = json.loads(response.text)
        return text_data
    except json.JSONDecodeError:
        raise ValueError("Gemini did not return valid JSON. Please retry.")


# =====================================================================
# 模块 3: 文本编码与 PyG 图构建 (Tensor Encoder)
# =====================================================================
def build_pyg_data(edge_index, text_data, encoder, y_label):
    """
    将大模型生成的文本转化为特征张量，并打包为 PyG Data 对象
    """
    # 按照节点顺序 0, 1, 2... 提取文本
    num_nodes = edge_index.max().item() + 1
    texts = [text_data[str(i)] for i in range(num_nodes)]

    # 文本转张量
    print(f"[Encoding] Encoding semantic descriptions for {num_nodes} nodes with SentenceTransformer...")
    x = encoder.encode(texts, convert_to_tensor=True)  # [num_nodes, embed_dim]

    # 构建 Data 对象
    y = torch.tensor([y_label], dtype=torch.long)  # 0 for ID, 1 for OOD
    data = Data(x=x, edge_index=edge_index, y=y)
    data.raw_texts = texts  # 顺便保存原始文本方便以后可视化

    return data


# =====================================================================
# 主流程：点火运行
# =====================================================================
if __name__ == "__main__":
    print(">>> Starting graph-level OOD data generation pipeline...")

    # 1. 初始化客户端和模型
    gemini_client = genai.Client(api_key=API_KEY)
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 2. 生成物理骨架 (固定)
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)
    print("\n--- Graph topology skeleton generated ---")
    for k, v in node_roles.items():
        print(f"Node {k}: {v}")

    # 3. 合成 ID 图 (标签 0)
    print("\n================ [Synthesizing In-Distribution (ID) Graph] ================")
    id_texts = generate_node_semantics(gemini_client, node_roles, is_ood=False)
    id_data = build_pyg_data(edge_index, id_texts, text_encoder, y_label=0)
    print(f"ID graph constructed. Feature shape: {id_data.x.shape}")

    # 4. 合成 Hard OOD 图 (标签 1)
    print("\n================ [Synthesizing Out-of-Distribution (Hard OOD) Graph] ================")
    ood_texts = generate_node_semantics(gemini_client, node_roles, is_ood=True)
    ood_data = build_pyg_data(edge_index, ood_texts, text_encoder, y_label=1)
    print(f"OOD graph constructed. Feature shape: {ood_data.x.shape}")

    # 5. 学术验收：打印 OOD 图的篡改文本
    print("\n[Inspection] Node semantics in the Hard OOD graph:")
    for i in range(5):
        prefix = "[Perturbed Edge Node]" if i > 0 else "[Normal Central Node]"
        print(f"Node {i} {prefix}: {ood_data.raw_texts[i]}")