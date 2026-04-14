import os
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from env_config import configure_proxy, get_api_key


# 生成图数据时使用的业务背景设定。
DOMAIN_THEME = "Financial Transaction Network"



# 图结构骨架生成。

def create_graph_skeleton(num_nodes=5):
    """
    构造星型图，并根据节点度数标注节点角色。
    """
    G = nx.star_graph(num_nodes - 1)

    # 根据节点度数分配角色标签。
    node_roles = {}
    for node, degree in G.degree():
        if degree > 1:
            node_roles[node] = f"Central Hub (Degree: {degree})"
        else:
            node_roles[node] = f"Edge Node (Degree: {degree})"

    # 转换为 PyG 使用的 edge_index 格式。
    edges = list(G.edges())
    # 无向图需要补全双向边。
    edges.extend([(v, u) for u, v in edges])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, node_roles



# 使用 Gemini 生成节点语义。

def generate_node_semantics(client, node_roles, is_ood=False):
    """
    根据拓扑角色和 OOD 状态生成节点文本描述。
    """
    # 根据节点角色与目标分布状态构造提示词。
    condition = "OUT-OF-DISTRIBUTION" if is_ood else "IN-DISTRIBUTION"
    ood_instruction = ""
    if is_ood:
        ood_instruction = (
            "Inject a severe semantic anomaly only into 'Node 1'. "
            "Keep all other nodes, including Node 0, 2, 3, and 4, "
            "consistent with the in-distribution setting. "
            "Node 1 is the only anomalous node."
        )
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
        model='gemini-3.1-flash-lite-preview',
        contents=prompt,
        config=config,
    )
    print("[Generation] Response received successfully.")

    # 解析返回的 JSON 结果。
    try:
        text_data = json.loads(response.text)
        return text_data
    except json.JSONDecodeError:
        raise ValueError("Gemini did not return valid JSON. Please retry.")


# 文本编码与 PyG 图数据构建。

def build_pyg_data(edge_index, text_data, encoder, y_label):
    """
    将生成的文本编码为向量，并封装成 PyG 的 Data 对象。
    """
    # 按节点顺序读取文本描述：0, 1, 2, ...
    num_nodes = edge_index.max().item() + 1
    texts = [text_data[str(i)] for i in range(num_nodes)]

    # 将文本编码为节点特征向量。
    print(f"[Encoding] Encoding semantic descriptions for {num_nodes} nodes with SentenceTransformer...")
    x = encoder.encode(texts, convert_to_tensor=True)

    # 构建图对象。
    y = torch.tensor([y_label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.raw_texts = texts

    return data


# 独立运行时的数据生成入口。

if __name__ == "__main__":
    print("Starting graph-level OOD data generation pipeline...")
    proxy_port = configure_proxy()
    if proxy_port:
        print(f"Proxy configured on port: {proxy_port}")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # 初始化客户端与文本编码器。
    gemini_client = genai.Client(api_key=get_api_key())
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 构建图结构骨架。
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)
    print("\n--- Graph topology skeleton generated ---")
    for k, v in node_roles.items():
        print(f"Node {k}: {v}")

    # 生成一张 ID 图。
    print("\nGenerating an in-distribution graph...")
    id_texts = generate_node_semantics(gemini_client, node_roles, is_ood=False)
    id_data = build_pyg_data(edge_index, id_texts, text_encoder, y_label=0)
    print(f"ID graph constructed. Feature shape: {id_data.x.shape}")

    # 生成一张 OOD 图。
    print("\nGenerating an out-of-distribution graph...")
    ood_texts = generate_node_semantics(gemini_client, node_roles, is_ood=True)
    ood_data = build_pyg_data(edge_index, ood_texts, text_encoder, y_label=1)
    print(f"OOD graph constructed. Feature shape: {ood_data.x.shape}")

    # 检查 OOD 图中各节点的文本描述。
    print("\nInspecting node semantics in the OOD graph:")
    for i in range(5):
        prefix = "[Perturbed edge node]" if i > 0 else "[Reference central node]"
        print(f"Node {i} {prefix}: {ood_data.raw_texts[i]}")