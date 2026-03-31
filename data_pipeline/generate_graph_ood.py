import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data

# =====================================================================
# [配置区] 填入你的 Gemini API 密钥
# =====================================================================
client = genai.Client(api_key="AIzaSyBwB5Vft5rqw8l87bl0e3KmUteacVqsY_A")

print("正在加载文本编码器 (SentenceTransformer)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
print("编码器加载完成！")

# =====================================================================
# [核心提示词] 强制 Gemini 遵守“角色驱动”与“拓扑锁定”
# =====================================================================
SYSTEM_PROMPT = """
You are an expert graph data synthesizer for machine learning research. 
Your task is to generate a realistic text-attributed graph based on a provided topological skeleton and a target 'Hard OOD (Out-of-Distribution) Label'.

Rules for Generation:
1. STRICT TOPOLOGY: You MUST preserve the exact 'num_nodes' and 'edges' from the input skeleton. Do not add or remove nodes/edges.
2. ROLE-DRIVEN SEMANTICS: You must assign text attributes to each node based on its degree (connectivity):
   - Central Nodes (High degree): Generate text representing core, generalized concepts related to the '{ood_label}'.
   - Edge Nodes (Low degree): Generate text representing specific, derived, or granular details related to the central node's concept.
3. OUTPUT FORMAT: You must strictly output valid JSON matching this schema:
   {{
     "nodes": [{{"id": int, "text": "string"}}],
     "edges": [[int, int]]
   }}
"""


# =====================================================================
# [处理管道] 校验与张量化
# =====================================================================
def validate_and_parse_graph(llm_graph, expected_nodes):
    nodes = llm_graph.get('nodes', [])
    edges = llm_graph.get('edges', [])

    if len(nodes) != expected_nodes:
        raise ValueError(f"节点数量不符！期望 {expected_nodes}，实际返回 {len(nodes)}")

    sorted_nodes = sorted(nodes, key=lambda x: x['id'])
    texts = [n['text'] for n in sorted_nodes]
    x = encoder.encode(texts, convert_to_tensor=True)

    valid_edges = []
    for u, v in edges:
        if 0 <= u < expected_nodes and 0 <= v < expected_nodes:
            valid_edges.extend([[u, v], [v, u]])
        else:
            raise ValueError(f"边越界: [{u}, {v}]")

    if not valid_edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()

    print("\n[导师质检] Gemini 为不同度数节点生成的语义文本:")
    for i, txt in enumerate(texts):
        print(f"Node {i}: {txt}")
    print("-" * 50 + "\n")

    return Data(x=x, edge_index=edge_index)


# =====================================================================
# [引擎唤醒区] 带有最高级别 Debug 探针与强制 JSON 约束
# =====================================================================
def generate_single_ood_graph(skeleton_dict, ood_label):
    prompt = f"Target Hard OOD Label: {ood_label}\nInput Skeleton: {json.dumps(skeleton_dict)}"

    try:
        # 强制 API 层面锁定 JSON 输出格式
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(ood_label=ood_label),
            response_mime_type="application/json",  # 物理封锁 Markdown 幻觉
            temperature=0.7,
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',  # 升级为最新一代引擎
            contents=prompt,
            config=config,
        )

        # --- 【最高级别 Debug 探针】无条件拦截原始数据 ---
        raw_text = response.text
        print("\n" + "▼" * 50)
        print("[大模型原始返回数据裸眼审查 - 导师专用]")
        print(raw_text)
        print("▲" * 50 + "\n")

        # 容错清洗
        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text.strip("` \n")
            if cleaned_text.lower().startswith("json"):
                cleaned_text = cleaned_text[4:].strip()

        llm_output = json.loads(cleaned_text)
        return validate_and_parse_graph(llm_output, skeleton_dict['num_nodes'])

    except Exception as e:
        # 精准捕获错误类型
        print(f"\n[致命报错拦截] 错误类型: {type(e).__name__}")
        print(f"错误详情: {e}")
        return None

# =====================================================================
# [点火测试]
# =====================================================================
if __name__ == "__main__":
    mock_skeleton = {
        "num_nodes": 4,
        "edges": [[0, 1], [0, 2], [1, 3]],
        "node_degrees": {"0": 2, "1": 2, "2": 1, "3": 1}
    }

    target_label = "Quantum Cryptography"
    print(">>> 启动 Gemini 原生图级数据生成管道...")

    pyg_data = generate_single_ood_graph(mock_skeleton, target_label)

    if pyg_data:
        print(f">>> 成功！最终生成的 PyG 图数据结构: {pyg_data}")
        print(f"特征矩阵 X 维度: {pyg_data.x.shape}")
        print(f"边表 Edge Index 维度: {pyg_data.edge_index.shape}")