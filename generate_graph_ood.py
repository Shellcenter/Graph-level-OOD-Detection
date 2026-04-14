import os
import json
import networkx as nx
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from env_config import configure_proxy, get_api_key


# Business-domain setting for generated graphs.
DOMAIN_THEME = "Financial Transaction Network"



# Graph skeleton generation.

def create_graph_skeleton(num_nodes=5):
    """
    Build a star-shaped graph and annotate node roles by degree.
    """
    G = nx.star_graph(num_nodes - 1)

    # Assign node roles from degree statistics.
    node_roles = {}
    for node, degree in G.degree():
        if degree > 1:
            node_roles[node] = f"Central Hub (Degree: {degree})"
        else:
            node_roles[node] = f"Edge Node (Degree: {degree})"

    # Convert to the PyG edge_index format.
    edges = list(G.edges())
    # Duplicate edges for the undirected graph representation.
    edges.extend([(v, u) for u, v in edges])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return edge_index, node_roles



# Semantic generation with Gemini.

def generate_node_semantics(client, node_roles, is_ood=False):
    """
    Generate node descriptions conditioned on topology and OOD status.
    """
    # Build the prompt from node roles and the requested condition.
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

    # Parse the returned JSON payload.
    try:
        text_data = json.loads(response.text)
        return text_data
    except json.JSONDecodeError:
        raise ValueError("Gemini did not return valid JSON. Please retry.")


# Text encoding and PyG data construction.

def build_pyg_data(edge_index, text_data, encoder, y_label):
    """
    Encode generated text and package it as a PyG Data object.
    """
    # Read descriptions in node order: 0, 1, 2, ...
    num_nodes = edge_index.max().item() + 1
    texts = [text_data[str(i)] for i in range(num_nodes)]

    # Encode text into node feature vectors.
    print(f"[Encoding] Encoding semantic descriptions for {num_nodes} nodes with SentenceTransformer...")
    x = encoder.encode(texts, convert_to_tensor=True)

    # Build the graph object.
    y = torch.tensor([y_label], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.raw_texts = texts

    return data


# Standalone generation entry point.

if __name__ == "__main__":
    print("Starting graph-level OOD data generation pipeline...")
    proxy_port = configure_proxy()
    if proxy_port:
        print(f"Proxy configured on port: {proxy_port}")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # Initialize the client and encoder.
    gemini_client = genai.Client(api_key=get_api_key())
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Build the graph skeleton.
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)
    print("\n--- Graph topology skeleton generated ---")
    for k, v in node_roles.items():
        print(f"Node {k}: {v}")

    # Generate one ID graph.
    print("\nGenerating an in-distribution graph...")
    id_texts = generate_node_semantics(gemini_client, node_roles, is_ood=False)
    id_data = build_pyg_data(edge_index, id_texts, text_encoder, y_label=0)
    print(f"ID graph constructed. Feature shape: {id_data.x.shape}")

    # Generate one OOD graph.
    print("\nGenerating an out-of-distribution graph...")
    ood_texts = generate_node_semantics(gemini_client, node_roles, is_ood=True)
    ood_data = build_pyg_data(edge_index, ood_texts, text_encoder, y_label=1)
    print(f"OOD graph constructed. Feature shape: {ood_data.x.shape}")

    # Inspect the generated OOD node descriptions.
    print("\nInspecting node semantics in the OOD graph:")
    for i in range(5):
        prefix = "[Perturbed edge node]" if i > 0 else "[Reference central node]"
        print(f"Node {i} {prefix}: {ood_data.raw_texts[i]}")