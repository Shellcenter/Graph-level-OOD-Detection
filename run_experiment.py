import torch
import torch.nn as nn
import torch.optim as optim

# 导入核心组件 (已修复你之前的路径问题)
from generate_graph_ood import create_graph_skeleton, generate_node_semantics, build_pyg_data, API_KEY
from train_ood_model import AnomalyAwareModel
from google import genai
from sentence_transformers import SentenceTransformer


def train_model():
    print(">>> 1. 初始化引擎与环境...")
    # 🚀 统一空间1：获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[硬件探针] 当前正在使用的计算设备: {device}")

    client = genai.Client(api_key=API_KEY)
    text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 🚀 统一空间2：将模型强行搬入 GPU！(你刚才极有可能是漏了这里的 .to(device))
    model = AnomalyAwareModel(sem_dim=384, topo_hidden=64, align_dim=32).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    print("\n>>> 2. 准备极简训练集 (1张ID图, 1张OOD图) 作为概念验证...")
    edge_index, node_roles = create_graph_skeleton(num_nodes=5)

    id_texts = generate_node_semantics(client, node_roles, is_ood=False)
    id_data = build_pyg_data(edge_index, id_texts, text_encoder, y_label=0.0)

    ood_texts = generate_node_semantics(client, node_roles, is_ood=True)
    ood_data = build_pyg_data(edge_index, ood_texts, text_encoder, y_label=1.0)

    dataset = [id_data, ood_data]

    print("\n>>> 3. 开始训练 (Training Loop) ...")
    epochs = 30

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for data in dataset:
            # 统一空间3：将数据连同它的标签全部搬入 GPU！
            data = data.to(device)
            optimizer.zero_grad()

            # 统一空间4：索引也要进 GPU！
            batch_index = torch.zeros(data.x.size(0), dtype=torch.long).to(device)

            # 🚀 接收 4 个返回值
            logits, alphas, z_topo, z_sem = model(data.x, data.edge_index, batch_index)

            # 🌟 1. 主干分类损失 (BCE Loss)
            loss_cls = criterion(logits.unsqueeze(0), data.y.float())

            # 🌟 2. 辅助对齐损失 (MSE Loss)
            # 核心机制：我们只要求正常图 (ID) 强制对齐，所以乘以 (1.0 - data.y.float())
            # 这样当 data.y = 1.0 (异常图) 时，(1 - 1) = 0，异常图不参与对齐，保留冲突！
            mse_loss = nn.MSELoss()(z_topo, z_sem)
            loss_align = mse_loss * (1.0 - data.y.float())

            # 🌟 3. 总体损失 = 分类损失 + 权重 * 对齐损失
            lambda_align = 0.5  # 对齐约束的权重，0.5是个不错的初始值
            loss = loss_cls + lambda_align * loss_align

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Loss: {total_loss / len(dataset):.4f}")

    print("\n>>> 4. 训练结束，学术验收！")
    model.eval()
    with torch.no_grad():
        # 🚀 统一空间5：验收阶段也要保持在 GPU！
        ood_data = ood_data.to(device)
        batch_index = torch.zeros(ood_data.x.size(0), dtype=torch.long).to(device)
        # 记得把第 4 步里的模型调用也改成接 4 个值：
        logits, alphas, _, _ = model(ood_data.x, ood_data.edge_index, batch_index)
        final_logits = logits.unsqueeze(0)
        print(f"- 模型预测 Logits: {final_logits.item():.4f}")

        print("\n[验收结果] 面对 Hard OOD 图：")
        print(f"- 模型预测 Logits: {logits.item():.4f} (大于0通常代表偏向OOD)")
        print("- 模型分配的注意力权重 (Alphas):")

        # 🚀 统一空间6：显卡里的张量必须用 .cpu() 拿回内存才能打印成 numpy！
        for i, alpha in enumerate(alphas.squeeze().cpu().numpy()):
            role = "中心正常" if i == 0 else "边缘变异"
            print(f"  Node {i} ({role}): {alpha:.4f}")


if __name__ == "__main__":
    train_model()