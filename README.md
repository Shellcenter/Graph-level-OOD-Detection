# Graph-level OOD Detection

这是一个用于图分布外检测实验的研究代码仓库，包含两类主要实验：

- 节点级 OOD 检测：基于 `Cora` 数据集，对测试节点特征施加扰动后评估异常检测性能。
- 图级 OOD 检测：利用 Gemini 生成图节点语义，再训练图级 OOD 分类模型。

## 安全配置

本仓库不会包含任何真实 API Key。

请仅在本地项目根目录创建 `.env` 文件，并填写你自己的配置：

```env
GEMINI_API_KEY=your_real_api_key
OPENAI_API_KEY=
PROXY_PORT=9674
HF_ENDPOINT=https://hf-mirror.com
```

注意事项：

- 不要把真实密钥写进任何 `.py`、`.md` 或提交记录中。
- 不要提交本地 `.env` 文件。
- 仓库中保留的 `.env.example` 仅作为模板。

## 主要文件

- `main.py`：节点级主实验入口
- `run_cora_benchmark.py`：节点级基线与主模型对比实验
- `generate_graph_ood.py`：图级数据生成脚本
- `run_experiment.py`：图级训练与评估主入口
- `run_analysis.py`：图级结果分析与可视化
- `test_models.py`：测试 Gemini API Key 是否可用
- `env_config.py`：从 `.env` 读取代理与 API Key

## 实验前准备

1. 安装依赖，包括 `torch`、`torch-geometric`、`python-dotenv`、`sentence-transformers`、`google-genai` 等。
2. 在本地创建 `.env` 并填写真实配置。
3. 如果需要访问外网资源，确认代理端口正确且代理程序已启动。

## 如何运行

### 1. 测试 API 配置

```powershell
python test_models.py
```

### 2. 节点级实验

运行主实验：

```powershell
python main.py
```

运行基线对比实验：

```powershell
python run_cora_benchmark.py
```

### 3. 图级实验

先单独测试数据生成：

```powershell
python generate_graph_ood.py
```

再运行图级训练与评估：

```powershell
python run_experiment.py
```

最后运行分析脚本：

```powershell
python run_analysis.py
```

## 缓存与本地文件

以下内容默认视为本地产物，不建议提交：

- `data/`
- `logs/`
- `graph_ood_dataset.pt`
- `attention_heatmap.png`
- `.env`

如果你删除了 `data/` 或 `graph_ood_dataset.pt`，下次运行对应实验时会重新下载或重新生成。
