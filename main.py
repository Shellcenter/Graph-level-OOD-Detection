import sys
import os
# 🚀 导师的强行引路代码：把当前项目根目录强行加入 Python 的雷达中！
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import logging
from models.anomaly_aware import NodeAnomalyAwareModel


def main():
    # 1. 严肃的参数解析 (像 LLMGuard 的 parse.py 一样)
    parser = argparse.ArgumentParser(description="Graph OOD Detection Benchmark")
    parser.add_argument("--dataset", type=str, default="Cora", help="数据集名称")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    args = parser.parse_args()

    # 2. 专业的日志输出，不再只有 print
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"正在启动实验，数据集: {args.dataset}, 学习率: {args.lr}")

    # ... 这里的训练逻辑保持不变，但调用的是 models/ 里的类 ...

if __name__ == "__main__":
    main()