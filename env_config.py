import os

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv is required. Install it with: pip install python-dotenv"
    ) from exc

load_dotenv()


def configure_proxy():
    """根据环境变量配置本地 HTTP(S) 代理。"""
    proxy_port = os.getenv("PROXY_PORT")
    if proxy_port:
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
    return proxy_port


def get_api_key():
    """从环境变量中读取 API Key。"""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API key not found in .env. Set GEMINI_API_KEY "
            "or OPENAI_API_KEY."
        )
    return api_key
