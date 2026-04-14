import os

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise ImportError(
        "python-dotenv 未安装，请先运行: pip install python-dotenv"
    ) from exc

load_dotenv()


def configure_proxy():
    proxy_port = os.getenv("PROXY_PORT")
    if proxy_port:
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        os.environ["http_proxy"] = proxy_url
        os.environ["https_proxy"] = proxy_url
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
    return proxy_port


def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未在 .env 中找到 API Key。请设置 GEMINI_API_KEY，"
            "或兼容使用 OPENAI_API_KEY。"
        )
    return api_key
