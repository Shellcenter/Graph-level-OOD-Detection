from google import genai
from env_config import configure_proxy, get_api_key

proxy_port = configure_proxy()
if proxy_port:
    print(f"成功加载配置，正在使用端口: {proxy_port}")

print(">>> Checking API key access...")
client = genai.Client(api_key=get_api_key())

# 3. 强制拉取可用模型清单！
print("Available Gemini models:")
for model in client.models.list():
    if "gemini" in model.name:
        print(f" - {model.name}")