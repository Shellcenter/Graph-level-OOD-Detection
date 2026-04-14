import os
from google import genai

# 1. 注入代理防卡死
PROXY_PORT = "9674"  # 换成你的真实端口
os.environ['http_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'
os.environ['https_proxy'] = f'http://127.0.0.1:{PROXY_PORT}'

# 2. 填入你的钥匙
API_KEY = "AIzaSyBwB5Vft5rqw8l87bl0e3KmUteacVqsY_A"

print(">>> Checking API key access...")
client = genai.Client(api_key=API_KEY)

# 3. 强制拉取可用模型清单！
print("Available Gemini models:")
for model in client.models.list():
    if "gemini" in model.name:
        print(f" - {model.name}")