from google import genai
from env_config import configure_proxy, get_api_key

proxy_port = configure_proxy()
if proxy_port:
    print(f"Proxy configured on port: {proxy_port}")

print("Checking API key access...")
client = genai.Client(api_key=get_api_key())

# List available Gemini models.
print("Available Gemini models:")
for model in client.models.list():
    if "gemini" in model.name:
        print(f" - {model.name}")