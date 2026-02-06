
import os
import sys

# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("--- Starting Proxy Test ---")
print(f"Initial HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"Initial HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")

# Import config, which triggers ProxyManager.configure()
try:
    from config import Config
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)

print("\n--- After Config Import ---")
print(f"Final HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
print(f"Final HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")

# Verify connectivity manually
import requests
try:
    print("\nTesting connection to huggingface.co...")
    response = requests.head("https://huggingface.co", timeout=5)
    print(f"Connection successful! Status code: {response.status_code}")
except Exception as e:
    print(f"Connection failed: {e}")
