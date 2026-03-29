import requests
import json

url = "http://localhost:8000/chat"
# Using unicode-escaped characters to bypass any environment encoding issues
query = "\u0410 \u0437\u0430 \u0433\u0440\u0430\u043d\u0438\u0446\u0435\u0439, \u0433\u0434\u0435 \u0442\u044b \u0431\u044b\u0432\u0430\u043b?"
payload = {
    "message": query,
    "history": []
}

try:
    # Explicitly ensure the session handles UTF-8
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        print(f"RESPONSE:\n{json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    else:
        print(f"ERROR {response.status_code}: {response.text}")
except Exception as e:
    print(f"EXCEPTION: {e}")
