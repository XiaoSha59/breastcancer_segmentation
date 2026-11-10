import requests
from requests.auth import HTTPBasicAuth

usernames = ["salala1706", "Salala1706"]
key = "KGAT_33fafe947e581fe30c5a78aa1538ee99"

for u in usernames:
    test_payload = {"name": "test.txt", "contentLength": 10, "lastModifiedEpochSeconds": 1700000000}
    r = requests.post(
        "https://www.kaggle.com/api/v1/blobs/upload",
        auth=HTTPBasicAuth(u, key),
        json=test_payload,
    )
    print(f"Username: {u} | Status: {r.status_code} | Response: {r.text[:200]}")
