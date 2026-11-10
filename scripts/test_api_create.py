import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Write credentials file
kaggle_dir = Path("/home/XiaoSha/.kaggle")
kaggle_dir.mkdir(exist_ok=True)
with open(kaggle_dir / "kaggle.json", "w") as f:
    f.write('{"username":"salala1706","key":"KGAT_33fafe947e581fe30c5a78aa1538ee99"}')
os.chmod(kaggle_dir / "kaggle.json", 0o600)

api = KaggleApi()
try:
    api.authenticate()
    print("Authentication OK!")
    # Test creating a tiny 1-file dataset
    test_dir = Path("/home/XiaoSha/test_dataset_kaggle")
    test_dir.mkdir(exist_ok=True)
    with open(test_dir / "hello.txt", "w") as f:
        f.write("Hello Kaggle")

    meta = {
        "title": "Test Dummy Salala",
        "id": "salala1706/test-dummy-salala",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    import json
    with open(test_dir / "dataset-metadata.json", "w") as f:
        json.dump(meta, f)

    res = api.dataset_create_new(folder=str(test_dir), dir_mode="tar", quiet=False)
    print("Dataset create result:", res)
except Exception as e:
    print("KaggleApi Exception:", type(e), e)
