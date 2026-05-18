from huggingface_hub import HfApi

api = HfApi()
username = "zijian2022"

datasets = api.list_datasets(author=username)

for ds in datasets:
    #if ds.id.startswith(f"{username}/record-test"):
    try:
        api.update_repo_visibility(
            repo_id=ds.id,
            private=True,
            repo_type="dataset"
        )
        print(f"{ds.id} -> private")
    except Exception as e:
        print(f"{ds.id} ❌ {e}")