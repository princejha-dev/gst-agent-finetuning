"""Upload the dataset to the Hugging Face Hub.
Usage: huggingface-cli login   (or set HF_TOKEN), then:
       python push_to_hub.py --repo YOUR_USERNAME/gst-toolcalling-india
"""
# import argparse
# from huggingface_hub import HfApi

# ap = argparse.ArgumentParser()
# ap.add_argument("--repo", required=True)
# a = ap.parse_args()

# api = HfApi()
# api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
# api.upload_file(path_or_fileobj="DATASET_CARD.md", path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
# api.upload_file(path_or_fileobj="tools.json", path_in_repo="tools.json", repo_id=a.repo, repo_type="dataset")
# api.upload_file(path_or_fileobj="generate_dataset.py", path_in_repo="generate_dataset.py", repo_id=a.repo, repo_type="dataset")
# api.upload_folder(folder_path="data", path_in_repo="data", repo_id=a.repo, repo_type="dataset")
# print(f"Done: https://huggingface.co/datasets/{a.repo}")



import argparse
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
a = ap.parse_args()

api = HfApi()

api.create_repo(
    repo_id=a.repo,
    repo_type="dataset",
    exist_ok=True
)

api.upload_file(
    path_or_fileobj="DATASET_CARD.md",
    path_in_repo="README.md",
    repo_id=a.repo,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj="src/tools.json",
    path_in_repo="tools.json",
    repo_id=a.repo,
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj="src/generate_dataset.py",
    path_in_repo="generate_dataset.py",
    repo_id=a.repo,
    repo_type="dataset"
)

api.upload_folder(
    folder_path="data",
    path_in_repo="data",
    repo_id=a.repo,
    repo_type="dataset"
)

print(f"Done: https://huggingface.co/datasets/{a.repo}")
