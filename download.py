from huggingface_hub import snapshot_download

snapshot_download(
repo_id="BAAI/bge-reranker-base",
local_dir="C:/models/bge-reranker-base",
resume_download=True
)
