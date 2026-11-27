"""
Keep the built index in a private Hugging Face dataset repo.

The texts and the index built from them aren't committed, so a deployment
built from the repository has no library. With INDEX_REPO set, the server
downloads the two index files at startup when it has none, and
`python -m app.scripts.upload_index` uploads a locally built index.

Both read the access token from HF_TOKEN (huggingface_hub does this itself).
"""

import os
import shutil

INDEX_FILES = ("faiss.index", "metadata.json")


def _repo(repo: str | None = None) -> str:
    return (repo or os.getenv("INDEX_REPO", "")).strip()


def download_index_if_missing(data_dir: str = "data") -> bool:
    """
    Download the index into data_dir if INDEX_REPO is set and data_dir has
    neither index file. Returns True if it downloaded.

    An index that is already there is never replaced: it may hold documents
    uploaded through the API. If only one of the two files is there, nothing
    is downloaded and VectorStore reports the broken state.

    Raises:
        RuntimeError: the download failed. The server should not start with
            an empty library when one was asked for.
    """
    repo = _repo()
    if not repo:
        return False
    if any(os.path.exists(os.path.join(data_dir, name)) for name in INDEX_FILES):
        return False

    from huggingface_hub import HfApi, hf_hub_download

    print(f"📥 No index in {data_dir}; downloading it from {repo}...")
    staging = os.path.join(data_dir, ".staging_download")
    try:
        shutil.rmtree(staging, ignore_errors=True)
        os.makedirs(staging)
        # Both files from one commit, so an upload that lands between the two
        # downloads can't leave a mismatched pair
        revision = HfApi().dataset_info(repo).sha
        if not revision:
            raise ValueError("the repo reported no commit")
        for name in INDEX_FILES:
            hf_hub_download(
                repo, name, repo_type="dataset", revision=revision, local_dir=staging
            )
        # Moved in only once both are complete. If this is interrupted
        # between the two, VectorStore refuses to start on the lone file.
        for name in INDEX_FILES:
            os.replace(os.path.join(staging, name), os.path.join(data_dir, name))
    except Exception as e:
        raise RuntimeError(
            f"Could not download the index from the Hugging Face dataset "
            f"{repo!r} ({type(e).__name__}: {e}). Check INDEX_REPO and HF_TOKEN."
        ) from e
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"✓ Index downloaded ({revision[:7]})")
    return True


def upload_index(data_dir: str = "data", repo: str | None = None) -> str:
    """
    Upload the index in data_dir to the dataset repo in one commit, creating
    the repo as private if it doesn't exist. Returns the commit's URL.

    Raises:
        ValueError: no repo was given and INDEX_REPO is unset.
        FileNotFoundError: an index file is missing from data_dir.
        RuntimeError: the repo is public. The index holds the full text of
            every passage, so it is never uploaded to a public repo.
    """
    repo = _repo(repo)
    if not repo:
        raise ValueError("Set INDEX_REPO to a Hugging Face dataset repo, e.g. your-name/philosearch-index")
    for name in INDEX_FILES:
        if not os.path.isfile(os.path.join(data_dir, name)):
            raise FileNotFoundError(
                f"{os.path.join(data_dir, name)} not found. Build the index "
                "first with `python -m app.scripts.index_texts`."
            )

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    if api.dataset_info(repo).private is not True:
        raise RuntimeError(
            f"The dataset repo {repo!r} is public. The index contains the full "
            "text of every passage; make the repo private before uploading."
        )
    commit = api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=data_dir,
        allow_patterns=list(INDEX_FILES),
        commit_message="Update the index",
    )
    return commit.commit_url
