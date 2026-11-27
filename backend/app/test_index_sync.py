"""
Tests for services/index_sync.py, against a fake huggingface_hub (no network).
Run from backend/:  python -m unittest app.test_index_sync
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services import index_sync
from app.services.index_sync import download_index_if_missing, upload_index


class FakeHub:
    """Stands in for the huggingface_hub module: one dataset repo."""

    def __init__(self, files=None, private=True, exists=True):
        self.files = {"faiss.index": b"vectors", "metadata.json": b"{}"} if files is None else files
        self.private = private
        self.exists = exists
        self.sha = "abc1234def"
        self.downloads = []  # (filename, revision)
        self.created = []  # (repo, private)
        self.uploads = []  # upload_folder keyword arguments
        self.fail_on = None  # filename whose download raises
        hub = self

        class HfApi:
            def dataset_info(self, repo):
                if not hub.exists:
                    raise FileNotFoundError(f"404: {repo}")
                return types.SimpleNamespace(sha=hub.sha, private=hub.private)

            def create_repo(self, repo, repo_type, private, exist_ok):
                assert repo_type == "dataset" and exist_ok
                if not hub.exists:
                    hub.exists, hub.private = True, private
                hub.created.append((repo, private))

            def upload_folder(self, **kwargs):
                hub.uploads.append(kwargs)
                return types.SimpleNamespace(commit_url=f"https://hub.example/{kwargs['repo_id']}/commit/1")

        self.HfApi = HfApi

    def hf_hub_download(self, repo, name, repo_type, revision, local_dir):
        assert repo_type == "dataset"
        self.downloads.append((name, revision))
        if name == self.fail_on or name not in self.files:
            raise OSError(f"cannot fetch {name}")
        with open(os.path.join(local_dir, name), "wb") as f:
            f.write(self.files[name])


class IndexSyncTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.data = os.path.join(self.dir.name, "data")
        self.addCleanup(self.dir.cleanup)

    def run_with(self, hub, env, fn, *args, **kwargs):
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), mock.patch.dict(
            os.environ, env, clear=True
        ), mock.patch("builtins.print"):
            return fn(*args, **kwargs)

    def write(self, name, content=b"local"):
        os.makedirs(self.data, exist_ok=True)
        with open(os.path.join(self.data, name), "wb") as f:
            f.write(content)

    def read(self, name):
        with open(os.path.join(self.data, name), "rb") as f:
            return f.read()

    # --- download ---

    def test_does_nothing_without_index_repo(self):
        hub = FakeHub()
        self.assertFalse(self.run_with(hub, {}, download_index_if_missing, self.data))
        self.assertFalse(self.run_with(hub, {"INDEX_REPO": "  "}, download_index_if_missing, self.data))
        self.assertEqual(hub.downloads, [])
        self.assertFalse(os.path.exists(self.data))

    def test_downloads_both_files_from_one_revision(self):
        hub = FakeHub()
        self.assertTrue(self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data))
        self.assertEqual(self.read("faiss.index"), b"vectors")
        self.assertEqual(self.read("metadata.json"), b"{}")
        self.assertEqual(hub.downloads, [("faiss.index", hub.sha), ("metadata.json", hub.sha)])
        self.assertEqual(sorted(os.listdir(self.data)), ["faiss.index", "metadata.json"])

    def test_keeps_an_existing_index(self):
        self.write("faiss.index")
        self.write("metadata.json")
        hub = FakeHub()
        self.assertFalse(self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data))
        self.assertEqual(self.read("faiss.index"), b"local")
        self.assertEqual(hub.downloads, [])

    def test_leaves_a_half_index_alone(self):
        for name in index_sync.INDEX_FILES:
            with self.subTest(present=name):
                hub = FakeHub()
                data = os.path.join(self.dir.name, name + "_only")
                os.makedirs(data)
                with open(os.path.join(data, name), "wb") as f:
                    f.write(b"local")
                self.assertFalse(self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, data))
                self.assertEqual(os.listdir(data), [name])

    def test_failed_download_installs_nothing(self):
        for fail_on in index_sync.INDEX_FILES:
            with self.subTest(fail_on=fail_on):
                hub = FakeHub()
                hub.fail_on = fail_on
                data = os.path.join(self.dir.name, "fail_" + fail_on)
                with self.assertRaises(RuntimeError) as raised:
                    self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, data)
                self.assertIn("me/idx", str(raised.exception))
                self.assertIn("HF_TOKEN", str(raised.exception))
                self.assertEqual(os.listdir(data), [])

    def test_missing_repo_raises(self):
        hub = FakeHub(exists=False)
        with self.assertRaises(RuntimeError):
            self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data)
        self.assertEqual(os.listdir(self.data), [])

    def test_repo_without_a_commit_raises(self):
        hub = FakeHub()
        hub.sha = None
        with self.assertRaises(RuntimeError):
            self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data)
        self.assertEqual(hub.downloads, [])
        self.assertEqual(os.listdir(self.data), [])

    def test_clears_a_stale_staging_folder(self):
        stale = os.path.join(self.data, ".staging_download")
        os.makedirs(stale)
        with open(os.path.join(stale, "faiss.index"), "wb") as f:
            f.write(b"partial")
        hub = FakeHub()
        self.assertTrue(self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data))
        self.assertEqual(self.read("faiss.index"), b"vectors")
        self.assertEqual(sorted(os.listdir(self.data)), ["faiss.index", "metadata.json"])

    def test_retry_after_failure_succeeds(self):
        hub = FakeHub()
        hub.fail_on = "metadata.json"
        with self.assertRaises(RuntimeError):
            self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data)
        hub.fail_on = None
        self.assertTrue(self.run_with(hub, {"INDEX_REPO": "me/idx"}, download_index_if_missing, self.data))
        self.assertEqual(sorted(os.listdir(self.data)), ["faiss.index", "metadata.json"])

    # --- upload ---

    def test_upload_needs_a_repo(self):
        self.write("faiss.index")
        self.write("metadata.json")
        with self.assertRaises(ValueError):
            self.run_with(FakeHub(), {}, upload_index, self.data)

    def test_upload_needs_both_files(self):
        self.write("faiss.index")
        hub = FakeHub()
        with self.assertRaises(FileNotFoundError):
            self.run_with(hub, {"INDEX_REPO": "me/idx"}, upload_index, self.data)
        self.assertEqual(hub.created, [])
        self.assertEqual(hub.uploads, [])

    def test_upload_creates_a_private_repo_and_sends_only_the_index(self):
        self.write("faiss.index")
        self.write("metadata.json")
        self.write("metadata.json.old")
        hub = FakeHub(exists=False)
        url = self.run_with(hub, {"INDEX_REPO": "me/idx"}, upload_index, self.data)
        self.assertEqual(hub.created, [("me/idx", True)])
        self.assertEqual(len(hub.uploads), 1)
        sent = hub.uploads[0]
        self.assertEqual(sent["repo_id"], "me/idx")
        self.assertEqual(sent["repo_type"], "dataset")
        self.assertEqual(sent["folder_path"], self.data)
        self.assertEqual(sent["allow_patterns"], ["faiss.index", "metadata.json"])
        self.assertIn("me/idx", url)

    def test_upload_refuses_a_public_repo(self):
        self.write("faiss.index")
        self.write("metadata.json")
        for private in (False, None):
            with self.subTest(private=private):
                hub = FakeHub(private=private)
                with self.assertRaises(RuntimeError) as raised:
                    self.run_with(hub, {"INDEX_REPO": "me/idx"}, upload_index, self.data)
                self.assertIn("public", str(raised.exception))
                self.assertEqual(hub.uploads, [])

    def test_upload_repo_argument_overrides_the_environment(self):
        self.write("faiss.index")
        self.write("metadata.json")
        hub = FakeHub()
        self.run_with(hub, {"INDEX_REPO": "me/idx"}, upload_index, self.data, repo="me/other")
        self.assertEqual(hub.uploads[0]["repo_id"], "me/other")


if __name__ == "__main__":
    unittest.main()
