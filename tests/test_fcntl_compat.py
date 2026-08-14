import importlib.util
import tempfile
import unittest
from pathlib import Path


class FcntlCompatTests(unittest.TestCase):
    @unittest.skipUnless(__import__("os").name == "nt", "Windows compatibility module")
    def test_lock_and_unlock_empty_file(self):
        path = Path(__file__).parents[1] / "training" / "compat" / "fcntl.py"
        spec = importlib.util.spec_from_file_location("fl_fcntl_compat", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.NamedTemporaryFile(mode="w+") as file:
            module.flock(file, module.LOCK_EX)
            module.flock(file, module.LOCK_UN)
