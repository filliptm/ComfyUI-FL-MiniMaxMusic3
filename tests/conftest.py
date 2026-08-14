import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import server


class FakeRoutes:
    def get(self, _path):
        return lambda function: function

    def post(self, _path):
        return lambda function: function


server.PromptServer.instance = SimpleNamespace(
    routes=FakeRoutes(),
    client_id=None,
    send_sync=lambda *_args, **_kwargs: None,
)

CUSTOM_NODES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CUSTOM_NODES))
PACKAGE = "ComfyUI-FL-MiniMaxMusic3"


def pack_module(name):
    return importlib.import_module(f"{PACKAGE}.{name}")
