import runpy
import sys
from pathlib import Path


def main():
    train_path = Path(sys.argv[1]).resolve()
    sys.argv = [str(train_path), *sys.argv[2:]]

    # PyTorch 2.11 must materialize this module before the model stack invokes its lazy compiler hook.
    import torch._inductor.decomposition  # noqa: F401

    runpy.run_path(str(train_path), run_name="__main__")


if __name__ == "__main__":
    main()
