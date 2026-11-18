"""Which LeRobot release this branch of fastload is written for.

fastload hooks into LeRobot internals that change between minor releases, so every LeRobot minor version
has its own fastload branch (``lerobot-0.4``, ``lerobot-0.5``, ...). Installing the wrong one fails here
with the branch to use instead of breaking somewhere inside training.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

BRANCH = "lerobot-0.4"
MIN_VERSION = (0, 4, 2)  # inclusive
MAX_VERSION = (0, 5, 0)  # exclusive
REPO = "https://github.com/eunyoung030927/lerobot-fastload.git"


def _parse(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split("+")[0].split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        if not num:
            break
        parts.append(int(num))
        if len(parts) == 3:
            break
    return tuple(parts + [0] * (3 - len(parts)))


def check_lerobot_version() -> str:
    try:
        installed = version("lerobot")
    except PackageNotFoundError as e:  # pragma: no cover
        raise RuntimeError("fastload: LeRobot is not installed") from e
    v = _parse(installed)
    if not (MIN_VERSION <= v < MAX_VERSION):
        fmt = ".".join
        want = f"lerobot-{v[0]}.{v[1]}"
        raise RuntimeError(
            f"fastload branch {BRANCH} supports LeRobot >={fmt(map(str, MIN_VERSION))},<{fmt(map(str, MAX_VERSION))}, "
            f"but LeRobot {installed} is installed. Install the matching branch:\n"
            f"    pip install git+{REPO}@{want}"
        )
    return installed
