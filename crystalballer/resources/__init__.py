from pathlib import Path
from typing import Union

from crystalballer.o3d_utils.stls import GlobeGeometry


def resource_path(path: Union[Path, str]) -> Path:
    """Verify a resource path exists and return it as a Path object."""
    return Path(path).resolve(strict=True)


_STL_DIR = resource_path("crystalballer/resources/stls")

GLOBE = GlobeGeometry(
    stl_paths=[
        resource_path(_STL_DIR / "globe" / "globe_lower.stl"),
        resource_path(_STL_DIR / "globe" / "globe_upper.stl"),
    ]
)
