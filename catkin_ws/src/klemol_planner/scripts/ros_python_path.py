from __future__ import annotations

from pathlib import Path
import sys


def enable_generated_ros_modules(package_root: Path) -> None:
    catkin_ws = package_root.parents[1]
    devel_lib = catkin_ws / "devel" / "lib"
    generated_paths = sorted(devel_lib.glob("python*/dist-packages"))
    for generated_path in generated_paths:
        if str(generated_path) not in sys.path:
            sys.path.append(str(generated_path))

    try:
        import klemol_planner
    except Exception:
        return

    for generated_path in generated_paths:
        generated_package = generated_path / "klemol_planner"
        if generated_package.exists() and str(generated_package) not in klemol_planner.__path__:
            klemol_planner.__path__.append(str(generated_package))
