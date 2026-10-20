import importlib
import pathlib
import sys
import types

PACKAGE_NAME: str = "adaptive_samplers_pkg"


def module(name: str) -> types.ModuleType:
    root: pathlib.Path = pathlib.Path(__file__).resolve().parent
    if PACKAGE_NAME not in sys.modules:
        package: types.ModuleType = types.ModuleType(PACKAGE_NAME)
        package.__path__ = [str(root)]
        package.__package__ = PACKAGE_NAME
        sys.modules[PACKAGE_NAME] = package
    return importlib.import_module(f"{PACKAGE_NAME}.{name}")
