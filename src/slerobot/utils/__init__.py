from . import constants

__all__ = ["constants", "control_utils", "robot_utils", "utils"]


def __getattr__(name: str):
    if name == "control_utils":
        from . import control_utils

        return control_utils
    if name == "robot_utils":
        from . import robot_utils

        return robot_utils
    if name == "utils":
        from . import utils as utils_module

        return utils_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
