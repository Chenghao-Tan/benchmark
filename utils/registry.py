from __future__ import annotations

from copy import deepcopy

global_registry: dict[str, dict[str, type]] = {
    "Dataset": {},
    "PreProcess": {},
    "Method": {},
    "TargetModel": {},
    "Evaluation": {},
}


def register(name: str):
    def decorator(cls: type) -> type:
        registry_type = None
        mro_names = {base.__name__ for base in cls.__mro__}

        if "DatasetObject" in mro_names:
            registry_type = "Dataset"
        elif "PreProcessObject" in mro_names:
            registry_type = "PreProcess"
        elif "MethodObject" in mro_names:
            registry_type = "Method"
        elif "ModelObject" in mro_names:
            registry_type = "TargetModel"
        elif "EvaluationObject" in mro_names:
            registry_type = "Evaluation"

        if registry_type is None:
            raise TypeError(f"Cannot register unsupported class type: {cls.__name__}")
        if name in global_registry[registry_type]:
            raise KeyError(f"{registry_type} '{name}' is already registered")

        global_registry[registry_type][name] = cls
        return cls

    return decorator


def get_registry(registry_type: str) -> dict[str, type]:
    registry_type = registry_type.lower()

    mapping = {
        "dataset": "Dataset",
        "preprocess": "PreProcess",
        "method": "Method",
        "targetmodel": "TargetModel",  # Backup name
        "model": "TargetModel",
        "evaluation": "Evaluation",
    }
    type_name = mapping.get(registry_type)
    if type_name is None:
        raise KeyError(f"Unknown registry type: {type_name}")
    return deepcopy(global_registry[type_name])
