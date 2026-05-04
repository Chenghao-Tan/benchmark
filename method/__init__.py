from __future__ import annotations

from importlib import import_module


def _optional_import(module_path: str, symbol: str) -> None:
    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        return
    globals()[symbol] = getattr(module, symbol)


_optional_import("method.apas.apas", "ApasMethod")
_optional_import("method.arg_ensembling.arg_ensembling", "ArgEnsemblingMethod")
_optional_import("method.cchvae.cchvae", "CchvaeMethod")
_optional_import("method.cemsp.cemsp", "CemspMethod")
_optional_import("method.cfrl.cfrl", "CfrlMethod")
_optional_import("method.cfvae.cfvae", "CfvaeMethod")
_optional_import("method.claproar.claproar", "ClaproarMethod")
_optional_import("method.clue.clue", "ClueMethod")
_optional_import("method.cogs.cogs", "CogsMethod")
_optional_import("method.cols.cols", "ColsMethod")
_optional_import("method.cruds.cruds", "CrudsMethod")
_optional_import("method.cvas_proj.cvas_proj", "CvasProjMethod")
_optional_import("method.dice.dice", "DiceMethod")
_optional_import("method.diverse_dist.diverse_dist", "DiverseDistMethod")
_optional_import("method.face.face", "FaceMethod")
_optional_import("method.feature_tweak.feature_tweak", "FeatureTweakMethod")
_optional_import("method.gravitational.gravitational", "GravitationalMethod")
_optional_import("method.gs.gs", "GsMethod")
_optional_import("method.larr.larr", "LarrMethod")
_optional_import("method.mace.mace", "MaceMethod")
_optional_import("method.method_object", "MethodObject")
_optional_import("method.probe.probe", "ProbeMethod")
_optional_import("method.proplace.proplace", "ProplaceMethod")
_optional_import("method.rbr.rbr", "RbrMethod")
_optional_import("method.revise.revise", "ReviseMethod")
_optional_import("method.roar.roar", "RoarMethod")
_optional_import("method.sns.sns", "SnsMethod")
_optional_import("method.toy.toy", "ToyMethod")
_optional_import("method.trex.trex", "TrexMethod")
_optional_import("method.wachter.wachter", "WachterMethod")
