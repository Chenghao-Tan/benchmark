from __future__ import annotations

import pandas as pd

from dataset.dataset_object import DatasetObject
from preprocess.preprocess_object import PreProcessObject
from preprocess.preprocess_utils import ensure_flag_absent
from utils.registry import register
from utils.seed import seed_context


@register("cfrl_adult_aggregate")
class CfrlAdultAggregatePreProcess(PreProcessObject):
    _EDUCATION_MAP = {
        "10th": "Dropout",
        "11th": "Dropout",
        "12th": "Dropout",
        "1st-4th": "Dropout",
        "5th-6th": "Dropout",
        "7th-8th": "Dropout",
        "9th": "Dropout",
        "Preschool": "Dropout",
        "HS-grad": "High School grad",
        "Some-college": "High School grad",
        "Masters": "Masters",
        "Prof-school": "Prof-School",
        "Assoc-acdm": "Associates",
        "Assoc-voc": "Associates",
    }
    _OCCUPATION_MAP = {
        "Adm-clerical": "Admin",
        "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar",
        "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar",
        "Handlers-cleaners": "Blue-Collar",
        "Machine-op-inspct": "Blue-Collar",
        "Other-service": "Service",
        "Priv-house-serv": "Service",
        "Prof-specialty": "Professional",
        "Protective-serv": "Other",
        "Sales": "Sales",
        "Tech-support": "Other",
        "Transport-moving": "Blue-Collar",
    }
    _COUNTRY_MAP = {
        "Cambodia": "SE-Asia",
        "Canada": "British-Commonwealth",
        "China": "China",
        "Columbia": "South-America",
        "Cuba": "Other",
        "Dominican-Republic": "Latin-America",
        "Ecuador": "South-America",
        "El-Salvador": "South-America",
        "England": "British-Commonwealth",
        "France": "Euro_1",
        "Germany": "Euro_1",
        "Greece": "Euro_2",
        "Guatemala": "Latin-America",
        "Haiti": "Latin-America",
        "Holand-Netherlands": "Euro_1",
        "Honduras": "Latin-America",
        "Hong": "China",
        "Hungary": "Euro_2",
        "India": "British-Commonwealth",
        "Iran": "Other",
        "Ireland": "British-Commonwealth",
        "Italy": "Euro_1",
        "Jamaica": "Latin-America",
        "Japan": "Other",
        "Laos": "SE-Asia",
        "Mexico": "Latin-America",
        "Nicaragua": "Latin-America",
        "Outlying-US(Guam-USVI-etc)": "Latin-America",
        "Peru": "South-America",
        "Philippines": "SE-Asia",
        "Poland": "Euro_2",
        "Portugal": "Euro_2",
        "Puerto-Rico": "Latin-America",
        "Scotland": "British-Commonwealth",
        "South": "Euro_2",
        "Taiwan": "China",
        "Thailand": "SE-Asia",
        "Trinadad&Tobago": "Latin-America",
        "United-States": "United-States",
        "Vietnam": "SE-Asia",
        "Yugoslavia": "Yugoslavia",
    }
    _MARITAL_MAP = {
        "Never-married": "Never-Married",
        "Married-AF-spouse": "Married",
        "Married-civ-spouse": "Married",
        "Married-spouse-absent": "Separated",
        "Separated": "Separated",
        "Divorced": "Separated",
        "Widowed": "Widowed",
    }
    _FEATURE_ORDER = [
        "Age",
        "Workclass",
        "Education",
        "Marital Status",
        "Occupation",
        "Relationship",
        "Race",
        "Sex",
        "Capital Gain",
        "Capital Loss",
        "Hours per week",
        "Country",
        "Target",
    ]
    _FEATURE_TYPE = {
        "Age": "numerical",
        "Workclass": "categorical",
        "Education": "categorical",
        "Marital Status": "categorical",
        "Occupation": "categorical",
        "Relationship": "categorical",
        "Race": "categorical",
        "Sex": "categorical",
        "Capital Gain": "numerical",
        "Capital Loss": "numerical",
        "Hours per week": "numerical",
        "Country": "categorical",
        "Target": "binary",
    }
    _FEATURE_MUTABILITY = {
        "Age": True,
        "Workclass": True,
        "Education": True,
        "Marital Status": False,
        "Occupation": True,
        "Relationship": False,
        "Race": False,
        "Sex": False,
        "Capital Gain": True,
        "Capital Loss": True,
        "Hours per week": True,
        "Country": True,
        "Target": False,
    }
    _FEATURE_ACTIONABILITY = {
        "Age": "same-or-increase",
        "Workclass": "any",
        "Education": "any",
        "Marital Status": "none",
        "Occupation": "any",
        "Relationship": "none",
        "Race": "none",
        "Sex": "none",
        "Capital Gain": "any",
        "Capital Loss": "any",
        "Hours per week": "any",
        "Country": "any",
        "Target": "none",
    }

    def __init__(self, seed: int | None = None, **kwargs):
        self._seed = seed

    def transform(self, input: DatasetObject) -> DatasetObject:
        with seed_context(self._seed):
            ensure_flag_absent(input, "cfrl_aggregated")
            df = input.snapshot()
            df = df.drop(columns=["fnlwgt", "Education-Num"])
            df["Education"] = df["Education"].replace(self._EDUCATION_MAP)
            df["Occupation"] = df["Occupation"].replace(self._OCCUPATION_MAP)
            df["Country"] = df["Country"].replace(self._COUNTRY_MAP)
            df["Marital Status"] = df["Marital Status"].replace(self._MARITAL_MAP)
            df["Target"] = df["Target"].astype(str).str.strip().eq(">50K").astype(int)
            df = df.loc[:, self._FEATURE_ORDER].copy(deep=True)

            input.update("cfrl_aggregated", True, df=df)
            input.update("feature_order", list(self._FEATURE_ORDER))
            input.update("raw_feature_type", dict(self._FEATURE_TYPE))
            input.update("raw_feature_mutability", dict(self._FEATURE_MUTABILITY))
            input.update("raw_feature_actionability", dict(self._FEATURE_ACTIONABILITY))
            return input
