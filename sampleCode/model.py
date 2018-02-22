import os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.modelselection import train_test_split, GridSearchCv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings

warnings.filterwarnings("ignore")


data = pd.read_csv("train.csv")
