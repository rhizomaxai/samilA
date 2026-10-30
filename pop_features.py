"""
공통 모듈: 인구 데이터 전처리 + 노트북과 같은 모델 만들기 + '최근 값' 모델
- train_models.py(모델 학습)와 app.py(웹앱)가 함께 import 합니다.
- joblib 모델 안에 이 파일의 클래스가 들어 있으므로 app.py와 같은 폴더에 두어야 합니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

REGIONS = ["수원시", "경기도"]


def load_population(csv_path: Path) -> pd.DataFrame:
    """KOSIS 가로형 CSV(cp949) → 세로형 (지역·연도·총인구수·남자인구수·여자인구수)
    원본은 연도마다 [총인구, 남자, 여자] 3칸이 반복되는 구조다."""
    raw = pd.read_csv(csv_path, encoding="cp949", header=None)
    years = raw.iloc[0, 1::3].astype(int).tolist()
    rows = []
    for _, r in raw.iloc[2:].iterrows():
        v = r.iloc[1:].astype(int).to_numpy()
        for i, y in enumerate(years):
            rows.append({"지역": r.iloc[0], "연도": y, "총인구수": v[3 * i],
                         "남자인구수": v[3 * i + 1], "여자인구수": v[3 * i + 2]})
    return pd.DataFrame(rows)


def build_model(algorithm, degree=2, random_state=42):
    """노트북의 build_model과 같은 구조: (중앙값 대체 + 표준화) → 모델. 입력 열은 '연도' 하나."""
    prep = make_column_transformer((make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["연도"]))
    if algorithm == "poly":
        return make_pipeline(prep, PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())
    if algorithm == "tree":
        return make_pipeline(prep, DecisionTreeRegressor(max_depth=4, random_state=random_state))
    return make_pipeline(prep, LinearRegression())


class LastValue(BaseEstimator, RegressorMixin):
    """'내년도 올해와 같겠지' — 가장 최근 연도의 인구를 그대로 내놓는 단순 예측"""

    def fit(self, X, y):
        order = np.argsort(np.asarray(X["연도"]))
        self.last_ = float(np.asarray(y)[order[-1]])
        return self

    def predict(self, X):
        return np.full(len(X), self.last_)
