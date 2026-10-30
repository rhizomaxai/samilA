"""
모델 학습 & joblib 저장 스크립트 (인구 감소 정말 인가영? 팀 노트북 재현 + 확장)
실행:  python train_models.py
결과:  models/<모델키>.joblib — {지역: 모델} 딕셔너리, 입력 = DataFrame['연도']
       models/catalog.json    — 지역별 점수·2026~2030 예측
평가:  ① 노트북 방식: 무작위 70/30 분할(random_state=42) + 3겹 교차 검증
       ② 시간 순서: 2015~2021년(7개)으로 학습 → 2022~2025년(4개) 예측
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from pop_features import REGIONS, LastValue, build_model, load_population

ROOT = Path(__file__).parent
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)
RS = 42

pop = load_population(ROOT / "data" / "population_raw.csv")
FUTURE = pd.DataFrame({"연도": list(range(2026, 2031))})

SPECS = [
    ("baseline", "기준선 (평균값)", "학습 데이터의 평균 인구를 그대로 예측", lambda: DummyRegressor(strategy="mean"), True),
    ("linear", "최종 모델: 선형 회귀", "연도로 직선 추세를 그음 (팀 최종 모델)", lambda: build_model("linear"), True),
    ("tree", "결정 트리 (깊이 4)", "연도 구간별로 계단 모양 예측 (노트북 비교 모델)", lambda: build_model("tree"), True),
    ("poly2", "2차 다항 회귀", "연도로 2차 곡선을 그음 — 증가 후 감소 모양을 따라감", lambda: build_model("poly"), False),
    ("last", "최근 값 그대로", "'내년도 올해와 같겠지' — 가장 최근 인구를 그대로 사용", LastValue, False),
]

catalog = []
for key, name, desc, make, in_nb in SPECS:
    bundle, info = {}, {"key": key, "name": name, "desc": desc, "in_notebook": in_nb, "by_region": {}}
    for region in REGIONS:
        d = pop[pop["지역"] == region].sort_values("연도").reset_index(drop=True)
        X, y = d[["연도"]], d["총인구수"]
        a, b, c, e = train_test_split(X, y, test_size=0.3, random_state=RS)
        p = make().fit(a, c).predict(b)
        cv = cross_val_score(make(), X, y, cv=KFold(3, shuffle=True, random_state=RS), scoring="r2").mean()
        tp = make().fit(X.iloc[:7], y.iloc[:7]).predict(X.iloc[7:])
        full = make().fit(X, y)
        bundle[region] = full
        info["by_region"][region] = {
            "r2_split": round(float(r2_score(e, p)), 3), "mae_split": round(float(mean_absolute_error(e, p)), 1),
            "cv_r2": round(float(cv), 3),
            "mae_time": round(float(mean_absolute_error(y.iloc[7:], tp))),
            "test_pred": [round(float(v)) for v in tp],
            "future": {int(yr): round(float(v)) for yr, v in zip(FUTURE["연도"], full.predict(FUTURE))},
        }
    joblib.dump(bundle, OUT / f"{key}.joblib", compress=3)
    catalog.append(info)
    s = info["by_region"]["수원시"]
    print(f"{name:<18} [수원시] R² {s['r2_split']:8.3f} MAE {s['mae_split']:9.1f} CV {s['cv_r2']:7.3f}"
          f" | 시간순 MAE {s['mae_time']:6} | 2026 {s['future'][2026]:,}")

(OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
