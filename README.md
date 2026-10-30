# 수원시 인구 변화 분석 — Streamlit 웹앱 (인구 감소 정말 인가영? 팀)

## 폴더 구조
```
streamlit_app/
├── app.py               # 웹앱 (6개 탭: 문제 정의 ~ 결론, 경기도 비교·미래 예측 포함)
├── pop_features.py      # 공통 전처리 + 노트북과 같은 build_model + '최근 값' 모델
├── train_models.py      # 모델 학습 → models/*.joblib 저장 (노트북 결과 재현)
├── requirements.txt
├── data/population_raw.csv   # 행정안전부 주민등록인구현황 (KOSIS, cp949)
└── models/
    ├── baseline.joblib  linear.joblib  tree.joblib  poly2.joblib  last.joblib
    └── catalog.json     # 지역별 점수·2026~2030 예측
```
각 `.joblib` 파일은 `{지역: 모델}` 딕셔너리(수원시·경기도)이고, 입력은 `연도` 열 하나인 DataFrame입니다.

## 실행 / 배포
```bash
pip install -r requirements.txt
python train_models.py      # (선택) 모델 다시 만들기
streamlit run app.py
```
GitHub에 올린 뒤 share.streamlit.io에서 `app.py`를 지정하면 배포됩니다.
`pop_features.py`는 반드시 app.py와 같은 폴더에 있어야 joblib 모델을 불러올 수 있습니다.
