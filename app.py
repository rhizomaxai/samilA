"""
수원시 인구, 정말 줄고 있을까? — 삼일고 2학년 '인구 감소 정말 인가영?' 팀 수원시 인구 변화 분석
실행:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import pop_features  # noqa: F401  (joblib 모델 안의 클래스를 불러오는 데 필요)
from pop_features import REGIONS, load_population

ROOT = Path(__file__).parent
BLUE, PINK, GREEN, GRAY, RED = "#2563eb", "#db2777", "#16a34a", "#94a3b8", "#dc2626"
MODEL_COLORS = {"baseline": GRAY, "linear": BLUE, "tree": "#f59e0b", "poly2": "#7c3aed", "last": GREEN}

st.set_page_config(page_title="수원시 인구 변화 분석", page_icon="🏙️", layout="wide")


@st.cache_data
def get_data():
    return load_population(ROOT / "data" / "population_raw.csv")


@st.cache_data
def get_catalog():
    return json.loads((ROOT / "models" / "catalog.json").read_text(encoding="utf-8"))


@st.cache_resource          # 모델별로 한 번만 joblib 로드
def load_model(key):
    return joblib.load(ROOT / "models" / f"{key}.joblib")


def predict(key, region, years):
    return load_model(key)[region].predict(pd.DataFrame({"연도": list(years)}))


def man(n):
    return f"{n / 10000:,.1f}만 명"


pop = get_data()
sw = pop[pop["지역"] == "수원시"].set_index("연도")
gg = pop[pop["지역"] == "경기도"].set_index("연도")
catalog = get_catalog()
cat = {c["key"]: c for c in catalog}
names = {c["key"]: c["name"] for c in catalog}

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.markdown("### 🏙️ 인구 감소 정말 인가영?")
    st.caption("삼일고등학교 · 2학년 2반·3반")
    st.markdown("""
| 이름 | 역할 |
|---|---|
| 이소정 | 팀장·발표, 데이터 수집, 보고서·시각화 |
| 이가영 | 전처리, 모델링·평가 |
""")
    st.divider()
    st.caption("데이터: 행정안전부 「주민등록인구현황」 (KOSIS 국가통계포털) 행정구역(시군구)별 성별 인구수, "
               "2015~2025, 공공누리")
    st.caption("제출일: 2026. 10. 30.")

# ───────────────────────── 헤더 ─────────────────────────
st.title("🏙️ 수원시 인구, 정말 줄고 있을까?")
st.markdown("**저출산·고령화 시대, 2015~2025년 수원시와 경기도 인구 변화 분석 및 예측**")

peak_y = sw["총인구수"].idxmax()
c1, c2, c3, c4 = st.columns(4)
c1.metric("2025년 수원시 인구", man(sw.loc[2025, "총인구수"]),
          f"{sw.loc[2025, '총인구수'] - sw.loc[2015, '총인구수']:+,}명 (2015년 대비)", delta_arrow="off")
c2.metric(f"최고점 ({peak_y}년) 대비", f"{sw.loc[2025, '총인구수'] - sw['총인구수'].max():+,}명",
          f"{(sw.loc[2025, '총인구수'] / sw['총인구수'].max() - 1) * 100:+.1f}%", delta_arrow="off")
c3.metric("같은 기간 경기도", f"{(gg.loc[2025, '총인구수'] / gg.loc[2015, '총인구수'] - 1) * 100:+.1f}%",
          help="2015 → 2025 증가율")
c4.metric("최종 모델 2026 예측 (수원시)", man(cat["linear"]["by_region"]["수원시"]["future"]["2026"]),
          help="연도를 이용한 선형 회귀")

tabs = st.tabs(["① 문제 정의", "② 데이터", "③ 탐색: 정말 줄었을까", "④ 모델 평가", "⑤ 미래 인구 예측", "⑥ 결론·한계"])

# ───────────────────────── ① 문제 정의 ─────────────────────────
with tabs[0]:
    st.subheader("배경과 목적")
    st.info("저출산과 고령화로 인한 인구 감소가 중요한 사회 문제로 떠오르고 있다. 인구 변화는 지역 경제뿐 아니라 학교·주거·교통·"
            "복지시설 같은 행정 서비스에도 영향을 준다. 수원시 인구가 실제로 줄고 있는지 2015~2025년의 흐름으로 확인하고, "
            "연도에 따른 총인구수를 예측하는 모델을 만들어 보았다.")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("분석 질문")
        st.markdown("""
| 단계 | 내용 |
|---|---|
| 궁금증 | 수원시의 인구는 실제로 감소하고 있을까? |
| 질문 | 최근 10년 동안 수원시의 인구는 어떻게 변했을까? |
| **분석 질문** | **2015~2025년 수원시의 연도별 인구 데이터로, 연도를 이용해 총인구수를 예측할 수 있을까?** |
""")
    with right:
        st.subheader("머신러닝 문제로 바꾸기")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 정답 y | 총인구수 (명) |
| 특성 X | 연도 |
| 문제 종류 | ✅ **회귀** (정답이 숫자) |
| 사용자 | 수원시 시민, 행정·도시계획 담당자 |
""")

# ───────────────────────── ② 데이터 ─────────────────────────
with tabs[1]:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("수원시 연도별 인구 (전처리 후)")
        t = sw[["총인구수", "남자인구수", "여자인구수"]].copy()
        t["인구증가수"] = t["총인구수"].diff()
        t["성비 (여자 100명당 남자)"] = (t["남자인구수"] / t["여자인구수"] * 100).round(1)
        st.dataframe(t.style.format({"총인구수": "{:,}", "남자인구수": "{:,}", "여자인구수": "{:,}",
                                     "인구증가수": "{:+,.0f}", "성비 (여자 100명당 남자)": "{:.1f}"}),
                     use_container_width=True, height=430)
    with right:
        st.subheader("데이터 출처")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 데이터 | 행정구역(시군구)별 성별 인구수 |
| 출처 | 행정안전부 「주민등록인구현황」 (KOSIS) |
| 지역 | 경기도, 수원시 |
| 기간 | 2015 ~ 2025년 (11년) |
| 형식 | CSV · cp949 · 공공누리 |
""")
        st.subheader("전처리 기록")
        st.markdown("""
| 문제 | 처리 |
|---|---|
| 연도마다 [총·남·여] 3칸 반복 | 연도별 한 행으로 세로 변환 |
| 경기도·수원시 행이 함께 있음 | 수원시 행 추출 (모델) |
| 결측치·중복 | 없음 |
| 파생 변수 | 인구증가수 = 전년 대비 증감 |
""")

# ───────────────────────── ③ 탐색 ─────────────────────────
with tabs[2]:
    st.subheader("수원시 총인구 추이")
    fig = go.Figure()
    fig.add_scatter(x=sw.index, y=sw["총인구수"], mode="lines+markers+text", line=dict(color=BLUE, width=3),
                    text=[man(v) for v in sw["총인구수"]], textposition="top center", textfont=dict(size=10),
                    hovertemplate="%{x}년: %{y:,}명<extra></extra>", name="총인구")
    lo_y = sw["총인구수"].idxmin()
    fig.add_annotation(x=peak_y, y=sw.loc[peak_y, "총인구수"], text=f"최고 {peak_y}년", ay=-45, arrowcolor=RED,
                       font=dict(color=RED))
    fig.add_annotation(x=lo_y, y=sw.loc[lo_y, "총인구수"], text=f"최저 {lo_y}년", ay=40, arrowcolor=GREEN,
                       font=dict(color=GREEN))
    fig.update_layout(height=400, yaxis_title="총인구수 (명)", margin=dict(t=30), showlegend=False,
                      yaxis=dict(range=[sw["총인구수"].min() - 8000, sw["총인구수"].max() + 8000]))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("※ 세로축이 118만~121만 명 구간만 보여 줘서 변화가 커 보인다. 10년 동안 움직인 폭은 최대 약 1.9만 명(1.6%)이다.")

    left, right = st.columns(2)
    with left:
        st.markdown("**전년 대비 인구 증감 (인구증가수)**")
        inc = sw["총인구수"].diff().dropna()
        fig = go.Figure(go.Bar(x=inc.index, y=inc, marker_color=[GREEN if v > 0 else RED for v in inc],
                               text=[f"{v:+,.0f}" for v in inc], textposition="outside"))
        fig.update_layout(height=360, yaxis_title="명", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**수원시 vs 경기도 (2015년 = 100)**")
        fig = go.Figure()
        for d, nm, col in [(sw, "수원시", BLUE), (gg, "경기도", "#f97316")]:
            fig.add_scatter(x=d.index, y=d["총인구수"] / d.loc[2015, "총인구수"] * 100, mode="lines+markers",
                            name=nm, line=dict(color=col, width=3))
        fig.update_layout(height=360, yaxis_title="2015년 대비 지수", margin=dict(t=20),
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

    share = sw["총인구수"] / gg["총인구수"] * 100
    st.markdown("""
| | 발견 | 모델링에 주는 의미 |
|---|---|---|
| ① | 수원시 인구는 2017년 최고, 2021년 최저 → 올랐다 내렸다 다시 오르고 내림 | 매년 일정하게 변하지 않아 직선 모델에 한계 |
| ② | 2018\\~2021년 4년 연속 감소, 2022\\~2023년 반등, 2024\\~2025년 다시 감소 | 연도 하나로는 이런 굴곡을 설명하기 어려움 |
| ③ | 같은 기간 경기도는 꾸준히 증가 (+9.6%) | 수원시 감소는 '전국적 인구 감소'만의 문제가 아니라 지역 간 이동 영향일 수 있음 |
| ④ | 연도와 총인구수의 상관계수는 −0.19로 약함 | 연도만으로 총인구를 정확히 예측하기 어려움 |
""")
    st.info(f"**경기도 속 수원시 비중**: {share.loc[2015]:.2f}% (2015) → {share.loc[2025]:.2f}% (2025). "
            "경기도 인구는 늘었지만 수원시는 제자리여서, 경기도 안에서 수원시가 차지하는 비중은 꾸준히 줄었다.")

# ───────────────────────── ④ 모델 평가 ─────────────────────────
with tabs[3]:
    region = st.radio("지역", REGIONS, horizontal=True, key="eval_region")
    rows = []
    for c in catalog:
        s = c["by_region"][region]
        rows.append({"모델": c["name"], "R² (무작위 30% 평가)": s["r2_split"], "MAE (명)": s["mae_split"],
                     "3겹 교차검증 R²": s["cv_r2"], "시간순 평가 MAE (명)": s["mae_time"],
                     "출처": "노트북" if c["in_notebook"] else "웹앱 추가"})
    score = pd.DataFrame(rows)
    st.dataframe(score.style.format({"R² (무작위 30% 평가)": "{:.3f}", "MAE (명)": "{:,.0f}", "3겹 교차검증 R²": "{:.3f}",
                                     "시간순 평가 MAE (명)": "{:,.0f}"})
                 .highlight_min(subset=["MAE (명)", "시간순 평가 MAE (명)"], color="#bbf7d0"),
                 hide_index=True, use_container_width=True)
    st.caption("**무작위 30% 평가·3겹 교차검증** = 노트북 방식 (train_test_split·KFold, random_state=42). "
               "**시간순 평가** = 2015\\~2021년으로 학습 → 2022\\~2025년을 처음 보는 미래처럼 예측.")

    d = pop[pop["지역"] == region].set_index("연도")
    fig = go.Figure()
    fig.add_scatter(x=d.index, y=d["총인구수"], mode="lines+markers", name="실제", line=dict(color="black", width=3))
    for c in catalog:
        fig.add_scatter(x=list(range(2022, 2026)), y=c["by_region"][region]["test_pred"], mode="lines+markers",
                        name=c["name"], line=dict(color=MODEL_COLORS[c["key"]], dash="dot"))
    fig.add_vrect(x0=2021.5, x1=2025.5, fillcolor="#e2e8f0", opacity=0.4, line_width=0,
                  annotation_text="시간순 평가 구간", annotation_position="top left")
    fig.update_layout(height=420, yaxis_title="총인구수 (명)", margin=dict(t=20), legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.warning("**왜 R²가 −5, −6처럼 음수일까?** 평가용 데이터가 4개뿐이고 그 4개 연도의 인구 차이가 아주 작아서, "
                   "몇천 명만 틀려도 R²가 크게 음수가 된다. 수원시처럼 변화가 작은 자료에서는 R²보다 **MAE(평균 몇 명 틀렸나)**가 "
                   "더 읽기 쉽다. 수원시의 경우 시간순 평가에서도 단순 평균(기준선)보다 나은 모델이 없었다.")
    with right:
        st.error("**'남자인구수 포함' 교차검증 0.940은 데이터 누수** — 총인구수 = 남자인구수 + 여자인구수이므로 남자인구수를 "
                 "특성으로 넣으면 정답의 절반을 미리 알려 주는 셈이다. 미래의 남자인구수도 모르기 때문에 예측에 쓸 수 없다. "
                 "팀이 이 특성을 최종 모델에 넣지 않은 것은 올바른 판단이다.")

# ───────────────────────── ⑤ 미래 예측 ─────────────────────────
with tabs[4]:
    st.subheader("🔮 미래 인구 예측해 보기")
    ctrl, out = st.columns([1, 3])
    with ctrl:
        region = st.radio("지역", REGIONS, key="pred_region")
        mkey = st.selectbox("모델 선택 (joblib 파일 교체)", list(names), index=1, format_func=lambda k: names[k])
        year = st.slider("예측할 연도", 2026, 2035, 2026)
        st.caption(f"📂 `models/{mkey}.joblib` → `['{region}']`")
        st.markdown(f"**모델 설명** — {cat[mkey]['desc']}")
    d = pop[pop["지역"] == region].set_index("연도")
    yrs = np.arange(2015, max(year, 2030) + 1)
    pred = predict(mkey, region, yrs)
    p = float(predict(mkey, region, [year])[0])
    mae = cat[mkey]["by_region"][region]["mae_time"]
    with out:
        a, b, c = st.columns(3)
        a.metric(f"{year}년 {region} 예측", f"{p:,.0f}명")
        b.metric("2025년 대비", f"{p - d.loc[2025, '총인구수']:+,.0f}명")
        c.metric("예상 범위 (± 시간순 MAE)", f"{(p - mae) / 10000:,.1f} ~ {(p + mae) / 10000:,.1f}만 명")
        fig = go.Figure()
        fig.add_scatter(x=d.index, y=d["총인구수"], mode="lines+markers", name="실제", line=dict(color="black", width=3))
        fig.add_scatter(x=yrs, y=pred, mode="lines", name=names[mkey], line=dict(color=MODEL_COLORS[mkey], width=2.5))
        fig.add_scatter(x=[year], y=[p], mode="markers", marker=dict(size=14, symbol="star", color=MODEL_COLORS[mkey]),
                        showlegend=False)
        fig.add_vrect(x0=2025.5, x1=yrs[-1] + 0.5, fillcolor="#e2e8f0", opacity=0.4, line_width=0,
                      annotation_text="미래 (외삽)", annotation_position="top left")
        fig.update_layout(height=420, yaxis_title="총인구수 (명)", margin=dict(t=20), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**{region} 2026\\~2030년 모델별 예측 (명)**")
    tbl = pd.DataFrame({c["name"]: c["by_region"][region]["future"] for c in catalog}).T
    st.dataframe(tbl.style.format("{:,.0f}"), use_container_width=True)
    st.caption("결정 트리와 최근 값 모델은 데이터 밖(미래) 연도를 넣어도 마지막 값만 반복한다. "
               "2차 다항 회귀는 곡선이 계속 꺾여 내려가므로 먼 미래일수록 믿기 어렵다.")

# ───────────────────────── ⑥ 결론·한계 ─────────────────────────
with tabs[5]:
    st.subheader("결론")
    st.success(f"**수원시 인구는 '꾸준히 감소'했다기보다 제자리에서 오르내렸다.** 2015년보다 2025년이 {sw.loc[2025, '총인구수'] - sw.loc[2015, '총인구수']:+,}명 "
               f"많지만, 최고점인 {peak_y}년보다는 {sw['총인구수'].max() - sw.loc[2025, '총인구수']:,}명 적다. 같은 기간 경기도는 "
               f"{(gg.loc[2025, '총인구수'] / gg.loc[2015, '총인구수'] - 1) * 100:.1f}% 늘어, 경기도 안에서 수원시의 비중은 줄었다.\n\n"
               "**예측**: 연도 하나만 쓰는 선형 회귀는 R² −6.34로 평균값 예측(기준선)보다 나은 결과를 보여 주지 못했다. "
               "수원시처럼 오르내림이 있는 작은 변화는 연도만으로 예측하기 어렵고, 출생·사망·전입·전출 같은 정보가 필요하다.")
    st.subheader("한계와 보완 방법")
    st.markdown("""
| 구분 | 한계 | 보완 방법 |
|---|---|---|
| 데이터 | 11개 연도뿐, 연령별·출생 정보 없음 | 더 긴 기간, 연령별 인구·출생률·전입전출 자료 추가 |
| 모델 | 연도 하나로만 예측 → 인구 이동·재개발 등 반영 못 함 | 인구 변화 요인을 특성으로 추가, 여러 모델 비교 |
| 평가 | 평가 데이터 4개, 무작위로 섞어 과거·미래가 뒤섞임 | 시간 순서로 나눠 평가, MAE 중심으로 해석 |
| 해석 | 주민등록 기준이라 실제 거주 인구와 다를 수 있음 | 생활인구 등 다른 통계와 비교 |
""")
    st.subheader("윤리 점검")
    st.markdown("✅ 개인 식별 정보 없음 (시·도 단위 통계) · ✅ 출처·이용 조건 준수 · ✅ 결과를 과장하지 않고 한계 제시 · "
                "⚠️ 인구 감소를 특정 집단(세대·성별) 탓으로 해석하지 않기")
    st.subheader("다음에 해 보고 싶은 것")
    st.write("연령별 인구수와 출생률 자료를 추가해 수원시의 고령화·저출산 정도를 자세히 보고, 다른 시·군과 비교해 지역별 차이를 분석하고 싶다.")
    st.caption("참고: 행정안전부, 「주민등록인구현황」, 행정구역(시군구)별 성별 인구수, KOSIS 국가통계포털, 2026.10.29 내려받음.")
