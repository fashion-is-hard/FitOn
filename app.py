import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ================================
# 0. 페이지 설정
# ================================
st.set_page_config(
    page_title="코트 사이즈 추천",
    page_icon="🧥",
    layout="wide"
)

# ================================
# 1. CUSTOM CSS (CJ ONSTYLE 스타일)
# ================================
CUSTOM_CSS = """
<style>
/* 전체 배경: 퍼플 그라데이션 */
.main {
    background: linear-gradient(135deg, #640FAF 0%, #7323B9 30%, #913CD2 70%, #A055D7 100%);
}

/* 제목 컬러: 네온 라임 */
h1, h2, h3 {
    color: #23EB96 !important;
    font-family: "Pretendard", "Noto Sans KR", sans-serif;
    font-weight: 700;
}

/* 일반 텍스트 */
body, p, span, div {
    font-family: "Noto Sans KR", sans-serif;
    color: white;
}

/* 카드 형태의 white container */
.white-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    color: #333;
}

/* 버튼 스타일: 퍼플 → 네온 라임 그라데이션 */
.stButton>button {
    background: linear-gradient(90deg, #7323B9, #913CD2, #23EB96);
    color: white;
    border: none;
    padding: 0.7rem 1.8rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
}
.stButton>button:hover {
    opacity: 0.9;
}

/* 테이블 여백 및 폰트 */
.dataframe {
    font-size: 0.9rem;
}

/* 입력 위젯 텍스트 색상 */
label, .stTextInput, .stNumberInput, .stSelectbox {
    color: white !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ================================
# 타이틀
# ================================
st.markdown("<h1>🧥 코트 사이즈 추천 (CJ ONSTYLE Edition)</h1>", unsafe_allow_html=True)
st.write(" ")

# ================================
# 2. SKU 정의
# ================================
size_order = ["XXS", "XS", "S", "M", "L", "XL", "XXL"]
length_order = ["short", "medium", "long"]

armhole_spec = {
    "XXS": 410, "XS": 430, "S": 450, "M": 470,
    "L": 490, "XL": 510, "XXL": 530
}
shoulder_spec = {
    "XXS": 380, "XS": 395, "S": 410, "M": 425,
    "L": 440, "XL": 455, "XXL": 470
}
length_spec = {
    "short": 800,
    "medium": 950,
    "long": 1100
}

STANDARD_ALLOWANCE = (25, 15)
LENGTH_WEIGHT = 0.2
ARM_WEIGHT    = 0.5
SHO_WEIGHT    = 0.3

# ================================
# 모델 로드
# ================================
@st.cache_resource
def load_models():
    with open("armhole_model.pkl", "rb") as f:
        arm_model_ = pickle.load(f)
    with open("knee_model.pkl", "rb") as f:
        knee_model_ = pickle.load(f)
    with open("shoulder_model.pkl", "rb") as f:
        sho_model_ = pickle.load(f)
    return arm_model_, knee_model_, sho_model_

arm_model, knee_model, sho_model = load_models()

# ================================
# SKU 테이블 생성
# ================================
def get_sku_table():
    rows = []
    for s in size_order:
        for ln in length_order:
            rows.append({
                "Size": s,
                "Length": ln,
                "Armhole(mm)": armhole_spec[s],
                "Shoulder(mm)": shoulder_spec[s],
                "Coat length(mm)": length_spec[ln]
            })
    return pd.DataFrame(rows)

sku_df = get_sku_table()

# ================================
# 추천 함수
# ================================
def recommend_standard(pred_arm_mm, pred_knee_mm, pred_sho_mm):
    ah_allow, sh_allow = STANDARD_ALLOWANCE

    need_arm = pred_arm_mm + ah_allow
    need_sho = pred_sho_mm + sh_allow
    target_len = pred_knee_mm

    best = None
    best_cost = float("inf")

    for s in size_order:
        coat_arm = armhole_spec[s]
        coat_sho = shoulder_spec[s]

        for Lname, Lmm in length_spec.items():
            cost = (
                ARM_WEIGHT * abs(coat_arm - need_arm) +
                SHO_WEIGHT * abs(coat_sho - need_sho) +
                LENGTH_WEIGHT * abs(Lmm - target_len)
            )
            if cost < best_cost:
                best_cost = cost
                best = (s, Lname)

    return best


# ================================
# 레이아웃 구성
# ================================
left, right = st.columns([1.1, 1.4])

with left:
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.subheader("상품 정보")
    st.image("https://placehold.co/600x800/7323B9/FFFFFF?text=COAT+IMAGE", caption="(이미지 교체 가능)")
    st.markdown("**모던 유니섹스 코트 — CJ ONSTYLE Edition**")
    st.markdown("₩ 249,000")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.subheader("코트 SKU (21종)")
    st.dataframe(sku_df, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write(" ")
st.markdown("---")

# ================================
# 입력 폼
# ================================
st.subheader("신체 치수 입력")

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("성별", ["여", "남"])
        age = st.number_input("나이", min_value=10, max_value=100, value=28)
    with c2:
        height_cm = st.number_input("키 (cm)", 140.0, 210.0, 165.0)
        weight_kg = st.number_input("몸무게 (kg)", 30.0, 150.0, 55.0)
    with c3:
        waist_in = st.number_input("허리둘레 (inch)", 20.0, 60.0, 28.0)
        foot_mm = st.number_input("발 사이즈 (mm)", 210, 300, 245)

    submitted = st.form_submit_button("추천 결과 보기")

# ================================
# 예측 + 추천
# ================================
if submitted:
    sex_encoded = 1 if sex == "남" else 0
    height_mm = height_cm * 10
    waist_mm = waist_in * 25.4

    X = pd.DataFrame([{
        "성별": sex_encoded,
        "나이": age,
        "키": height_mm,
        "몸무게": weight_kg,
        "허리둘레": waist_mm,
        "발사이즈": foot_mm
    }])

    pred_arm = float(arm_model.predict(X)[0])
    pred_knee = float(knee_model.predict(X)[0])
    pred_sho = float(sho_model.predict(X)[0])

    pred_arm_cm = pred_arm / 10
    pred_knee_cm = pred_knee / 10
    pred_sho_cm = pred_sho / 10

    # 결과 표시
    st.markdown("<h3>📏 예측된 신체 치수</h3>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("암홀둘레", f"{pred_arm_cm:.2f} cm")
    m2.metric("무릎높이", f"{pred_knee_cm:.2f} cm")
    m3.metric("어깨너비", f"{pred_sho_cm:.2f} cm")

    size, length = recommend_standard(pred_arm, pred_knee, pred_sho)

    st.markdown("<h3>✨ 추천 코트 사이즈 (Standard Fit)</h3>", unsafe_allow_html=True)
    st.success(f"**{size} / {length.capitalize()}** 사이즈가 가장 잘 맞아요!")


