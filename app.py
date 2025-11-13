import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

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

/* 라벨은 네온 라임, 입력 텍스트는 진한 회색 */
label {
    color: #23EB96 !important;
}

/* 인풋/셀렉트/텍스트영역 안 글자색 */
input, textarea, select, subheader {
    color: black !important;
}

/* Streamlit selectbox 내부 텍스트 색 강제 */
div[data-baseweb="select"] * {
    color: black !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("<h1>🧥 코트 사이즈 추천 (CJ ONSTYLE Edition)</h1>", unsafe_allow_html=True)
st.write(" ")

# ================================
# 2. 데이터 & 모델 학습
# ================================

ARM_CSV_PATH  = "암홀둘레.csv"   # 암홀둘레가 들어있는 CSV
KNEE_CSV_PATH = "무릎높이.csv"      # 무릎높이가 들어있는 CSV
SHO_CSV_PATH  = "어깨너비.csv"  # 어깨너비가 들어있는 CSV

FEATURE_COLS = ["성별", "나이", "키", "몸무게", "허리둘레", "발사이즈"]
TARGET_ARM = "암홀둘레"
TARGET_KNEE = "무릎높이"
TARGET_SHO = "어깨너비"


def train_model_from_csv(csv_path: str, target_col: str):
    """단일 CSV에서 하나의 회귀모델 학습"""
    df = pd.read_csv(csv_path)

    # 성별 인코딩
    data = df.copy()
    sex_map = {"남": 1, "여": 0}
    data["성별"] = data["성별"].map(sex_map)
    data = data.dropna(subset=["성별"])

    X = data[FEATURE_COLS]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 필요하면 여기서 R2 등 찍어서 로그 보고 싶으면 계산 가능
    # from sklearn.metrics import r2_score
    # r2 = r2_score(y_test, model.predict(X_test))
    # print(target_col, "R2:", r2)

    return model


@st.cache_resource
def load_data_and_train():
    """암홀/무릎/어깨 3개 CSV에서 각각 모델 학습"""
    arm_model = train_model_from_csv(ARM_CSV_PATH, TARGET_ARM)
    knee_model = train_model_from_csv(KNEE_CSV_PATH, TARGET_KNEE)
    sho_model = train_model_from_csv(SHO_CSV_PATH, TARGET_SHO)
    return arm_model, knee_model, sho_model


try:
    arm_model, knee_model, sho_model = load_data_and_train()
except Exception as e:
    st.error(f"데이터 로드/모델 학습 중 오류가 발생했습니다: {e}")
    st.stop()

# ================================
# 3. SKU 정의
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

STANDARD_ALLOWANCE = (25, 15)  # (암홀, 어깨)
LENGTH_WEIGHT = 0.2
ARM_WEIGHT    = 0.5
SHO_WEIGHT    = 0.3

def get_sku_table():
    rows = []
    for s in size_order:
        for ln in length_order:
            rows.append({
                "Size": s,
                "Length": ln,
                "Armhole(mm)": armhole_spec[s],
                "Shoulder(mm)": shoulder_spec[s],
                "Coat length(mm)": length_spec[ln],
            })
    return pd.DataFrame(rows)

sku_df = get_sku_table()

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

    return best  # (size, length)

# ================================
# 4. 레이아웃 (상품 카드 + SKU 표)
# ================================
left, right = st.columns([1.1, 1.4])

with left:
    #st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.subheader("상품 정보")
    # 👉 GitHub 레포에 있는 실제 이미지 파일 사용
    # app.py와 같은 폴더에 "Gemini_Generated_Image_u57y6xu57y6xu57y.png" 가 있다고 가정
    st.image(
        "Gemini_Generated_Image_u57y6xu57y6xu57y.png",
        #caption="모던 유니섹스 코트",
        use_column_width=True
    )
    st.markdown("**모던 유니섹스 코트 — CJ ONSTYLE Edition**")
    st.markdown("₩ 249,000")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    #st.markdown("<div class='white-card'>", unsafe_allow_html=True)
    st.subheader("코트 SKU (사이즈 × 기장 = 21종)")
    st.dataframe(sku_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.write(" ")
st.markdown("---")

# ================================
# 5. 입력 폼
# ================================
st.subheader("신체 치수 입력")

with st.form("input_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        sex = st.selectbox("성별", ["여", "남"])
        age = st.number_input("나이 (세)", 10, 100, 28)
    with c2:
        height_cm = st.number_input("키 (cm)", 140.0, 210.0, 165.0)
        weight_kg = st.number_input("몸무게 (kg)", 30.0, 150.0, 55.0)
    with c3:
        waist_in = st.number_input("허리 사이즈 (인치)", 20.0, 60.0, 28.0, step=0.5)
        foot_mm = st.number_input("발사이즈 (mm)", 210, 300, 245)

    submitted = st.form_submit_button("추천 결과 보기")

# ================================
# 6. 예측 + 추천
# ================================
if submitted:
    sex_encoded = 1 if sex == "남" else 0
    height_mm = height_cm * 10.0
    waist_mm = waist_in * 25.4

    X = pd.DataFrame([{
        "성별": sex_encoded,
        "나이": float(age),
        "키": float(height_mm),
        "몸무게": float(weight_kg),
        "허리둘레": float(waist_mm),
        "발사이즈": float(foot_mm),
    }])

    # 예측
    pred_arm_mm  = float(arm_model.predict(X)[0])
    pred_knee_mm = float(knee_model.predict(X)[0])
    pred_sho_mm  = float(sho_model.predict(X)[0])

    pred_arm_cm  = round(pred_arm_mm / 10.0, 2)
    pred_knee_cm = round(pred_knee_mm / 10.0, 2)
    pred_sho_cm  = round(pred_sho_mm / 10.0, 2)

    st.markdown("<h3>📏 예측된 신체 치수</h3>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("암홀둘레", f"{pred_arm_cm} cm")
    m2.metric("무릎높이", f"{pred_knee_cm} cm")
    m3.metric("어깨너비", f"{pred_sho_cm} cm")

    size, length_name = recommend_standard(pred_arm_mm, pred_knee_mm, pred_sho_mm)

    st.markdown("<h3>✨ 추천 코트 사이즈 (Standard Fit)</h3>", unsafe_allow_html=True)
    st.success(f"추천 사이즈: **{size} / {length_name.capitalize()}**")

else:
    st.info("신체 치수를 입력한 뒤 **'추천 결과 보기'** 버튼을 눌러주세요.")



