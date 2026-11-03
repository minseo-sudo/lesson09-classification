
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
    page_icon="📋",
    layout="wide"
)

st.title("📋 모델 채점표")
st.caption(
    "테스트 데이터를 이용하여 분류 모델의 예측 결과를 평가합니다."
)


# =====================================
# 2. 데이터 불러오기
# =====================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "minseo-sudo/data/refs/heads/main/stroke.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    df = load_data()

except Exception as error:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(error)
    st.stop()


# =====================================
# 3. 사용할 속성 선택
# =====================================

st.header("1. 예측에 사용할 속성")

feature_names = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}

selected_features = st.multiselect(
    "모델에 사용할 속성을 선택하세요.",
    options=list(feature_names.keys()),
    default=[
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease"
    ],
    format_func=lambda x: feature_names[x]
)

if len(selected_features) < 2:
    st.warning(
        "예측에 사용할 속성을 두 개 이상 선택해 주세요."
    )
    st.stop()


# =====================================
# 4. 데이터 정렬 및 분리
# =====================================

st.header("2. 학습용 데이터와 테스트용 데이터")

required_columns = [
    "id",
    "stroke",
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease"
] + selected_features

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(
        "데이터에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# 번호 순으로 정렬
df = df.sort_values("id").reset_index(drop=True)

if df["stroke"].isna().any():
    st.error("stroke 열에 결측값이 있습니다.")
    st.stop()

if not set(df["stroke"].unique()).issubset({0, 1}):
    st.error("stroke 열에는 0과 1만 있어야 합니다.")
    st.stop()


# 10명씩 묶어 앞의 3명을 테스트용으로 분리
test_mask = np.arange(len(df)) % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int).copy()
y_test = test_df["stroke"].astype(int).copy()


if len(train_df) == 0 or len(test_df) == 0:
    st.error("학습 데이터 또는 테스트 데이터가 없습니다.")
    st.stop()

if set(y_train.unique()) != {0, 1}:
    st.error("학습 데이터에 0과 1이 모두 있어야 합니다.")
    st.stop()

if set(y_test.unique()) != {0, 1}:
    st.error("테스트 데이터에 0과 1이 모두 있어야 합니다.")
    st.stop()


# 데이터 인원수
test_total = len(test_df)
test_positive = int((y_test == 1).sum())

st.write(
    f"**테스트용 사람 수: {test_total:,}명 | "
    f"그중 실제 뇌졸중인 사람: {test_positive:,}명**"
)

col1, col2 = st.columns(2)

with col1:
    st.metric("학습용 데이터", f"{len(train_df):,}명")

with col2:
    st.metric("테스트용 데이터", f"{len(test_df):,}명")

if test_total != 1533:
    st.warning(
        f"현재 테스트 데이터는 {test_total:,}명입니다. "
        "요청한 1,533명과 다릅니다. "
        "원본 데이터가 5,110명인지 확인해 주세요."
    )


# =====================================
# 5. BMI 결측값 처리
# =====================================

st.header("3. 결측값 처리")

if "bmi" in selected_features:

    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error(
            "학습 데이터의 BMI 중앙값을 계산할 수 없습니다."
        )
        st.stop()

    missing_train = int(X_train["bmi"].isna().sum())
    missing_test = int(X_test["bmi"].isna().sum())

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"체질량지수의 결측값을 학습 데이터의 중앙값 "
        f"{bmi_median:.2f}로 채웠습니다."
    )

    st.caption(
        f"학습용 결측값: {missing_train:,}개 / "
        f"테스트용 결측값: {missing_test:,}개"
    )

else:
    st.info(
        "체질량지수를 선택하지 않았으므로 "
        "BMI 결측값 처리를 하지 않았습니다."
    )


if X_train.isna().any().any() or X_test.isna().any().any():
    st.error(
        "선택한 속성에 결측값이 있습니다."
