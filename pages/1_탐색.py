import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 데이터 탐색")
st.caption("그래프와 표를 이용하여 데이터의 특징을 살펴봅니다.")


# =====================================
# 2. 데이터 불러오기
# =====================================

DATA_URL = "https://raw.githubusercontent.com/minseo-sudo/data/refs/heads/main/stroke.csv"

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
# 3. 나이와 평균 혈당 히스토그램
# =====================================

st.header("1. 나이와 평균 혈당의 분포")

st.write("각 변수의 데이터가 어느 구간에 많이 분포하는지 살펴봅니다.")

col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수"
        },
        color_discrete_sequence=["#4C78A8"]
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_age, use_container_width=True)


with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수"
        },
        color_discrete_sequence=["#F58518"]
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수"
    )

    st.plotly_chart(fig_glucose, use_container_width=True)


st.divider()


# =====================================
# 4. 뇌졸중 유무에 따른 상자그림
# =====================================

st.header("2. 뇌졸중 유무에 따른 비교")

box_df = df.copy()

box_df["뇌졸중 여부"] = box_df["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음"
})

col1, col2 = st.columns(2)

with col1:
    fig_age_box = px.box(
        box_df,
        x="뇌졸중 여부",
        y="age",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 나이 비교",
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_age_box.update_layout(showlegend=False)

    st.plotly_chart(fig_age_box, use_container_width=True)


with col2:
    fig_glucose_box = px.box(
        box_df,
        x="뇌졸중 여부",
        y="avg_glucose_level",
        color="뇌졸중 여부",
        title="뇌졸중 여부에 따른 평균 혈당 비교",
        color_discrete_map={
            "뇌졸중 없음": "#4C78A8",
            "뇌졸중 있음": "#E45756"
        }
    )

    fig_glucose_box.update_layout(showlegend=False)

    st.plotly_chart(fig_glucose_box, use_container_width=True)


# 그룹별 평균값
mean_table = (
    box_df.groupby("뇌졸중 여부")
    .agg(
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean")
    )
    .reindex(["뇌졸중 없음", "뇌졸중 있음"])
    .reset_index()
)

mean_table = mean_table.rename(columns={
    "뇌졸중 여부": "그룹",
    "평균_나이": "평균 나이",
    "평균_혈당": "평균 혈당"
})

mean_table[["평균 나이", "평균 혈당"]] = mean_table[
    ["평균 나이", "평균 혈당"]
].round(2)

st.subheader("그룹별 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================
# 5. 고혈압 유무에 따른 뇌졸중 비율
# =====================================

st.header("3. 고혈압 유무에 따른 뇌졸중 비율")

hypertension_table = (
    df.groupby("hypertension")
    .agg(
        전체_인원=("stroke", "count"),
        뇌졸중_인원=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean")
    )
    .reset_index()
)

hypertension_table["고혈압 여부"] = hypertension_table[
    "hypertension"
].map({
    0: "고혈압 없음",
    1: "고혈압 있음"
})

hypertension_table["뇌졸중_비율"] *= 100

fig_hypertension = px.bar(
    hypertension_table,
    x="고혈압 여부",
    y="뇌졸중_비율",
    text=hypertension_table["뇌졸중_비율"].apply(
        lambda x: f"{x:.2f}%"
    ),
    title="고혈압 유무에 따른 뇌졸중 비율",
    color="고혈압 여부",
    color_discrete_map={
        "고혈압 없음": "#4C78A8",
        "고혈압 있음": "#E45756"
    },
    labels={
        "뇌졸중_비율": "뇌졸중 비율(%)"
    }
)

fig_hypertension.update_layout(showlegend=False)
fig_hypertension.update_traces(textposition="outside")

st.plotly_chart(fig_hypertension, use_container_width=True)

hypertension_show = hypertension_table[
    ["고혈압 여부", "전체_인원", "뇌졸중_인원", "뇌졸중_비율"]
].copy()

hypertension_show.columns = [
    "고혈압 여부",
    "전체 인원",
    "뇌졸중 인원",
    "뇌졸중 비율(%)"
]

hypertension_show["뇌졸중 비율(%)"] = hypertension_show[
    "뇌졸중 비율(%)"
].round(2)

st.dataframe(
    hypertension_show,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================
# 6. 심장병 유무에 따른 뇌졸중 비율
# =====================================

st.header("4. 심장병 유무에 따른 뇌졸중 비율")

heart_table = (
    df.groupby("heart_disease")
    .agg(
        전체_인원=("stroke", "count"),
        뇌졸중_인원=("stroke", "sum"),
        뇌졸중_비율=("stroke", "mean")
    )
    .reset_index()
)

heart_table["심장병 여부"] = heart_table[
    "heart_disease"
].map({
    0: "심장병 없음",
    1: "심장병 있음"
})

heart_table["뇌졸중_비율"] *= 100

fig_heart = px.bar(
    heart_table,
    x="심장병 여부",
    y="뇌졸중_비율",
    text=heart_table["뇌졸중_비율"].apply(
        lambda x: f"{x:.2f}%"
    ),
    title="심장병 유무에 따른 뇌졸중 비율",
    color="심장병 여부",
    color_discrete_map={
        "심장병 없음": "#4C78A8",
        "심장병 있음": "#E45756"
    },
    labels={
        "뇌졸중_비율": "뇌졸중 비율(%)"
    }
)

fig_heart.update_layout(showlegend=False)
fig_heart.update_traces(textposition="outside")

st.plotly_chart(fig_heart, use_container_width=True)

heart_show = heart_table[
    ["심장병 여부", "전체_인원", "뇌졸중_인원", "뇌졸중_비율"]
].copy()

heart_show.columns = [
    "심장병 여부",
    "전체 인원",
    "뇌졸중 인원",
    "뇌졸중 비율(%)"
]

heart_show["뇌졸중 비율(%)"] = heart_show[
    "뇌졸중 비율(%)"
].round(2)

st.dataframe(
    heart_show,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================
# 7. BMI 결측자와 전체 비교
# =====================================

st.header("5. BMI 빈 값과 뇌졸중 비율")

missing_bmi_df = df[df["bmi"].isna()]

total_ratio = df["stroke"].mean() * 100

if len(missing_bmi_df) > 0:
    missing_ratio = missing_bmi_df["stroke"].mean() * 100
else:
    missing_ratio = 0

bmi_table = pd.DataFrame({
    "구분": [
        "전체 사람",
        "BMI 빈 값이 있는 사람"
    ],
    "사람 수": [
        len(df),
        len(missing_bmi_df)
    ],
    "뇌졸중 발생 인원": [
        int(df["stroke"].sum()),
        int(missing_bmi_df["stroke"].sum())
    ],
    "뇌졸중 비율(%)": [
        round(total_ratio, 2),
        round(missing_ratio, 2)
    ]
})

st.write(
    f"BMI가 비어 있는 사람 수: {len(missing_bmi_df):,}명"
)

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True
)


st.divider()


# =====================================
# 8. 흡연 상태별 사람 수
# =====================================

st.header("6. 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

smoking_table["흡연 상태"] = smoking_table[
    "흡연 상태"
].fillna("빈 값")

st.dataframe(
    smoking_table,
    use_container_width=True,
    hide_index=True
)


# =====================================
# 9. 마무리
# =====================================

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 탐색 페이지")
# =====================================
# 9. 마무리
# =====================================

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 데이터 탐색 페이지")

st.caption(
    "※ 그룹별 뇌졸중 비율은 각 그룹의 전체 인원 중 "
    "stroke 값이 1인 사람의 비율입니다."
)
