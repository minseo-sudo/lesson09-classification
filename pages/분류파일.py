import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 분류 모델")
st.caption("여러 속성을 이용하여 뇌졸중 여부를 예측하는 분류 모델을 만들어 봅니다.")

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
# 3. 사용할 속성 선택
# =====================================

st.header("1. 예측에 사용할 속성 선택")

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
    st.warning("예측에 사용할 속성을 두 개 이상 선택해 주세요.")
    st.stop()

# =====================================
# 4. 데이터 정렬 및 분리
# =====================================

st.header("2. 학습용 데이터와 테스트용 데이터")

df = df.sort_values("id").reset_index(drop=True)

if df["stroke"].isna().any():
    st.error("stroke 열에 결측값이 있습니다.")
    st.stop()

test_mask = np.arange(len(df)) % 10 < 3

train_df = df.loc[~test_mask].copy()
test_df = df.loc[test_mask].copy()

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

if len(test_df) != 1533:
    st.warning(f"테스트 데이터가 {len(test_df):,}명입니다. 원본 데이터 행 수를 확인해 주세요.")

col1, col2 = st.columns(2)

col1.metric("학습 데이터", f"{len(train_df):,}명")
col2.metric("테스트 데이터", f"{len(test_df):,}명")

# =====================================
# 5. BMI 결측치 처리
# =====================================

st.header("3. 결측치 처리")

if "bmi" in selected_features:
    bmi_median = X_train["bmi"].median()

    if pd.isna(bmi_median):
        st.error("학습 데이터의 BMI 중앙값을 계산할 수 없습니다.")
        st.stop()

    missing_train = X_train["bmi"].isna().sum()
    missing_test = X_test["bmi"].isna().sum()

    X_train["bmi"] = X_train["bmi"].fillna(bmi_median)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_median)

    st.write(
        f"BMI 결측값을 학습 데이터 중앙값 "
        f"{bmi_median:.2f}로 채웠습니다."
    )
else:
    st.info("체질량지수를 선택하지 않아 BMI 결측치 처리를 하지 않았습니다.")

if X_train.isna().any().any() or X_test.isna().any().any():
    st.error("선택한 속성에 결측값이 있습니다.")
    st.stop()

X_train = X_train.astype(float)
X_test = X_test.astype(float)

# =====================================
# 6. 학습 데이터 크기 맞추기
# =====================================

st.header("4. 학습 데이터 크기 맞추기")

rng = np.random.RandomState(42)

counts = y_train.value_counts()
count_0 = int(counts.get(0, 0))
count_1 = int(counts.get(1, 0))

if count_0 == 0 or count_1 == 0:
    st.error("학습 데이터에 두 클래스가 모두 있어야 합니다.")
    st.stop()

majority_class = 0 if count_0 >= count_1 else 1
minority_class = 1 - majority_class

majority_indices = np.where(y_train.to_numpy() == majority_class)[0]
minority_indices = np.where(y_train.to_numpy() == minority_class)[0]

additional_indices = rng.choice(
    minority_indices,
    size=len(majority_indices) - len(minority_indices),
    replace=True
)

balanced_indices = np.concatenate([
    majority_indices,
    additional_indices
])

rng.shuffle(balanced_indices)

X_balanced = X_train.iloc[balanced_indices].reset_index(drop=True)
y_balanced = y_train.iloc[balanced_indices].reset_index(drop=True)

st.write("학습 데이터의 클래스별 인원")

st.dataframe(
    pd.DataFrame({
        "구분": ["뇌졸중 아님", "뇌졸중"],
        "크기 맞추기 전": [count_0, count_1],
        "크기 맞추기 후": [
            int((y_balanced == 0).sum()),
            int((y_balanced == 1).sum())
        ]
    }),
    use_container_width=True,
    hide_index=True
)

# =====================================
# 7. 로지스틱 회귀
# =====================================

st.header("5. 로지스틱 회귀(확률로 답하는 모델)")

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=2000, random_state=42))
])

logistic_model.fit(X_balanced, y_balanced)

logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

logistic_train_accuracy = accuracy_score(y_train, logistic_train_pred)
logistic_test_accuracy = accuracy_score(y_test, logistic_test_pred)

# =====================================
# 8. 의사결정트리
# =====================================

st.header("6. 의사결정트리(질문으로 답하는 모델)")

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42
)

tree_model.fit(X_balanced, y_balanced)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)

tree_train_accuracy = accuracy_score(y_train, tree_train_pred)
tree_test_accuracy = accuracy_score(y_test, tree_test_pred)

# =====================================
# 9. 기준 모델
# =====================================

st.header("7. 기준 모델")

baseline_class = int(y_train.mode().iloc[0])

baseline_train_accuracy = accuracy_score(
    y_train,
    np.full(len(y_train), baseline_class)
)

baseline_test_accuracy = accuracy_score(
    y_test,
    np.full(len(y_test), baseline_class)
)

# =====================================
# 10. 정확도 카드
# =====================================

st.header("8. 모델 정확도 비교")

def show_accuracy(title, train_accuracy, test_accuracy):
    st.subheader(title)

    st.metric(
        "테스트 데이터 정확도",
        f"{test_accuracy * 100:.2f}%"
    )

    col1, col2 = st.columns(2)

    col1.caption("훈련 데이터 정확도")
    col1.markdown(f"### {train_accuracy * 100:.2f}%")

    col2.caption("테스트 데이터 정확도")
    col2.markdown(f"### {test_accuracy * 100:.2f}%")

col1, col2, col3 = st.columns(3)

with col1:
    show_accuracy(
        "로지스틱 회귀",
        logistic_train_accuracy,
        logistic_test_accuracy
    )

with col2:
    show_accuracy(
        "의사결정트리",
        tree_train_accuracy,
        tree_test_accuracy
    )

with col3:
    show_accuracy(
        "기준 모델",
        baseline_train_accuracy,
        baseline_test_accuracy
    )

# =====================================
# 11. 산점도 축 선택
# =====================================

st.header("9. 예측 결과 산점도")

col1, col2 = st.columns(2)

with col1:
    x_feature = st.selectbox(
        "가로축",
        selected_features,
        format_func=lambda x: feature_names[x]
    )

with col2:
    y_options = [x for x in selected_features if x != x_feature]

    y_feature = st.selectbox(
        "세로축",
        y_options,
        format_func=lambda x: feature_names[x]
    )

fixed_values = {}

for feature in selected_features:
    if feature not in [x_feature, y_feature]:
        fixed_values[feature] = float(X_test[feature].median())

if fixed_values:
    st.write("다른 속성은 테스트 데이터의 중앙값으로 고정했습니다.")

    for feature, value in fixed_values.items():
        st.write(f"- {feature_names[feature]}: {value:.2f}")

# =====================================
# 12. 예측 영역 계산
# =====================================

x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_margin = max((x_max - x_min) * 0.05, 0.1)
y_margin = max((y_max - y_min) * 0.05, 0.1)

x_range = np.linspace(x_min - x_margin, x_max + x_margin, 180)
y_range = np.linspace(y_min - y_margin, y_max + y_margin, 180)

xx, yy = np.meshgrid(x_range, y_range)

grid_data = pd.DataFrame(
    np.tile(X_test.median().to_numpy(), (xx.size, 1)),
    columns=selected_features
)

for feature, value in fixed_values.items():
    grid_data[feature] = value

grid_data[x_feature] = xx.ravel()
grid_data[y_feature] = yy.ravel()

positive_index = list(
    logistic_model.named_steps["model"].classes_
).index(1)

grid_probability = logistic_model.predict_proba(
    grid_data
)[:, positive_index].reshape(xx.shape)

grid_tree_prediction = tree_model.predict(
    grid_data
).reshape(xx.shape)

# =====================================
# 13. Plotly 산점도
# =====================================

plot_df = X_test.copy()

plot_df["실제 뇌졸중 여부"] = y_test.map({
    0: "뇌졸중 아님",
    1: "뇌졸중"
}).to_numpy()

fig = go.Figure()

fig.add_trace(go.Contour(
    x=x_range,
    y=y_range,
    z=grid_tree_prediction,
    colorscale=[
        [0, "rgba(76,120,168,0.18)"],
        [0.499, "rgba(76,120,168,0.18)"],
        [0.5, "rgba(228,87,86,0.18)"],
        [1, "rgba(228,87,86,0.18)"]
    ],
    contours=dict(
        start=0,
        end=1,
        size=1,
        coloring="fill"
    ),
    showscale=False,
    hoverinfo="skip",
    name="의사결정트리 예측 영역"
))

fig.add_trace(go.Contour(
    x=x_range,
    y=y_range,
    z=grid_probability,
    contours=dict(
        start=0.5,
        end=0.5,
        size=1,
        coloring="none"
    ),
    line=dict(color="black", width=3),
    showscale=False,
    hoverinfo="skip",
    name="로지스틱 회귀 0.5 경계선"
))

for label, color in [
    ("뇌졸중 아님", "#4C78A8"),
    ("뇌졸중", "#E45756")
]:
    subset = plot_df[
        plot_df["실제 뇌졸중 여부"] == label
    ]

    fig.add_trace(go.Scatter(
        x=subset[x_feature],
        y=subset[y_feature],
        mode="markers",
        name=label,
        marker=dict(
            color=color,
            size=6,
            opacity=0.75
        )
    ))

fig.update_layout(
    title="테스트 데이터와 모델 예측 영역",
    xaxis_title=feature_names[x_feature],
    yaxis_title=feature_names[y_feature],
    height=650,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

if (
    np.nanmin(grid_probability) <= 0.5
    and np.nanmax(grid_probability) >= 0.5
):
    st.success("0.5 경계선이 그림 안에 있습니다.")
else:
    st.warning("0.5 경계선이 그림 범위 밖에 있습니다.")

# =====================================
# 14. 의사결정트리 가지 그림
# =====================================

st.header("10. 의사결정트리 질문 구조")

tree_structure = tree_model.tree_

dot_lines = [
    "digraph DecisionTree {",
    'graph [rankdir=TB, bgcolor="white"];',
    'node [shape=box, style="rounded,filled", fontname="Malgun Gothic"];',
    'edge [fontname="Malgun Gothic"];'
]

used_features = set()
leaf_count = 0
negative_leaf_count = 0

def get_node_counts(node_id):
    samples = int(tree_structure.n_node_samples[node_id])

    values = tree_structure.value[node_id][0]
    proportions = values / values.sum()

    positive_count = int(round(proportions[1] * samples))
    negative_count = samples - positive_count

    return samples, negative_count, positive_count

def make_tree_dot(node_id):
    global leaf_count, negative_leaf_count

    left = tree_structure.children_left[node_id]
    right = tree_structure.children_right[node_id]

    samples, negative_count, positive_count = get_node_counts(node_id)

    positive_ratio = (
        positive_count / samples * 100
        if samples > 0 else 0
    )

    if left == _tree.TREE_LEAF:
        predicted_class = int(
            tree_model.classes_[
                np.argmax(tree_structure.value[node_id][0])
            ]
        )

        leaf_count += 1

        if predicted_class == 0:
            negative_leaf_count += 1

        answer = (
            "뇌졸중" if predicted_class == 1
            else "뇌졸중 아님"
        )

        color = (
            "#F6B3B3" if predicted_class == 1
            else "#B9D3EE"
        )

        label = (
            f"답: {answer}\\n"
            f"훈련용 인원: {samples}명\\n"
            f"뇌졸중: {positive_count}명\\n"
            f"뇌졸중 비율: {positive_ratio:.1f}%"
        )

        dot_lines.append(
            f'n{node_id} [label="{label}", fillcolor="{color}"];'
        )

        return

    feature_index = tree_structure.feature[node_id]
    feature = selected_features[feature_index]

    used_features.add(feature)

    threshold = tree_structure.threshold[node_id]

    label = (
        f"{feature_names[feature]} <= {threshold:.2f}?\\n"
        f"훈련용 인원: {samples}명\\n"
        f"뇌졸중: {positive_count}명\\n"
        f"뇌졸중 비율: {positive_ratio:.1f}%"
    )

    dot_lines.append(
        f'n{node_id} [label="{label}", fillcolor="#FFF2CC"];'
    )

    dot_lines.append(f'n{node_id} -> n{left} [label="예"];')
    dot_lines.append(f'n{node_id} -> n{right} [label="아니요"];')

    make_tree_dot(left)
    make_tree_dot(right)

make_tree_dot(0)

dot_lines.append("}")

st.graphviz_chart(
    "\n".join(dot_lines),
    use_container_width=True
)

# =====================================
# 15. 의사결정트리 요약
# =====================================

st.subheader("의사결정트리 요약")

st.write(f"- 답을 내는 마디: **{leaf_count}개**")
st.write(
    f"- 그중 뇌졸중 아님이라고 답하는 마디: "
    f"**{negative_leaf_count}개**"
)

used_feature_labels = [
    feature_names[x]
    for x in selected_features
    if x in used_features
]

st.write(
    "- 실제로 질문에 사용한 속성: "
    + (", ".join(used_feature_labels) if used_feature_labels else "없음")
)

# =====================================
# 16. 마무리
# =====================================

st.divider()

st.caption("🧠 뇌졸중 예측 실습실 | 분류 모델 페이지")
st.caption(
    "※ 테스트 데이터는 모델 학습과 크기 맞추기에 사용하지 않았습니다."
)
st.caption(
    "※ 의사결정트리의 마디별 인원수는 크기를 맞춘 학습 데이터 기준입니다."
)
st.caption(
    "※ 이 실습 모델은 교육용이며 실제 의료 진단에 사용할 수 없습니다."
)
