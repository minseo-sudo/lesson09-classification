
import streamlit as st
import pandas as pd
import json
from pathlib import Path

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 이해하고 뇌졸중 예측을 준비하는 실습 공간입니다.")

# ==========================================
# 2. 데이터 및 기록 파일 설정
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "greatsong/modudata/main/data/stroke.csv"
)

RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "기록.json"

# ==========================================
# 3. JSON 기록 불러오기
# ==========================================

def load_records():
    if RECORD_FILE.exists():
        try:
            with open(RECORD_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    return {}


# ==========================================
# 4. JSON 기록 저장하기
# ==========================================

def save_records(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    records = {
        "meanings": meanings,
        "source": source
    }

    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 5. CSV 데이터 불러오기
# ==========================================

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_URL,
        encoding="utf-8"
    )


try:
    df = load_data()

except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.error(str(e))
    st.stop()


# ==========================================
# 6. 저장된 기록 불러오기
# ==========================================

saved_records = load_records()

saved_meanings = saved_records.get("meanings", {})
saved_source = saved_records.get("source", "")


# ==========================================
# 7. 데이터 소개
# ==========================================

st.header("1. 데이터 소개")

st.write(
    """
    이 데이터는 사람들의 기본 정보와 건강 관련 정보를 바탕으로
    뇌졸중 발생 여부를 기록한 데이터입니다.

    나이, 성별, 고혈압, 심장병, 평균 혈당, BMI,
    흡연 상태 등의 정보를 포함하고 있습니다.

    마지막 열인 `stroke`는 뇌졸중 발생 여부를 나타냅니다.
    """
)

# ==========================================
# 8. 주요 통계 카드 네 개
# ==========================================

total_people = len(df)
total_columns = len(df.columns)

stroke_count = int((df["stroke"] == 1).sum())

stroke_ratio = (
    stroke_count / total_people * 100
    if total_people > 0
    else 0
)

st.subheader("📊 데이터 주요 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    label="전체 사람 수",
    value=f"{total_people:,}명"
)

col2.metric(
    label="전체 열 개수",
    value=f"{total_columns}개"
)

col3.metric(
    label="뇌졸중 발생 인원",
    value=f"{stroke_count:,}명"
)

col4.metric(
    label="뇌졸중 발생 비율",
    value=f"{stroke_ratio:.2f}%"
)

st.divider()


# ==========================================
# 9. 열 정보 및 우리말 뜻 입력
# ==========================================

st.header("2. 데이터 열 살펴보기")

st.write(
    "각 열의 이름과 값의 종류, 빈 값 개수를 확인하고 "
    "교재를 참고하여 우리말 뜻을 직접 입력해 보세요."
)

column_info = []

for column in df.columns:

    unique_count = df[column].nunique(dropna=True)
    missing_count = int(df[column].isna().sum())

    if pd.api.types.is_numeric_dtype(df[column]):
        value_type = f"숫자형 ({unique_count:,}종류)"
    else:
        value_type = f"문자형 ({unique_count:,}종류)"

    column_info.append({
        "열 이름": column,
        "우리말 뜻": saved_meanings.get(column, ""),
        "값의 종류": value_type,
        "빈 값 개수": missing_count
    })

info_df = pd.DataFrame(column_info)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수"
    ],
    column_config={
        "열 이름": st.column_config.TextColumn(
            "열 이름",
            width="medium"
        ),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            width="large",
            help="교재를 참고하여 각 열의 우리말 뜻을 입력하세요."
        ),
        "값의 종류": st.column_config.TextColumn(
            "값의 종류",
            width="medium"
        ),
        "빈 값 개수": st.column_config.NumberColumn(
            "빈 값 개수",
            format="%d"
        )
    },
    key="meaning_editor"
)

st.caption(
    "※ 우리말 뜻 열만 수정할 수 있습니다. "
    "빈 값은 데이터에 존재하는 결측치의 개수입니다."
)

st.divider()


# ==========================================
# 10. 데이터 처음 다섯 줄 확인
# ==========================================

st.header("3. 데이터 처음 다섯 줄")

st.write(
    "원본 데이터의 첫 번째 행부터 다섯 번째 행까지 확인합니다."
)

st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True
)

st.divider()


# ==========================================
# 11. 데이터 출처 입력
# ==========================================

st.header("4. 데이터 출처")

st.write(
    "교재에 적힌 데이터 출처를 아래에 입력하세요."
)

source_text = st.text_area(
    "데이터 출처 입력",
    value=saved_source,
    placeholder="교재에 적힌 데이터 출처를 입력하세요.",
    height=120
)

# ==========================================
# 12. 저장 버튼
# ==========================================

if st.button(
    "💾 입력 내용 저장하기",
    type="primary",
    use_container_width=True
):

    meanings_to_save = {}

    for _, row in edited_df.iterrows():
        meanings_to_save[row["열 이름"]] = row["우리말 뜻"]

    try:
        save_records(
            meanings=meanings_to_save,
            source=source_text
        )

        st.success(
            "저장되었습니다! 새로고침해도 입력 내용이 유지됩니다."
        )

    except OSError as e:
        st.error(f"파일 저장 중 오류가 발생했습니다: {e}")


# ==========================================
# 13. 하단 안내
# ==========================================

st.divider()

st.caption(
    "🧠 뇌졸중 예측 실습실 | 데이터 탐색 및 머신러닝 학습용"
)

st.caption(
    "입력한 우리말 뜻과 데이터 출처는 "
    "`내_기록/기록.json` 파일에 저장됩니다."
)
