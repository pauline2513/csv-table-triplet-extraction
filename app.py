import io
import json
import pandas as pd
import streamlit as st
from llm_triplet_extraction import extract_triplets_by_llm
# from text_postprocessing import extract_triplets_from_text
from triplets_from_text_extraction import process_triplets


def change_to_frame_format(triplets):
    triplets = triplets["triplets"]
    new_element = {"triplets": []}
    for triplet in triplets:
        subject = triplet["subject"]
        predicate = triplet["predicate"]
        object = triplet["object"]
        new_element["triplets"].append({
            "subject": {"text": subject, "frame": []},
            "predicate": {"text": predicate, "frame": []},
            "object": {"text": object, "frame": []}
        })
    return new_element


def render_result(triplets_result, format=True, download_button_id=0):
    if st.session_state["triplets_error"]:
        st.error(f"Ошибка извлечения триплетов: {st.session_state["triplets_error"]}")
    elif triplets_result is not None:
        if format:
            triplets_result = change_to_frame_format(triplets_result)
        st.write("Результат извлечения:")
        st.json(triplets_result)

        json_bytes = json.dumps(
            triplets_result,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        st.download_button(
            "Скачать JSON с триплетами",
            data=json_bytes,
            file_name=f"triplets.json",
            mime="application/json",
            key=f"download_triplets_{download_button_id}"
        )


def main():
    st.set_page_config(page_title="Triplet Extraction", layout="wide")
    st.title("Извлечение триплетов из CSV-таблиц")

    if "main_result" not in st.session_state:
        st.session_state["main_result"] = None
    if "processed_result" not in st.session_state:
        st.session_state["processed_result"] = None
    if "triplets_error" not in st.session_state:
        st.session_state["triplets_error"] = None
    if "main_source_name" not in st.session_state:
        st.session_state["main_source_name"] = None

    separator = st.radio(
        "Знак разделителя (для ллм по умолчанию описана точка с запятой из-за наличия текстов с запятыми)",
        options=["Запятая", "Точка с запятой"],
        index=0,
        key="separator_choice"
    )
    if separator == "Запятая":
        sep = ","
    else:
        sep = ";"

    uploaded = st.file_uploader(
        "Загрузите файл",
        type=["csv"],
        key="main_upload",
        accept_multiple_files=False
    )

    uploaded_df = None

    if uploaded is not None:
        uploaded_text = uploaded.getvalue().decode("utf-8")
        uploaded_df = pd.read_csv(io.BytesIO(uploaded.getvalue()), sep=sep)
        st.text(uploaded_text)

    run = st.button(
        "Извлечь триплеты",
        type="primary",
        disabled=uploaded is None,
        key="extract_triplets_btn"
    )

    if run and uploaded is not None:
        try:
            with st.spinner("Извлекаю триплеты..."):
                triplets_result = extract_triplets_by_llm(uploaded_df)

            st.session_state["triplets_error"] = None
            st.session_state["main_result"] = triplets_result
            st.session_state["processed_result"] = None
            st.session_state["main_source_name"] = uploaded.name

        except Exception as exc:
            st.session_state["main_result"] = None
            st.session_state["processed_result"] = None
            st.session_state["triplets_error"] = str(exc)

    if st.session_state["main_result"] is not None:
        st.subheader("Исходный результат")
        render_result(st.session_state["main_result"], format=True, download_button_id=1)

        mode_choice = st.radio(
            "Способ обработки предложений в триплетах",
            options=[
                "Извлечь общие SPO из всего текста (без учета текущей структуры S-P-O) каждого триплета",
                "Извлечь элементы отдельно с сохранением структуры (просто добавить frame)"
            ],
            index=0,
            key="mode_choice"
        )

        if mode_choice == "Извлечь общие SPO из всего текста (без учета текущей структуры S-P-O) каждого триплета":
            mode = "concat"
        else:
            mode = "separate"

        process_text = st.button("Обработать текст в триплетах", key="process_triplets_btn")

        if process_text:
            st.session_state["processed_result"] = process_triplets(
                st.session_state["main_result"],
                mode=mode
            )

        if st.session_state["processed_result"] is not None:
            st.subheader("Обработанный результат")
            render_result(st.session_state["processed_result"], format=False, download_button_id=2)


if __name__ == "__main__":
    main()
