import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text, inspect
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import sqlite3
import os

st.set_page_config(page_title="Universal AI SQL Assistant", layout="wide")
st.title("🏛 Universal AI SQL Assistant")


st.sidebar.header("Глобальные настройки")
lang = st.sidebar.selectbox("Язык запроса и ответа", [
                            "Русский", "English", "Татарча"])
db_dialect = st.sidebar.selectbox("Выберите СУБД", ["SQLite", "PostgreSQL"])
rag_source = st.sidebar.selectbox("Источник знаний (RAG)", ["Spider", "BIRD"])

if "db_path" not in st.session_state:
    st.session_state.db_path = None

db_file = st.sidebar.file_uploader(
    "Загрузите файл базы данных (.db, .sqlite, .sql)", type=["db", "sqlite", "sql"])

if db_file:
    path = "current_db.db"
    file_extension = db_file.name.split('.')[-1].lower()

    if file_extension in ['db', 'sqlite']:
        with open(path, "wb") as f:
            f.write(db_file.getbuffer())
    elif file_extension == 'sql':
        if os.path.exists(path):
            os.remove(path)
        sql_script = db_file.getvalue().decode('utf-8')
        conn = sqlite3.connect(path)
        conn.executescript(sql_script)
        conn.commit()
        conn.close()

    st.session_state.db_path = path


class SpiderKnowledgeBase:
    def __init__(self):
        self.dataset = load_dataset(
            "xlangai/spider", split='train', streaming=True)
        self.examples = []

    def get_few_shot_examples(self, numexamples=3):
        if not self.examples:
            it = iter(self.dataset)
            count = 0
            for ex in it:
                self.examples.append(
                    f"Вопрос: {ex['question']}\nSQL: {ex['query']}")
                count += 1
                if count >= 10:
                    break
        return "\n\n".join(self.examples[:numexamples])


class ProKnowledgeBase:
    def __init__(self):
        print("Загрузка базы знаний BIRD...")
        self.bird_ds = load_dataset(
            "birdsql/bird_mini_dev", split='mini_dev_sqlite', streaming=True)
        self.examples = []

    def get_complex_examples(self, num_examples=3):
        if not self.examples:
            it = iter(self.bird_ds)
            count = 0
            for ex in it:
                evidence = ex.get('evidence', 'Нет данных')
                question = ex.get('question', 'Нет вопроса')
                sql = ex.get('SQL', 'Нет SQL')

                example = (
                    f"Бизнес-контекст: {evidence}\n"
                    f"Вопрос: {question}\n"
                    f"SQL: {sql}"
                )
                self.examples.append(example)
                count += 1
                if count >= 10:
                    break
        return "\n\n---\n\n".join(self.examples[:num_examples])


def get_db_context(engine, rag_source):
    inspector = inspect(engine)
    context = "СТРУКТУРА ТАБЛИЦ В БАЗЕ ДАННЫХ:\n"
    for table in inspector.get_table_names():
        columns = [
            f"{c['name']} (тип: {c['type']})" for c in inspector.get_columns(table)]
        context += f"- Таблица '{table}', колонки: {', '.join(columns)}\n"

        fks = inspector.get_foreign_keys(table)
        for fk in fks:
            context += f"  (Связь: {table}.{fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]})\n"

    if rag_source == "Spider":
        spider_kb = SpiderKnowledgeBase()
        context += "\n\nПРИМЕРЫ ИЗ SPIDER:\n" + \
            spider_kb.get_few_shot_examples(numexamples=3)
    elif rag_source == "BIRD":
        bird_kb = ProKnowledgeBase()
        context += "\n\nПРИМЕРЫ ИЗ BIRD:\n" + \
            bird_kb.get_complex_examples(num_examples=3)
    return context


def ask_ai_for_sql(query, schema, error_msg=""):
    llm = ChatOllama(
        model="llama3", base_url="http://localhost:11434", temperature=0)

    system_msg = f"""Ты — SQL эксперт ({db_dialect}). Верни ТОЛЬКО SQL. 
    {f'ОШИБКА: {error_msg}.' if error_msg else ''}
    Держись СХЕМЫ:\n{schema}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", f"Запрос ({lang}): {query}")
    ])

    response = llm.invoke(prompt.format_messages())
    sql = response.content.replace("```sql", "").replace("```", "").strip()

    if "SELECT" in sql.upper():
        start = sql.upper().find("SELECT")
        sql = sql[start:].strip()
    return sql


if st.session_state.db_path:
    engine = create_engine(f"sqlite:///{st.session_state.db_path}")

    with st.expander("Cхема БД"):
        tables = inspect(engine).get_table_names()
        if tables:
            select = st.selectbox("Таблица", tables)
            st.dataframe(pd.read_sql_query(
                f"SELECT * FROM {select} LIMIT 5", engine))

    user_input = st.text_input("Задайте вопрос к данным:")

    if st.button("Начать анализ"):
        if user_input:
            schema = get_db_context(engine, rag_source)
            with st.spinner("Идет анализ..."):
                sql_code = ""
                err_msg = ""
                df = None

                for i in range(3):
                    try:
                        sql_code = ask_ai_for_sql(user_input, schema, err_msg)
                        df = pd.read_sql_query(text(sql_code), engine)
                        break
                    except Exception as e:
                        err_msg = str(e)

                if df is not None:
                    st.success("Готово!")
                    st.code(sql_code, language="sql")
                    st.dataframe(df, use_container_width=True)

                    if not df.empty and len(df.columns) >= 2:
                        num_cols = df.select_dtypes(include=['number']).columns
                        if not num_cols.empty:
                            st.subheader("Визуализация:")
                            fig = px.bar(
                                df, x=df.columns[0], y=num_cols[0], color=num_cols[0])
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Не удалось выполнить анализ. Ошибка: {err_msg}")
