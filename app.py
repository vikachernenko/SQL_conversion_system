import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text, inspect
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from datasets import load_dataset
import sqlite3
import json
import os

st.set_page_config(page_title="Universal AI SQL Assistant", layout="wide")
st.title("🏛 Universal AI SQL Assistant")

st.sidebar.header("Глобальные настройки")
lang = st.sidebar.selectbox("Язык запроса и ответа", ["Русский", "English", "Español"])
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


def call_agent(system_prompt, user_prompt, require_json=False):
    llm = ChatOllama(model="llama3", base_url="http://localhost:11434", temperature=0)

    if require_json:
        system_prompt += "\n\nВАЖНО: Верни ответ СТРОГО в формате валидного JSON без разметки Markdown (без ```json ... ```)."

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", user_prompt)
    ])

    response = llm.invoke(prompt.format_messages())
    content = response.content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()

    return content


def agent_analyzer(query, schema):
    system_prompt = f"""Ты - AI-Аналитик баз данных ({db_dialect}).
    Твоя задача: проанализировать вопрос пользователя и схему БД.
    Если вопрос однозначный и понятно, из каких таблиц брать данные - верни status "clear".
    Если вопрос слишком размыт, таблиц с похожими данными несколько, или неясны критерии - верни status "ambiguous" и задай уточняющий вопрос (question).

    СХЕМА БД:
    {schema}

    Ответь строго в JSON формате:
    {{"status": "clear", "reason": "понятно что искать"}} 
    ИЛИ 
    {{"status": "ambiguous", "question": "Ваш уточняющий вопрос пользователю на {lang} языке"}}"""

    try:
        response = call_agent(system_prompt, query, require_json=True)
        return json.loads(response)
    except Exception as e:
        return {"status": "clear"}


def agent_sql_generator(query, schema, error_msg="", previous_sql=""):
    system_prompt = f"""Ты — Senior SQL Разработчик. Твоя целевая СУБД: {db_dialect}. 
    Напиши SQL запрос для решения задачи. Верни ТОЛЬКО чистый SQL-код, без текста вокруг.

    КРИТИЧЕСКИЕ ПРАВИЛА:
    1. ОГРАНИЧЕНИЯ (LIMIT): В SQLite и PostgreSQL используй LIMIT N в конце. ЗАПРЕЩЕНО писать SELECT TOP N.
    2. РАБОТА С ДАТАМИ:
       - Если СУБД SQLite: КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать EXTRACT() или YEAR(). Для поиска по году используй функцию strftime('%Y', колонка_с_датой) = '2010' или просто колонка_с_датой LIKE '2010-%'.
       - Если СУБД PostgreSQL: используй EXTRACT(YEAR FROM колонка_с_датой).
    3. СХЕМА БД: Внимательно проверяй, в какой таблице находится нужная колонка, прежде чем делать JOIN. Не выдумывай колонки!

    СХЕМА БД:
    {schema}"""

    user_prompt = f"Запрос: {query}"
    if error_msg and previous_sql:
        user_prompt += f"\n\nВНИМАНИЕ! Твой прошлый запрос упал с ошибкой!\nТвой прошлый код:\n{previous_sql}\nОшибка БД:\n{error_msg}\nИсправь свой код, проверь названия таблиц/колонок и синтаксис!"

    response = call_agent(system_prompt, user_prompt)

    sql = response.strip()
    if sql.startswith("```sql"):
        sql = sql[6:]
    if sql.endswith("```"):
        sql = sql[:-3]

    upper_sql = sql.upper()
    if "SELECT " in upper_sql:
        sql = sql[upper_sql.find("SELECT "):]

    return sql.strip()


def agent_visualization(df):
    columns_info = df.dtypes.astype(str).to_dict()
    sample_data = df.head(1).to_dict(orient='records')

    system_prompt = """Ты - Эксперт по визуализации данных.
    Выбери лучший тип графика для Plotly Express.
    Доступные типы: 'bar' (столбчатая), 'pie' (круговая), 'line' (линейная), 'none' (если визуализация не нужна).

    ПРАВИЛА:
    1. Внимательно смотри на названия колонок в словаре "Пример данных". Используй ИМЕННО ИХ в качестве значений x, y, names или values.
    2. Если есть колонка с текстом/категорией и колонка с числами - это 'bar' или 'pie'.
    3. Отвечай СТРОГО в JSON: {"chart_type": "bar", "x": "точное_название_колонки_из_данных", "y": "название_колонки_с_числом"}
    4. Для 'pie' используй ключи "names" и "values".
    5. Если колонка только одна - возвращай {"chart_type": "none"}.
    """

    user_prompt = f"Типы колонок: {json.dumps(columns_info, ensure_ascii=False)}\nПример данных: {json.dumps(sample_data, ensure_ascii=False)}"

    try:
        response = call_agent(system_prompt, user_prompt, require_json=True)
        return json.loads(response)
    except:
        return {"chart_type": "none"}

if st.session_state.db_path:
    engine = create_engine(f"sqlite:///{st.session_state.db_path}")

    with st.expander("Cхема БД (Инспектор)", expanded=False):
        tables = inspect(engine).get_table_names()
        if tables:
            select = st.selectbox("Посмотреть таблицу", tables)
            st.dataframe(pd.read_sql_query(f"SELECT * FROM {select} LIMIT 5", engine))

    st.subheader("Чат с AI-Аналитиком")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg.get("df") is not None:
                st.dataframe(msg["df"], use_container_width=True)

            if msg.get("fig") is not None:
                st.plotly_chart(msg["fig"], use_container_width=True)

            if msg.get("sql") is not None:
                with st.expander("Посмотреть сгенерированный SQL"):
                    st.code(msg["sql"], language="sql")
    if prompt := st.chat_input("Например: Покажи топ-5 товаров по продажам..."):

        full_query = prompt
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant" and "df" not in \
                st.session_state.messages[-1]:
            full_query = f"Контекст: {st.session_state.messages[-2]['content']}. Уточнение: {prompt}"

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Агенты работают над запросом..."):
                schema = get_db_context(engine, rag_source)

                analysis = agent_analyzer(full_query, schema)

                if analysis.get("status") == "ambiguous":
                    question = analysis.get("question", "Пожалуйста, уточните ваш запрос.")
                    st.markdown(question)
                    st.session_state.messages.append({"role": "assistant", "content": question})
                else:
                    sql_code = ""
                    err_msg = ""
                    df = None

                    for attempt in range(3):
                        current_sql = agent_sql_generator(full_query, schema, err_msg, sql_code)
                        sql_code = current_sql

                        try:
                            df = pd.read_sql_query(text(sql_code), engine)
                            err_msg = ""
                            break
                        except Exception as e:
                            err_msg = str(e)
                            print(f"Попытка {attempt + 1} провалена. Ошибка: {err_msg}")

                    if df is not None:
                        st.markdown(f"Готово! Вот результат:")
                        st.dataframe(df, use_container_width=True)

                        fig = None
                        if not df.empty and len(df.columns) >= 2:
                            viz_config = agent_visualization(df)
                            chart_type = viz_config.get("chart_type", "none")

                            try:
                                if chart_type == "bar":
                                    fig = px.bar(df, x=viz_config.get("x"), y=viz_config.get("y"))
                                elif chart_type == "line":
                                    fig = px.line(df, x=viz_config.get("x"), y=viz_config.get("y"))
                                elif chart_type == "pie":
                                    fig = px.pie(df, names=viz_config.get("names", viz_config.get("x")),
                                                 values=viz_config.get("values", viz_config.get("y")))

                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                pass

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Отчет сформирован.",
                            "sql": sql_code,
                            "df": df,
                            "fig": fig
                        })
                    else:
                        error_text = f"Агенту не удалось исправить запрос после 3 попыток.\n\n**Последняя ошибка:**\n`{err_msg}`"
                        st.error(error_text)
                        with st.expander("Ошибочный SQL"):
                            st.code(sql_code, language="sql")
                        st.session_state.messages.append({"role": "assistant", "content": error_text, "sql": sql_code})