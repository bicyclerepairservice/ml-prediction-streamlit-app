import phik
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

@st.cache_resource
def load_pipeline():
    with open("models/lasso_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)
    
def compute_vif(df):
    df_num = df.select_dtypes(include=[np.number]).dropna()
    vif_values = []

    for i in range(df_num.shape[1]):
        vif_values.append(variance_inflation_factor(df_num.values, i))

    return pd.DataFrame({
        "feature": df_num.columns,
        "VIF": vif_values
    }).set_index('feature')

pipeline = load_pipeline()
preprocess = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]
all_features = list(pipeline.named_steps['preprocess'].feature_names_in_)
target = 'selling_price'

st.sidebar.header("Навигация")
page = st.sidebar.radio(
    "Перейти к разделу:",
    ["EDA", "Прогнозирование", "Веса модели"]
)

if page == "EDA":
    st.title("Обзор признакового пространства модели предсказания цены продажи автомобиля")
    df = pd.DataFrame()
    uploaded = st.file_uploader("Можете загрузить файлик самостоятельно для EDA в формате CSV самостоятельно", type=["csv"])
    if st.checkbox("Поставьте галочку, если хотите воспользоваться дефолтным датафреймом"):
        df = pd.read_csv('data/sample_test_data.csv').iloc[:, 1:]
    if uploaded:
        df = pd.read_csv(uploaded)

    model = load_pipeline()
    numerical_features = (
        model.named_steps['preprocess'].named_transformers_['num'].feature_names_in_
    )

    if not df.empty:
        df = df.loc[:, df.columns.isin(all_features + [target])]
        binary_cols = [c for c in df.columns if set(df[c].unique()) <= {0,1}]

        if st.checkbox("Показать загруженную табличку с признаками"):
            st.write("### Просмотр данных")
            st.dataframe(df.head())

        col1, col2 = st.columns(2)


        with col1:
            st.write("### Гистограмма распределения целевой переменной")
            fig, ax = plt.subplots()
            ax.hist(df[target], bins=30)
            ax.set_title(f"Корреляционная матрица признаков")
            st.pyplot(fig)

            st.write("### Корреляционная матрица Phik для признаков модели")
            phik_corr = df[numerical_features].phik_matrix()
            sns.heatmap(phik_corr, annot=True, fmt=".1f", ax=ax)
            st.pyplot(fig)


            st.write("### VIF для признаков в модели")
            fig, ax = plt.subplots(figsize=(6,4))
            vif_df = compute_vif(df.drop(columns=[target]))
            vif_df.plot(kind='bar', ax=ax)
            ax.set_title("VIF")
            st.pyplot(fig)


        with col2:
            st.write("### Ящик с усами для переменных")
            col_b = st.selectbox("Признак:", numerical_features, key="box_feat")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col_b], ax=ax)
            st.pyplot(fig)

            st.write("### Парное сравнение переменных на диаграмме рассеяния")
            num_col_1 = st.selectbox("Первый признак", df.columns, key="scatter_feat_1")
            num_col_2 = st.selectbox("Второй признак", df.columns, key="scatter_feat_2")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[num_col_1], y=df[num_col_2], ax=ax)
            ax.set_title(f"{num_col_1} против {num_col_2}")
            st.pyplot(fig)





if page == "Прогнозирование":
    st.title("Прогноз стоимости автомобиля")

    method = st.radio("Способ ввода данных", ["CSV", "Ручной ввод"])

    if method == "CSV":
        file = st.file_uploader("Загрузите CSV", type=["csv"])
        if file:
            df = pd.read_csv(file).iloc[:, 1:]
            st.write("Загруженные данные:")
            st.dataframe(df.head())

            pred = pipeline.predict(df[all_features])
            df["prediction"] = pred

            st.write("### Результат:")
            st.dataframe(df[['prediction']])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Скачать результат", csv, "predictions.csv")

    else:
        st.write("Введите параметры:")

        num_features = preprocess.feature_names_in_
        inputs = {}

        for col in num_features:
            inputs[col] = st.number_input(col, value=0.0)

        if st.button("Предсказать"):
            X = pd.DataFrame([inputs])
            pred = pipeline.predict(X)[0]

            st.success(f"Предсказанная цена: {pred:,.0f}")


if page == "Веса модели":
    st.title("Веса / коэффициенты модели")

    feature_names = preprocess.get_feature_names_out()
    coefs = model.coef_

    df_coef = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df_coef = df_coef.sort_values("coef")

    st.write("### Таблица коэффициентов")
    st.dataframe(df_coef)

    st.write("### График коэффициентов")
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(df_coef["feature"], df_coef["coef"])
    plt.tight_layout()
    st.pyplot(fig)

