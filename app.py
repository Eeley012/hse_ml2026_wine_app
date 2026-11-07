"""
Демо-приложение сервиса подбора вин.

Вкладки: оценка одного вина, модельные метрики, пакетное предсказание,
коэффициенты модели. Модель здесь не обучается, она загружается из model.joblib.

Про кэширование (ответ на вопрос задания 4.5):

- Модель грузится через st.cache_resource, потому что это один тяжелый объект
  на весь сервер: декоратор отдает всем сессиям ссылку на один и тот же объект,
  ничего не копируя и не сериализуя.
- Таблицы и массивы грузятся через st.cache_data: такие значения возвращаются
  копией на каждый вызов, поэтому одна сессия не может испортить общий кэш,
  изменив свой DataFrame. Для модели это не подошло бы: cache_data пытается
  сериализовать результат, и на каждый вызов мы получали бы новую копию модели.
- Если вызвать joblib.load без декоратора, модель будет читаться с диска при
  каждом действии пользователя: streamlit перезапускает скрипт сверху вниз на
  любой клик, так что каждое движение слайдера порога заново читало бы файл и
  заново считало predict_proba на всей тестовой выборке, хотя от порога
  вероятности не зависят.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Подбор хороших вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    """Один объект модели на весь сервер."""
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    """Вероятности класса 1 на тесте: считаются один раз, от порога не зависят."""
    return _model.predict_proba(X)[:, 1]


model = load_model()
config = load_config()
df_test = load_test()

FEATURES = list(model.feature_names_in_)
X_test = df_test[FEATURES]
y_test = df_test['target'].to_numpy()
p_test = test_proba(model, X_test)

stats = X_test.describe()

# ------------------------------- сайдбар -------------------------------
st.sidebar.header('Порог')
threshold = st.sidebar.slider(
    'Вино попадает в подборку, если вероятность не ниже порога',
    min_value=0.0, max_value=1.0, step=0.01,
    value=float(config['threshold']), key='threshold')
st.sidebar.caption(f"Порог из config.json: {config['threshold']:.4f}")

st.title('🍷 Подбор хороших вин')

tab_wine, tab_metrics, tab_batch, tab_coefs = st.tabs(
    ['Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

# ------------------------- 4.1 оценка одного вина -------------------------
with tab_wine:
    st.subheader('Оценка одного вина')
    st.caption('Значения по умолчанию — медианы по тестовой выборке, границы — минимум и максимум.')

    numeric = [f for f in FEATURES if f != 'is_red']
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(numeric):
        with columns[i % 3]:
            lo, hi = float(stats.loc['min', feature]), float(stats.loc['max', feature])
            values[feature] = st.number_input(
                feature, min_value=lo, max_value=hi,
                value=float(stats.loc['50%', feature]),
                step=(hi - lo) / 100, format='%.4f')

    color = st.radio('Цвет вина', ['красное', 'белое'], horizontal=True)
    values['is_red'] = 1 if color == 'красное' else 0

    wine = pd.DataFrame([values])[FEATURES]
    proba = float(model.predict_proba(wine)[0, 1])

    left, right = st.columns(2)
    left.metric('Вероятность, что вино хорошее', f'{proba:.1%}')
    if proba >= threshold:
        right.success(f'В подборку: вероятность {proba:.1%} не ниже порога {threshold:.2f}')
    else:
        right.warning(f'Мимо подборки: вероятность {proba:.1%} ниже порога {threshold:.2f}')

# --------------------------- 4.2 модельные метрики ---------------------------
with tab_metrics:
    st.subheader('Модельные метрики на тестовой выборке')

    y_pred = (p_test >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    precision = metrics.precision_score(y_test, y_pred, zero_division=0)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    cols = st.columns(3)
    cols[0].metric('precision', f'{precision:.3f}')
    cols[1].metric('recall', f'{recall:.3f}')
    cols[2].metric('F1', f'{f1:.3f}')

    left, right = st.columns([1, 2])

    with left:
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
        display = metrics.ConfusionMatrixDisplay(
            metrics.confusion_matrix(y_test, y_pred),
            display_labels=['не хорошее', 'хорошее'])
        display.plot(cmap='Blues', values_format='d', ax=ax_cm, colorbar=False)
        ax_cm.grid(False)
        ax_cm.set_title(f'Матрица ошибок, порог {threshold:.2f}')
        st.pyplot(fig_cm)

    with right:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))

        fpr, tpr, _ = metrics.roc_curve(y_test, p_test)
        ax_roc.plot(fpr, tpr, label=f'ROC-AUC = {metrics.roc_auc_score(y_test, p_test):.3f}')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.scatter(fp / (fp + tn), recall, color='red', zorder=5, s=70,
                       label=f'текущий порог {threshold:.2f}')
        ax_roc.set_xlabel('FPR')
        ax_roc.set_ylabel('TPR (recall)')
        ax_roc.set_title('ROC-кривая')
        ax_roc.legend(loc='lower right')

        prec_curve, rec_curve, _ = metrics.precision_recall_curve(y_test, p_test)
        ax_pr.plot(rec_curve, prec_curve,
                   label=f'PR-AUC = {metrics.average_precision_score(y_test, p_test):.3f}')
        ax_pr.axhline(y_test.mean(), color='k', ls='--', lw=1, label='случайная модель')
        ax_pr.scatter(recall, precision, color='red', zorder=5, s=70,
                      label=f'текущий порог {threshold:.2f}')
        ax_pr.set_xlabel('recall')
        ax_pr.set_ylabel('precision')
        ax_pr.set_title('PR-кривая')
        ax_pr.legend(loc='upper right')

        fig.tight_layout()
        st.pyplot(fig)

    st.info(
        f'Сейчас в подборку попадает {tp + fp} вин из {len(y_test)}. '
        f'Из них действительно хороших {tp} — это precision {precision:.0%}. '
        f'Мимо подборки проходит {fn} хороших вин из {tp + fn} — охват {recall:.0%}.\n\n'
        '**Если поднять порог** — в подборке останутся только вина, в которых модель уверена: '
        'разочарований будет меньше (precision растет), но подборка станет скудной '
        'и много хороших вин потеряется (recall падает).\n\n'
        '**Если опустить порог** — подборка станет шире и хороших вин в нее попадет больше '
        '(recall растет), но вместе с ними в нее начнут просачиваться посредственные '
        'вина (precision падает).')

# -------------------------- 4.3 пакетное предсказание --------------------------
with tab_batch:
    st.subheader('Пакетное предсказание')
    st.caption(f'Используется порог из сайдбара: {threshold:.2f}. '
               f'Нужны столбцы: {", ".join(FEATURES)}.')

    uploaded = st.file_uploader('CSV с признаками вин (без столбца target)', type='csv')

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f'Не удалось прочитать файл: {error}')
        else:
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                result = batch.copy()
                result['вероятность'] = model.predict_proba(batch[FEATURES])[:, 1]
                result['решение'] = np.where(
                    result['вероятность'] >= threshold, 'в подборку', 'мимо')

                st.success(f'Обработано вин: {len(result)}, '
                           f'в подборку попало {(result["решение"] == "в подборку").sum()}.')
                st.dataframe(result, width='stretch')
                st.download_button(
                    'Скачать результат',
                    data=result.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv')

# --------------------------- 4.4 что выучила модель ---------------------------
with tab_coefs:
    st.subheader('Коэффициенты логистической регрессии')
    st.caption('Признаки масштабированы, поэтому коэффициенты сравнимы между собой.')

    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()

    fig_coefs, ax = plt.subplots(figsize=(8, 6))
    ax.barh(coefs.index, coefs.values,
            color=['tab:red' if c < 0 else 'tab:green' for c in coefs.values])
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('коэффициент')
    ax.set_title('Вклад признаков в вероятность «хорошего» вина')
    fig_coefs.tight_layout()
    st.pyplot(fig_coefs)

    top_for = coefs.nlargest(3)
    top_against = coefs.nsmallest(3)
    left, right = st.columns(2)
    left.markdown('**Сильнее всего за хорошее вино:**\n\n' + '\n'.join(
        f'- `{name}` ({value:+.2f})' for name, value in top_for.items()))
    right.markdown('**Сильнее всего против:**\n\n' + '\n'.join(
        f'- `{name}` ({value:+.2f})' for name, value in top_against.items()))
