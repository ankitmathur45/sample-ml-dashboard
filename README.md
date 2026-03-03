# 🤖 ML Dashboard

An interactive machine learning dashboard built with Streamlit.
Upload any CSV dataset, explore it visually, and train multiple ML models — all in your browser.

## Features

- 📊 Automatic data profiling — shape, types, missing values
- 📈 Interactive visualizations — distributions and correlation heatmap
- 🤖 Train 3 models simultaneously — Logistic Regression, Random Forest, XGBoost
- 🎯 Confusion matrices and feature importance charts

## Demo

Upload any classification dataset as a CSV and select your target column.

## Setup

```bash
git clone git@github.com:ankitmathur45/sample-ml-dashboard.git
cd sample-ml-dashboard
uv venv .venv --python 3.12
.venv\Scripts\activate
uv pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- Python 3.12
- Streamlit
- Scikit-learn
- XGBoost
- Plotly
- Pandas

## Dataset Used for Testing

[Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
