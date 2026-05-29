

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Modeling for Course Demand and Revenue Forecasting on EduPro Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3, h4 {
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #334155;
}

label {
    color: white !important;
}

.css-1d391kg {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE SECTION
# =========================================================

st.title("📚 Predictive Modeling for Course Demand and Revenue Forecasting on EduPro Dashboard")
st.markdown("### Course Demand Prediction & Revenue Forecasting System")


# =========================================================
# LOAD DATASET
# =========================================================


xlsx_path = r"C:\Users\HP\Downloads\EduPro Online Platform.xlsx"

users_df = pd.read_excel(xlsx_path, sheet_name='Users')
teachers_df = pd.read_excel(xlsx_path, sheet_name='Teachers')
courses_df = pd.read_excel(xlsx_path, sheet_name='Courses')
transactions_df = pd.read_excel(xlsx_path, sheet_name='Transactions')


# =========================================================
# DATA PREPARATION
# =========================================================

transactions_df['TransactionDate'] = pd.to_datetime(
    transactions_df['TransactionDate'],
    errors='coerce'
)

# ---------------------------------------------------------

course_perf_df = transactions_df.groupby('CourseID').agg(
    enrollment_count=('TransactionID', 'count'),
    course_revenue=('Amount', 'sum'),
    avg_transaction_amount=('Amount', 'mean'),
    first_transaction_date=('TransactionDate', 'min'),
    last_transaction_date=('TransactionDate', 'max')
).reset_index()

course_perf_df['active_days'] = (
    course_perf_df['last_transaction_date']
    - course_perf_df['first_transaction_date']
).dt.days.fillna(0)

course_perf_df['revenue_per_enrollment'] = np.where(
    course_perf_df['enrollment_count'] > 0,
    course_perf_df['course_revenue']
    / course_perf_df['enrollment_count'],
    0
)

# ---------------------------------------------------------

course_teacher_df = transactions_df.groupby('CourseID').agg(
    primary_teacher_id=(
        'TeacherID',
        lambda x: x.mode().iloc[0]
        if not x.mode().empty else x.iloc[0]
    )
).reset_index()

# ---------------------------------------------------------

model_df = courses_df.merge(
    course_perf_df,
    on='CourseID',
    how='left'
)

model_df = model_df.merge(
    course_teacher_df,
    on='CourseID',
    how='left'
)

model_df = model_df.merge(
    teachers_df,
    left_on='primary_teacher_id',
    right_on='TeacherID',
    how='left'
)

# =========================================================
# HANDLE MISSING VALUES
# =========================================================

fill_cols = [
    'enrollment_count',
    'course_revenue',
    'avg_transaction_amount',
    'active_days',
    'revenue_per_enrollment'
]

for col in fill_cols:
    model_df[col] = model_df[col].fillna(0)

model_df['CourseRating'] = model_df['CourseRating'].fillna(
    model_df['CourseRating'].median()
)

model_df['TeacherRating'] = model_df['TeacherRating'].fillna(
    model_df['TeacherRating'].median()
)

model_df['YearsOfExperience'] = model_df['YearsOfExperience'].fillna(
    model_df['YearsOfExperience'].median()
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

model_df['price_band'] = pd.qcut(
    model_df['CoursePrice'].rank(method='first'),
    3,
    labels=['Low', 'Medium', 'High']
)

model_df['duration_bucket'] = pd.qcut(
    model_df['CourseDuration'].rank(method='first'),
    3,
    labels=['Short', 'Medium', 'Long']
)

model_df['rating_tier'] = pd.cut(
    model_df['CourseRating'],
    bins=[0, 2.5, 4.0, 5.0],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

model_df['experience_bucket'] = pd.cut(
    model_df['YearsOfExperience'],
    bins=[-1, 3, 7, 50],
    labels=['Junior', 'Mid', 'Senior']
)

model_df['expertise_category_match'] = (
    model_df['Expertise'].astype(str).str.lower().str.strip()
    ==
    model_df['CourseCategory'].astype(str).str.lower().str.strip()
).astype(int)

model_df['month_first_sale'] = (
    model_df['first_transaction_date']
    .dt.month
    .fillna(0)
)

model_df['quarter_first_sale'] = (
    model_df['first_transaction_date']
    .dt.quarter
    .fillna(0)
)

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🔍 Dashboard Filters")

selected_category = st.sidebar.multiselect(
    "Select Course Category",
    options=model_df['CourseCategory'].unique(),
    default=model_df['CourseCategory'].unique()
)

selected_level = st.sidebar.multiselect(
    "Select Course Level",
    options=model_df['CourseLevel'].unique(),
    default=model_df['CourseLevel'].unique()
)

price_range = st.sidebar.slider(
    "Course Price Range",
    int(model_df['CoursePrice'].min()),
    int(model_df['CoursePrice'].max()),
    (
        int(model_df['CoursePrice'].min()),
        int(model_df['CoursePrice'].max())
    )
)

# =========================================================
# FILTER DATA
# =========================================================

filtered_df = model_df[
    (model_df['CourseCategory'].isin(selected_category))
    &
    (model_df['CourseLevel'].isin(selected_level))
    &
    (model_df['CoursePrice'] >= price_range[0])
    &
    (model_df['CoursePrice'] <= price_range[1])
]

# =========================================================
# KPI SECTION
# =========================================================

st.markdown("---")

total_courses = filtered_df['CourseID'].nunique()
total_revenue = filtered_df['course_revenue'].sum()
total_enrollment = filtered_df['enrollment_count'].sum()
avg_rating = filtered_df['CourseRating'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📚 Total Courses", f"{total_courses}")

with col2:
    st.metric("💰 Total Revenue", f"${total_revenue:,.0f}")

with col3:
    st.metric("👨‍🎓 Total Enrollments", f"{total_enrollment:,.0f}")

with col4:
    st.metric("⭐ Avg Course Rating", f"{avg_rating:.2f}")

st.markdown("---")

# =========================================================
# REVENUE FORECAST VISUALIZATION
# =========================================================

st.subheader("📈 Revenue Forecast by Course Category")

category_revenue_df = filtered_df.groupby(
    'CourseCategory',
    as_index=False
).agg(
    category_revenue=('course_revenue', 'sum')
)

fig1 = px.bar(
    category_revenue_df,
    x='CourseCategory',
    y='category_revenue',
    color='category_revenue',
    text_auto='.2s',
    height=500
)

fig1.update_layout(
    xaxis_title="Course Category",
    yaxis_title="Revenue",
    template="plotly_dark"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# CATEGORY DEMAND COMPARISON
# =========================================================

st.subheader("📊 Category-Level Demand Comparison")

demand_df = filtered_df.groupby(
    'CourseCategory',
    as_index=False
).agg(
    total_enrollment=('enrollment_count', 'sum')
)

fig2 = px.pie(
    demand_df,
    names='CourseCategory',
    values='total_enrollment',
    hole=0.5
)

fig2.update_layout(
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# MODEL BUILDING
# =========================================================

feature_cols = [
    'CourseCategory',
    'CourseType',
    'CourseLevel',
    'CoursePrice',
    'CourseDuration',
    'CourseRating',
    'YearsOfExperience',
    'TeacherRating',
    'price_band',
    'duration_bucket',
    'rating_tier',
    'experience_bucket',
    'expertise_category_match',
    'month_first_sale',
    'quarter_first_sale'
]

cat_cols = [
    'CourseCategory',
    'CourseType',
    'CourseLevel',
    'price_band',
    'duration_bucket',
    'rating_tier',
    'experience_bucket'
]

num_cols = [
    'CoursePrice',
    'CourseDuration',
    'CourseRating',
    'YearsOfExperience',
    'TeacherRating',
    'expertise_category_match',
    'month_first_sale',
    'quarter_first_sale'
]

X_df = model_df[feature_cols]
y_df = model_df['course_revenue']

# ---------------------------------------------------------

preprocessor = ColumnTransformer([
    (
        'num',
        Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]),
        num_cols
    ),
    (
        'cat',
        Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]),
        cat_cols
    )
])

# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.2,
    random_state=42
)

model = Pipeline([
    ('prep', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.subheader("🤖 Revenue Prediction Model Performance")

col5, col6 = st.columns(2)

with col5:
    st.metric("R² Score", f"{r2:.3f}")

with col6:
    st.metric("MAE", f"{mae:,.2f}")

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

st.subheader("🔥 Feature Importance Explorer")

prep_obj = model.named_steps['prep']
model_obj = model.named_steps['model']

feature_names = prep_obj.get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model_obj.feature_importances_
})

importance_df = importance_df.sort_values(
    'Importance',
    ascending=False
).head(15)

fig3 = px.bar(
    importance_df,
    x='Importance',
    y='Feature',
    orientation='h',
    color='Importance',
    height=600
)

fig3.update_layout(
    template='plotly_dark'
)

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# COURSE DEMAND PREDICTION
# =========================================================

st.subheader("🎯 Course Demand Prediction Simulator")

col7, col8, col9 = st.columns(3)

with col7:
    input_price = st.number_input(
        "Course Price",
        min_value=1.0,
        value=100.0
    )

    input_duration = st.number_input(
        "Course Duration",
        min_value=1.0,
        value=10.0
    )

with col8:
    input_level = st.selectbox(
        "Course Level",
        model_df['CourseLevel'].dropna().unique()
    )

    input_category = st.selectbox(
        "Course Category",
        model_df['CourseCategory'].dropna().unique()
    )

with col9:
    input_teacher_exp = st.number_input(
        "Instructor Experience",
        min_value=0,
        value=5
    )

    input_teacher_rating = st.slider(
        "Instructor Rating",
        1.0,
        5.0,
        4.0
    )

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button("Predict Revenue"):

    input_df = pd.DataFrame({
        'CourseCategory': [input_category],
        'CourseType': [model_df['CourseType'].mode()[0]],
        'CourseLevel': [input_level],
        'CoursePrice': [input_price],
        'CourseDuration': [input_duration],
        'CourseRating': [4.0],
        'YearsOfExperience': [input_teacher_exp],
        'TeacherRating': [input_teacher_rating],
        'price_band': ['Medium'],
        'duration_bucket': ['Medium'],
        'rating_tier': ['High'],
        'experience_bucket': ['Mid'],
        'expertise_category_match': [1],
        'month_first_sale': [6],
        'quarter_first_sale': [2]
    })

    predicted_revenue = model.predict(input_df)[0]

    st.success(
        f"💰 Predicted Course Revenue: ${predicted_revenue:,.2f}"
    )

# =========================================================
# RAW DATA SECTION
# =========================================================

with st.expander("📂 View Dataset"):

    st.dataframe(filtered_df)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "<center><h5 style='color:gray;'>EduPro Analytics Dashboard | Streamlit Project</h5></center>",
    unsafe_allow_html=True
)