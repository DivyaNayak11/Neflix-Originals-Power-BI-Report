#import the libraries
import pandas as pd
import numpy as np
from tqdm import tqdm



# load dataset
xlsx_path = r'"C:\Users\HP\Downloads\EduPro Online Platform.xlsx'

users_df = pd.read_excel(xlsx_path, sheet_name='Users')
teachers_df = pd.read_excel(xlsx_path, sheet_name='Teachers')
courses_df = pd.read_excel(xlsx_path, sheet_name='Courses')
transactions_df = pd.read_excel(xlsx_path, sheet_name='Transactions')

print(users_df.head())
print(teachers_df.head())
print(courses_df.head())
print(transactions_df.head())



# data preparation
transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'], errors='coerce')

course_perf_df = transactions_df.groupby('CourseID').agg(
    enrollment_count=('TransactionID', 'count'),
    course_revenue=('Amount', 'sum'),
    avg_transaction_amount=('Amount', 'mean'),
    first_transaction_date=('TransactionDate', 'min'),
    last_transaction_date=('TransactionDate', 'max')
).reset_index()

course_perf_df['active_days'] = (
    course_perf_df['last_transaction_date'] - course_perf_df['first_transaction_date']
).dt.days.fillna(0)

course_perf_df['revenue_per_enrollment'] = np.where(
    course_perf_df['enrollment_count'] > 0,
    course_perf_df['course_revenue'] / course_perf_df['enrollment_count'],
    0
)

course_teacher_df = transactions_df.groupby('CourseID').agg(
    primary_teacher_id=('TeacherID', lambda series_vals: series_vals.mode().iloc[0] if not series_vals.mode().empty else series_vals.iloc[0])
).reset_index()

model_df = courses_df.merge(course_perf_df, on='CourseID', how='left')
model_df = model_df.merge(course_teacher_df, on='CourseID', how='left')
model_df = model_df.merge(teachers_df, left_on='primary_teacher_id', right_on='TeacherID', how='left')

model_df['enrollment_count'] = model_df['enrollment_count'].fillna(0)
model_df['course_revenue'] = model_df['course_revenue'].fillna(0)
model_df['avg_transaction_amount'] = model_df['avg_transaction_amount'].fillna(0)
model_df['active_days'] = model_df['active_days'].fillna(0)
model_df['revenue_per_enrollment'] = model_df['revenue_per_enrollment'].fillna(0)

model_df['CourseRating'] = model_df['CourseRating'].fillna(model_df['CourseRating'].median())
model_df['TeacherRating'] = model_df['TeacherRating'].fillna(model_df['TeacherRating'].median())
model_df['YearsOfExperience'] = model_df['YearsOfExperience'].fillna(model_df['YearsOfExperience'].median())

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
    model_df['Expertise'].astype(str).str.lower().str.strip() ==
    model_df['CourseCategory'].astype(str).str.lower().str.strip()
).astype(int)

model_df['month_first_sale'] = model_df['first_transaction_date'].dt.month.fillna(0)
model_df['quarter_first_sale'] = model_df['first_transaction_date'].dt.quarter.fillna(0)

print(model_df.head())
print(model_df[['enrollment_count', 'course_revenue']].describe())



category_revenue_df = model_df.groupby('CourseCategory', as_index=False).agg(
    category_revenue=('course_revenue', 'sum')
).sort_values('category_revenue', ascending=False)

print(category_revenue_df)



# model development & evaluation
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

X_df = model_df[feature_cols].copy()
y_enroll = model_df['enrollment_count'].copy()
y_revenue = model_df['course_revenue'].copy()

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

model_dict = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01, max_iter=10000),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
    X_df, y_enroll, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_df, y_revenue, test_size=0.2, random_state=42
)

result_rows = []
best_models = {}

for target_name, train_parts, test_parts in [
    ('Enrollment Count', (X_train_e, y_train_e), (X_test_e, y_test_e)),
    ('Course Revenue', (X_train_r, y_train_r), (X_test_r, y_test_r))
]:
    best_rmse = None
    best_name = None
    best_pipe = None

    for model_name, model_obj in tqdm(model_dict.items(), desc=target_name):
        pipe_obj = Pipeline([
            ('prep', preprocessor),
            ('model', model_obj)
        ])

        pipe_obj.fit(train_parts[0], train_parts[1])
        pred_vals = pipe_obj.predict(test_parts[0])

        mae_val = mean_absolute_error(test_parts[1], pred_vals)
        rmse_val = mean_squared_error(test_parts[1], pred_vals) ** 0.5
        r2_val = r2_score(test_parts[1], pred_vals)

        result_rows.append({
            'Target': target_name,
            'Model': model_name,
            'MAE': mae_val,
            'RMSE': rmse_val,
            'R2': r2_val
        })

        if best_rmse is None or rmse_val < best_rmse:
            best_rmse = rmse_val
            best_name = model_name
            best_pipe = pipe_obj

    best_models[target_name] = {
        'name': best_name,
        'pipeline': best_pipe
    }

results_df = pd.DataFrame(result_rows).sort_values(['Target', 'RMSE'])

print(results_df)
print(best_models['Enrollment Count']['name'])
print(best_models['Course Revenue']['name'])



#feature importance analysis
importance_tables = []

for target_name in ['Enrollment Count', 'Course Revenue']:
    best_pipe = best_models[target_name]['pipeline']
    model_obj = best_pipe.named_steps['model']
    prep_obj = best_pipe.named_steps['prep']
    feature_names = prep_obj.get_feature_names_out()

    if hasattr(model_obj, 'feature_importances_'):
        imp_vals = model_obj.feature_importances_
    elif hasattr(model_obj, 'coef_'):
        imp_vals = np.abs(model_obj.coef_)
    else:
        continue

    temp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': imp_vals,
        'target': target_name
    }).sort_values('importance', ascending=False).head(12)

    importance_tables.append(temp_df)

feature_importance_df = pd.concat(importance_tables, ignore_index=True)

print(feature_importance_df)



# export output
results_df.to_csv('edupro_model_metrics.csv', index=False)
model_df.to_csv('edupro_modeling_dataset.csv', index=False)
feature_importance_df.to_csv('edupro_feature_importance.csv', index=False)
