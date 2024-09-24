# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns
# # # from sklearn.preprocessing import StandardScaler
# # # from sklearn.model_selection import train_test_split
# # # from sklearn.linear_model import LogisticRegression
# # # from sklearn.tree import DecisionTreeClassifier
# # # from sklearn.ensemble import RandomForestClassifier
# # # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # # from imblearn.over_sampling import SMOTE

# # # # Load the data
# # # @st.cache_data
# # # def load_data():
# # #     data = pd.read_csv('creditcard.csv')
# # #     return data

# # # data = load_data()

# # # st.title('Credit Card Fraud Detection Analysis')

# # # st.header('1. Top 5 Entries')
# # # st.write(data.head())

# # # st.header('2. Dataframe Shape')
# # # st.write(data.shape)

# # # st.header('3. Missing Data Count')
# # # st.write(data.isnull().sum())

# # # # st.header('4. Dataframe Info')
# # # # buffer = io.StringIO()
# # # # data.info(buf=buffer)
# # # # s = buffer.getvalue()
# # # # st.text(s)

# # # # Standardize 'Amount' column
# # # sc = StandardScaler()
# # # data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

# # # # Undersampling
# # # st.header('5. Undersampling')
# # # normal = data[data['Class'] == 0]
# # # fraud = data[data['Class'] == 1]
# # # normal_sample = normal.sample(n=len(fraud))
# # # new_data = pd.concat([normal_sample, fraud], ignore_index=True)

# # # st.write("Undersampled data size:")
# # # st.write(new_data['Class'].value_counts())
# # # st.write("Top 5 rows of undersampled data:")
# # # st.write(new_data.head())

# # # # Prepare undersampled data for modeling
# # # X = new_data.drop('Class', axis=1)
# # # y = new_data['Class']
# # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # # # Function to train and evaluate models
# # # def train_evaluate_model(model, X_train, X_test, y_train, y_test):
# # #     model.fit(X_train, y_train)
# # #     y_pred = model.predict(X_test)
# # #     return {
# # #         'Accuracy': accuracy_score(y_test, y_pred),
# # #         'Precision': precision_score(y_test, y_pred),
# # #         'Recall': recall_score(y_test, y_pred),
# # #         'F1 Score': f1_score(y_test, y_pred)
# # #     }

# # # # Train and evaluate models on undersampled data
# # # st.header('6-8. Model Performance on Undersampled Data')
# # # models = {
# # #     'Logistic Regression': LogisticRegression(),
# # #     'Decision Tree': DecisionTreeClassifier(),
# # #     'Random Forest': RandomForestClassifier()
# # # }

# # # undersampled_results = {}
# # # for name, model in models.items():
# # #     st.subheader(name)
# # #     results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
# # #     st.write(results)
# # #     undersampled_results[name] = results['Accuracy'] * 100

# # # # Visualize model performance on undersampled data
# # # st.header('9. Model Performance Comparison (Undersampled)')
# # # fig, ax = plt.subplots()
# # # sns.barplot(x=list(undersampled_results.keys()), y=list(undersampled_results.values()))
# # # plt.title('Model Accuracy Comparison (Undersampled Data)')
# # # plt.ylabel('Accuracy (%)')
# # # plt.xticks(rotation=45)
# # # st.pyplot(fig)

# # # # Oversampling
# # # st.header('10. Oversampling')
# # # X = data.drop('Class', axis=1)
# # # y = data['Class']
# # # X_res, y_res = SMOTE().fit_resample(X, y)

# # # st.write("Oversampled data size:")
# # # st.write(y_res.value_counts())
# # # st.write("Top 5 rows of oversampled features:")
# # # st.write(pd.DataFrame(X_res).head())

# # # # Prepare oversampled data for modeling
# # # X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

# # # # Train and evaluate models on oversampled data
# # # st.header('11-13. Model Performance on Oversampled Data')
# # # oversampled_results = {}
# # # for name, model in models.items():
# # #     st.subheader(name)
# # #     results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
# # #     st.write(results)
# # #     oversampled_results[name] = results['Accuracy'] * 100

# # # # Visualize model performance on oversampled data
# # # st.header('14. Model Performance Comparison (Oversampled)')
# # # fig, ax = plt.subplots()
# # # sns.barplot(x=list(oversampled_results.keys()), y=list(oversampled_results.values()))
# # # plt.title('Model Accuracy Comparison (Oversampled Data)')
# # # plt.ylabel('Accuracy (%)')
# # # plt.xticks(rotation=45)
# # # st.pyplot(fig)

# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from sklearn.preprocessing import StandardScaler
# # from sklearn.model_selection import train_test_split
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.tree import DecisionTreeClassifier
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# # from imblearn.over_sampling import SMOTE

# # # Load the data
# # @st.cache_data
# # def load_data():
# #     data = pd.read_csv('creditcard.csv')
# #     return data

# # data = load_data()

# # st.title('Credit Card Fraud Detection Analysis')

# # st.header('1. Top 5 Entries')
# # st.write(data.head())

# # st.header('2. Dataframe Shape')
# # st.write(data.shape)

# # st.header('3. Missing Data Count')
# # st.write(data.isnull().sum())

# # # Standardize 'Amount' column
# # sc = StandardScaler()
# # data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

# # # Standardize all features except 'Class'
# # scaler = StandardScaler()
# # X_full = data.drop('Class', axis=1)
# # X_full = scaler.fit_transform(X_full)

# # # Undersampling
# # st.header('5. Undersampling')
# # normal = data[data['Class'] == 0]
# # fraud = data[data['Class'] == 1]
# # normal_sample = normal.sample(n=len(fraud))
# # new_data = pd.concat([normal_sample, fraud], ignore_index=True)

# # st.write("Undersampled data size:")
# # st.write(new_data['Class'].value_counts())
# # st.write("Top 5 rows of undersampled data:")
# # st.write(new_data.head())

# # # Prepare undersampled data for modeling
# # X = new_data.drop('Class', axis=1)
# # X = scaler.fit_transform(X)  # Scale the features
# # y = new_data['Class']
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# # # Function to train and evaluate models
# # def train_evaluate_model(model, X_train, X_test, y_train, y_test):
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_test)
# #     return {
# #         'Accuracy': accuracy_score(y_test, y_pred),
# #         'Precision': precision_score(y_test, y_pred),
# #         'Recall': recall_score(y_test, y_pred),
# #         'F1 Score': f1_score(y_test, y_pred)
# #     }

# # # Train and evaluate models on undersampled data
# # st.header('6-8. Model Performance on Undersampled Data')
# # models = {
# #     'Logistic Regression': LogisticRegression(max_iter=500),  # Increase max_iter
# #     'Decision Tree': DecisionTreeClassifier(),
# #     'Random Forest': RandomForestClassifier()
# # }

# # undersampled_results = {}
# # for name, model in models.items():
# #     st.subheader(name)
# #     results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
# #     st.write(results)
# #     undersampled_results[name] = results['Accuracy'] * 100

# # # Visualize model performance on undersampled data
# # st.header('9. Model Performance Comparison (Undersampled)')
# # fig, ax = plt.subplots()
# # sns.barplot(x=list(undersampled_results.keys()), y=list(undersampled_results.values()))
# # plt.title('Model Accuracy Comparison (Undersampled Data)')
# # plt.ylabel('Accuracy (%)')
# # plt.xticks(rotation=45)
# # st.pyplot(fig)

# # # Oversampling
# # st.header('10. Oversampling')
# # X = X_full  # Already scaled
# # y = data['Class']
# # X_res, y_res = SMOTE().fit_resample(X, y)

# # st.write("Oversampled data size:")
# # st.write(y_res.value_counts())
# # st.write("Top 5 rows of oversampled features:")
# # st.write(pd.DataFrame(X_res).head())

# # # Prepare oversampled data for modeling
# # X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

# # # Train and evaluate models on oversampled data
# # st.header('11-13. Model Performance on Oversampled Data')
# # oversampled_results = {}
# # for name, model in models.items():
# #     st.subheader(name)
# #     results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
# #     st.write(results)
# #     oversampled_results[name] = results['Accuracy'] * 100

# # # Visualize model performance on oversampled data
# # st.header('14. Model Performance Comparison (Oversampled)')
# # fig, ax = plt.subplots()
# # sns.barplot(x=list(oversampled_results.keys()), y=list(oversampled_results.values()))
# # plt.title('Model Accuracy Comparison (Oversampled Data)')
# # plt.ylabel('Accuracy (%)')
# # plt.xticks(rotation=45)
# # st.pyplot(fig)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from imblearn.over_sampling import SMOTE

# # Load the data
# @st.cache_data
# def load_data():
#     data = pd.read_csv('creditcard.csv')
#     return data

# data = load_data()

# st.title('Credit Card Fraud Detection Analysis')

# # Sidebar with checkboxes for data overview
# with st.sidebar:
#     st.header('Dataset Overview Options')
#     first_rows_checkbox = st.checkbox('Display first 5 rows')
#     last_rows_checkbox = st.checkbox('Display last 5 rows')
#     missing_data_checkbox = st.checkbox('Display missing data count')
#     shape_checkbox = st.checkbox('Display dataframe shape')
    
#     # Sampling checkboxes
#     st.header('Sampling Options')
#     undersample_checkbox = st.checkbox('Undersampling')
#     oversample_checkbox = st.checkbox('Oversampling')

# # Data overview display
# if first_rows_checkbox:
#     st.header('First 5 Rows of the Dataset')
#     st.write(data.head())

# if last_rows_checkbox:
#     st.header('Last 5 Rows of the Dataset')
#     st.write(data.tail())

# if missing_data_checkbox:
#     st.header('Missing Data Count')
#     st.write(data.isnull().sum())

# if shape_checkbox:
#     st.header('Dataframe Shape')
#     st.write(data.shape)

# # Standardize 'Amount' column
# sc = StandardScaler()
# data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

# # Standardize all features except 'Class'
# scaler = StandardScaler()
# X_full = data.drop('Class', axis=1)
# X_full = scaler.fit_transform(X_full)

# # Undersampling logic
# if undersample_checkbox:
#     st.header('Undersampling')

#     normal = data[data['Class'] == 0]
#     fraud = data[data['Class'] == 1]
#     normal_sample = normal.sample(n=len(fraud))
#     new_data = pd.concat([normal_sample, fraud], ignore_index=True)

#     st.write("Undersampled data size:")
#     st.write(new_data['Class'].value_counts())
#     st.write("Top 5 rows of undersampled data:")
#     st.write(new_data.head())

#     # Prepare undersampled data for modeling
#     X = new_data.drop('Class', axis=1)
#     X = scaler.fit_transform(X)  # Scale the features
#     y = new_data['Class']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#     # Function to train and evaluate models
#     def train_evaluate_model(model, X_train, X_test, y_train, y_test):
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         return {
#             'Accuracy': accuracy_score(y_test, y_pred),
#             'Precision': precision_score(y_test, y_pred),
#             'Recall': recall_score(y_test, y_pred),
#             'F1 Score': f1_score(y_test, y_pred)
#         }

#     # Train and evaluate models on undersampled data
#     st.header('Model Performance on Undersampled Data')
#     models = {
#         'Logistic Regression': LogisticRegression(max_iter=500),  # Increase max_iter
#         'Decision Tree': DecisionTreeClassifier(),
#         'Random Forest': RandomForestClassifier()
#     }

#     undersampled_results = {}
#     for name, model in models.items():
#         st.subheader(name)
#         results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
#         st.write(results)
#         undersampled_results[name] = results['Accuracy'] * 100

#     # Visualize model performance on undersampled data
#     st.header('Model Performance Comparison (Undersampled)')
#     fig, ax = plt.subplots()
#     sns.barplot(x=list(undersampled_results.keys()), y=list(undersampled_results.values()))
#     plt.title('Model Accuracy Comparison (Undersampled Data)')
#     plt.ylabel('Accuracy (%)')
#     plt.xticks(rotation=45)
#     st.pyplot(fig)

# # Oversampling logic
# if oversample_checkbox:
#     st.warning("Dataset is large. Please wait...")
#     st.header('Oversampling')

#     X = X_full  # Already scaled
#     y = data['Class']
#     X_res, y_res = SMOTE().fit_resample(X, y)

#     st.write("Oversampled data size:")
#     st.write(y_res.value_counts())
#     st.write("Top 5 rows of oversampled features:")
#     st.write(pd.DataFrame(X_res).head())

#     # Prepare oversampled data for modeling
#     X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

#     # Train and evaluate models on oversampled data
#     st.header('Model Performance on Oversampled Data')
#     oversampled_results = {}
#     for name, model in models.items():
#         st.subheader(name)
#         results = train_evaluate_model(model, X_train, X_test, y_train, y_test)
#         st.write(results)
#         oversampled_results[name] = results['Accuracy'] * 100

#     # Visualize model performance on oversampled data
#     st.header('Model Performance Comparison (Oversampled)')
#     fig, ax = plt.subplots()
#     sns.barplot(x=list(oversampled_results.keys()), y=list(oversampled_results.values()))
#     plt.title('Model Accuracy Comparison (Oversampled Data)')
#     plt.ylabel('Accuracy (%)')
#     plt.xticks(rotation=45)
#     st.pyplot(fig)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_data()

st.title('Credit Card Fraud Detection Analysis')

# Sidebar with checkboxes for data overview
with st.sidebar:
    st.header('Dataset Overview Options')
    first_rows_checkbox = st.checkbox('Display first 5 rows')
    last_rows_checkbox = st.checkbox('Display last 5 rows')
    missing_data_checkbox = st.checkbox('Display missing data count')
    shape_checkbox = st.checkbox('Display dataframe shape')
    count_plot_checkbox = st.checkbox('Display count plot for fraud distribution')
    
    # Sampling dropdowns for undersampling and oversampling
    st.header('Sampling Options')
    undersample_dropdown = st.expander('Undersampling')
    with undersample_dropdown:
        undersample_lr = st.checkbox('Logistic Regression (Undersampling)')
        undersample_dt = st.checkbox('Decision Tree (Undersampling)')
        undersample_rf = st.checkbox('Random Forest (Undersampling)')
        undersample_compare = st.checkbox('Bar Graph Comparison (Undersampling)')
        
    oversample_dropdown = st.expander('Oversampling')
    with oversample_dropdown:
        oversample_lr = st.checkbox('Logistic Regression (Oversampling)')
        oversample_dt = st.checkbox('Decision Tree (Oversampling)')
        oversample_rf = st.checkbox('Random Forest (Oversampling)')
        oversample_compare = st.checkbox('Bar Graph Comparison (Oversampling)')

# Data overview display
if first_rows_checkbox:
    st.header('First 5 Rows of the Dataset')
    st.write(data.head())

if last_rows_checkbox:
    st.header('Last 5 Rows of the Dataset')
    st.write(data.tail())

if missing_data_checkbox:
    st.header('Missing Data Count')
    st.write(data.isnull().sum())

if shape_checkbox:
    st.header('Dataframe Shape')
    st.write(data.shape)

if count_plot_checkbox:
    st.header('Fraud Distribution Count Plot')
    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=data, ax=ax)
    plt.title('Fraud vs Non-Fraud Transactions')
    st.pyplot(fig)

# Standardize 'Amount' column
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))

# Standardize all features except 'Class'
scaler = StandardScaler()
X_full = data.drop('Class', axis=1)
X_full = scaler.fit_transform(X_full)

# Function to train and evaluate models
def train_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

def display_model_results(name, results):
    st.subheader(f'{name} Performance')
    results_df = pd.DataFrame([results], index=[name])
    st.table(results_df)

# Undersampling logic
if undersample_lr or undersample_dt or undersample_rf or undersample_compare:
    st.header('Undersampling')
    normal = data[data['Class'] == 0]
    fraud = data[data['Class'] == 1]
    normal_sample = normal.sample(n=len(fraud))
    new_data = pd.concat([normal_sample, fraud], ignore_index=True)

    st.write("Undersampled data size:")
    st.write(new_data['Class'].value_counts())
    st.write("Top 5 rows of undersampled data:")
    st.write(new_data.head())

    # Prepare undersampled data for modeling
    X = new_data.drop('Class', axis=1)
    X = scaler.fit_transform(X)  # Scale the features
    y = new_data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    undersampled_results = {}
    
    if undersample_lr:
        lr_model = LogisticRegression(max_iter=500)
        results = train_evaluate_model(lr_model, X_train, X_test, y_train, y_test)
        display_model_results('Logistic Regression (Undersampling)', results)
        undersampled_results['Logistic Regression'] = results['Accuracy'] * 100
        
    if undersample_dt:
        dt_model = DecisionTreeClassifier()
        results = train_evaluate_model(dt_model, X_train, X_test, y_train, y_test)
        display_model_results('Decision Tree (Undersampling)', results)
        undersampled_results['Decision Tree'] = results['Accuracy'] * 100
        
    if undersample_rf:
        rf_model = RandomForestClassifier()
        results = train_evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        display_model_results('Random Forest (Undersampling)', results)
        undersampled_results['Random Forest'] = results['Accuracy'] * 100
        
    if undersample_compare:
        st.header('Model Performance Comparison (Undersampled)')
        fig, ax = plt.subplots()
        sns.barplot(x=list(undersampled_results.keys()), y=list(undersampled_results.values()))
        plt.title('Model Accuracy Comparison (Undersampled Data)')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Oversampling logic
if oversample_lr or oversample_dt or oversample_rf or oversample_compare:
    st.warning("Dataset is large. Please wait...")
    st.header('Oversampling')

    X = X_full  # Already scaled
    y = data['Class']
    X_res, y_res = SMOTE().fit_resample(X, y)

    st.write("Oversampled data size:")
    st.write(y_res.value_counts())
    st.write("Top 5 rows of oversampled features:")
    st.write(pd.DataFrame(X_res).head())

    # Prepare oversampled data for modeling
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.20, random_state=42)

    oversampled_results = {}
    
    if oversample_lr:
        lr_model = LogisticRegression(max_iter=500)
        results = train_evaluate_model(lr_model, X_train, X_test, y_train, y_test)
        display_model_results('Logistic Regression (Oversampling)', results)
        oversampled_results['Logistic Regression'] = results['Accuracy'] * 100

    if oversample_dt:
        dt_model = DecisionTreeClassifier()
        results = train_evaluate_model(dt_model, X_train, X_test, y_train, y_test)
        display_model_results('Decision Tree (Oversampling)', results)
        oversampled_results['Decision Tree'] = results['Accuracy'] * 100

    if oversample_rf:
        rf_model = RandomForestClassifier()
        results = train_evaluate_model(rf_model, X_train, X_test, y_train, y_test)
        display_model_results('Random Forest (Oversampling)', results)
        oversampled_results['Random Forest'] = results['Accuracy'] * 100

    if oversample_compare:
        st.header('Model Performance Comparison (Oversampled)')
        fig, ax = plt.subplots()
        sns.barplot(x=list(oversampled_results.keys()), y=list(oversampled_results.values()))
        plt.title('Model Accuracy Comparison (Oversampled Data)')
        plt.ylabel('Accuracy (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)
