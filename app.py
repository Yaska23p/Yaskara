import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import numpy as np

# Simulated user database for login feature
users = {'user1': 'password1', 'user2': 'password2'}

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type='password')
    if st.sidebar.button("Login"):
        if users.get(username) == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

def data_preparation(data):
    st.subheader("Data Preparation")
    if st.button("Clean Data"):
        data.dropna(inplace=True)
        data.to_csv("cleaned_dataset.csv", index=False)
        st.write("Data cleaned and saved as cleaned_dataset.csv")
    return data

def exploratory_data_analysis(data):
    st.subheader("Exploratory Data Analysis")

    # Load cleaned dataset
    cleaned_data = pd.read_csv("cleaned_dataset.csv")

    # Heatmap
    fig, ax = plt.subplots()
    sns.heatmap(cleaned_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Pairplot
    pairplot_fig = sns.pairplot(cleaned_data)
    st.pyplot(pairplot_fig)

def modeling(data, y, feature_names):
    st.write("### 🧠 Latih Model Machine Learning Anda")

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    # Training the model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    st.success("Model berhasil dilatih!")

    # Predictions and accuracy
    y_pred = model.predict(X_test)
    st.write("### 📊 Evaluasi Model")
    st.metric("🎯 Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
    st.text("📋 Classification Report:")
    st.text(classification_report(y_test, y_pred))

    if st.checkbox("Tampilkan Feature Importance"):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        feature_importance = rf_model.feature_importances_
        fig = px.bar(x=feature_names, y=feature_importance, title="Feature Importance")
        st.plotly_chart(fig)
        
    return model

def prediction(model, feature_names):
    st.title("🤖 Prediction")
    st.write("### 🔮 Prediksi dengan Dataset Baru")
    pred_data = st.file_uploader("Upload dataset baru untuk prediksi", type=["csv"])
    if pred_data is not None:
        pred_data = pd.read_csv(pred_data)
        
        # Pastikan hanya kolom yang sesuai digunakan dari dataset prediksi
        pred_data = pred_data[feature_names.intersection(pred_data.columns)]

        st.write("### 📋 Dataset Prediksi Overview")
        st.dataframe(pred_data.head())

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        pred_data_imputed = imputer.fit_transform(pred_data)
        pred_data_imputed_df = pd.DataFrame(pred_data_imputed, columns=pred_data.columns)

        # Predictions
        predictions = model.predict(pred_data_imputed_df)
        pred_data['Prediction'] = predictions
        st.write("### 🔮 Hasil Prediksi")
        st.dataframe(pred_data)

        st.write("### 📈 Visualisasi Prediksi")
        fig_pred = px.bar(pred_data, x=pred_data.index, y='Prediction', title="Prediksi Hasil")
        st.plotly_chart(fig_pred)

def cross_validation(model, data, y):
    st.title("📊 Cross-validation")
    st.write("### 🧪 Evaluasi Model dengan Cross-Validation")

    # Using cross-validation to evaluate the model
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cross_val_scores = cross_val_score(model, data, y, cv=skf, scoring='accuracy')

    st.write("### 📊 Cross-Validation Results")
    st.metric("🎯 Akurasi Rata-rata", f"{np.mean(cross_val_scores):.2f}")
    st.text(f"📋 Scores dari tiap fold: {cross_val_scores}")

    # Displaying the distribution of cross-validation results
    st.write("### 📊 Distribusi Skor Cross-Validation")
    fig_cv, ax = plt.subplots()
    ax.hist(cross_val_scores, bins=5, edgecolor='black')
    ax.set_title("Distribusi Skor Cross-Validation")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Frequency")
    st.pyplot(fig_cv)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False

    if not st.session_state['logged_in']:
        login()
    else:
        st.title('Wine Quality Prediction')
        st.sidebar.title(f"Welcome {st.session_state['username']}")
        
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Data Overview:")
            st.write(data.head())

            if 'quality' in data.columns:
                data = data_preparation(data)
                
                if st.checkbox("Perform EDA"):
                    exploratory_data_analysis(data)
                
                if st.checkbox("Modeling"):
                    scaler = StandardScaler()
                    X = data.iloc[:, :-1]
                    y = data['quality']
                    feature_names = X.columns
                    X_scaled = scaler.fit_transform(X)
                    model = modeling(X_scaled, y, feature_names)
                    
                    if st.checkbox("Make Predictions"):
                        prediction(model, feature_names)
                    
                    if st.checkbox("Perform Cross-validation"):
                        cross_validation(model, X_scaled, y)
            else:
                st.error("The dataset does not contain a 'quality' column.")
        else:
            st.info("Please upload a CSV file to proceed.")

if __name__ == '__main__':
    main()
