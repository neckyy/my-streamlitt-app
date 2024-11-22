import streamlit as st
import pandas as pd
import folium
import geopandas as gpd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from streamlit_folium import folium_static
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


import streamlit as st
from pathlib import Path

# Fungsi untuk memuat file CSS
def load_css(file_name):
    css_path = Path("assets") / file_name
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Panggil fungsi untuk memuat CSS
load_css("styles.css")

# Menyimpan data login (username dan password)
users = {"admin": "admin123", "user": "pass123"}  # username : password

# Fungsi untuk form login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username in users and users[username] == password:
            # Menyimpan status login di session_state
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Berhasil!")
            st.rerun()  # Refresh untuk menampilkan sidebar setelah login
        else:
            st.error("Username atau Password Salah!")

# Mengecek apakah pengguna sudah login
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    # Jika belum login, tampilkan form login
    login()
else:
    # Jika sudah login, tampilkan sidebar dengan opsi halaman
    st.sidebar.title("Halaman")
    page = st.sidebar.radio("Pilih Halaman", ("Dashboard", "Eksplorasi Data Analysis", "Model Machine Learning", "Benchmarking Algoritma"))
    
    if page == "Dashboard":

                # Menambahkan judul utama di dashboard
                st.title("Dashboard Analisis Protes & Kekerasan di Indonesia")

                # Menambahkan deskripsi tentang tujuan dashboard
                st.markdown("""
                    Dashboard ini membantu menganalisis data terkait protes dan kekerasan di Indonesia berdasarkan data dari **Armed Conflict Location & Event Data Project (ACLED)**.
                    Anda dapat menjelajahi tren, visualisasi data interaktif, dan analisis mendalam.
                """)


                # Data tren
                data = pd.DataFrame({
                    'Tahun': [2016, 2017, 2018, 2019,],
                    'Protes': [50, 60, 120, 80,],
                    'Kekerasan': [30, 40, 90, 60,]
                })

                # Menambahkan grafik interaktif menggunakan Plotly
                st.header("Grafik Interaktif: Tren Protes dan Kekerasan")
                fig_interaktif = px.line(
                    data,
                    x="Tahun",
                    y=["Protes", "Kekerasan"],
                    labels={"value": "Jumlah Kejadian", "variable": "Kategori", "Tahun": "Tahun"},
                    title="Tren Protes dan Kekerasan di Indonesia (2016-2019)",
                    markers=True
                )
                fig_interaktif.update_traces(line_shape="spline", mode="lines+markers")
                st.plotly_chart(fig_interaktif)

                # Menambahkan analisis visual menggunakan Seaborn
                st.header("Komparasi Protes dan Kekerasan")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=data.melt(id_vars='Tahun', var_name='Kategori', value_name='Jumlah'), 
                            x='Tahun', y='Jumlah', hue='Kategori', palette='Set2', ax=ax)
                ax.set_title("Jumlah Protes dan Kekerasan di Indonesia (2016-2019)", fontsize=16)
                ax.set_xlabel("Tahun", fontsize=14)
                ax.set_ylabel("Jumlah Kejadian", fontsize=14)
                st.pyplot(fig)

                # Menambahkan analisis interaktif dengan slider
                st.header("Analisis Berdasarkan Tahun")
                tahun = st.slider("Pilih Tahun", 2016, 2017, 2018)
                st.write(f"Data untuk tahun {tahun}:")
                # Filter data berdasarkan tahun
                data_tahun = data[data['Tahun'] == tahun]
                st.write(data_tahun)

                # Menambahkan input teks untuk pengunjung
                st.header("Masukkan Pendapat Anda")
                user_input = st.text_area("Apa pendapat Anda tentang protes dan kekerasan di Indonesia?", "")
                if user_input:
                    st.success("Terima kasih atas pendapat Anda! Kami akan mempertimbangkan masukan Anda.")
                else:
                    st.info("Tunggu masukan Anda!")

    elif page == "Eksplorasi Data Analysis":
        st.write("Analisis Potensi Tindak Kekerasan di Indonesia")

        # Menampilkan judul aplikasi dan garis pemisah
        st.markdown("<h1 style='text-align: center; color: #2a3d4f;'>Eksplorasi Data Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Memuat data dengan cache_data
        @st.cache_data
        def load_data(uploaded_file):
            data = pd.read_csv(uploaded_file)  # Membaca dataset yang diunggah
            return data

        # Upload file dataset sekali
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

        if uploaded_file is not None:
            data = load_data(uploaded_file)  # Memanggil fungsi untuk memuat data
            st.write(data.head())  # Menampilkan beberapa baris data untuk memastikan dataset terunggah

        # Sidebar untuk navigasi
        st.sidebar.title("Dasbor Analisis")
        main_option = st.sidebar.selectbox(
            "Pilih Menu",
            ("Eksplorasi Data Analysis")
        )

        # Menu Eksplorasi Data Analysis
        if main_option == "Eksplorasi Data Analysis":
            exploration_option = st.sidebar.selectbox(
                "Pilih Submenu",
                ("Aktor", "Analisis Bivariate", "Analisis Korelasi", "Clustering", "Insight Realtime", "Jumlah Korban", "Kejadian Perbulan tiap Tahun", "Pemisahan Data Berdasarkan Lokasi", "Plot Tren Tahunan")
            )

            if uploaded_file is not None:
                if exploration_option == "Jumlah Korban":
                    plt.figure(figsize=(10, 6))
                    sns.histplot(data['fatalities'], kde=True, bins=30)
                    plt.title('Distribusi Jumlah Korban')
                    plt.xlabel('Jumlah Korban')
                    plt.ylabel('Frekuensi')
                    st.pyplot(plt)

                elif exploration_option == "Pristiwa per Tahun":
                    plt.figure(figsize=(10, 6))
                    sns.countplot(x='year', data=data)
                    plt.title('Jumlah Kejadian per Tahun')
                    plt.xlabel('Tahun')
                    plt.ylabel('Jumlah Kejadian')
                    st.pyplot(plt)

                elif exploration_option == "Analisis Bivariate":
                    plt.figure(figsize=(10,6))
                    sns.boxplot(x='event_type', y='fatalities', data=data)
                    plt.title('Jumlah Korban Berdasarkan Jenis Peristiwa')
                    plt.xlabel('Jenis Peristiwa')
                    plt.ylabel('Jumlah Korban')
                    plt.xticks(rotation=45)
                    st.pyplot(plt)

                elif exploration_option == "Analisis Korelasi":
                    correlation_matrix = data[['fatalities', 'year', 'latitude', 'longitude']].corr()
                    plt.figure(figsize=(8,6))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.title('Matriks Korelasi')
                    st.pyplot(plt)

                elif exploration_option == "Pemisahan Data Berdasarkan Lokasi":
                    plt.figure(figsize=(10,6))
                    sns.countplot(x='admin1', data=data)
                    plt.title('Distribusi Peristiwa Berdasarkan Provinsi')
                    plt.xlabel('Provinsi')
                    plt.ylabel('Jumlah Kejadian')
                    plt.xticks(rotation=90)
                    st.pyplot(plt)

                elif exploration_option == "Kejadian Perbulan tiap Tahun":
                    data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')
                    data['month'] = data['event_date'].dt.month
                    plt.figure(figsize=(12, 6))
                    sns.countplot(x='month', hue='year', data=data, palette='Set2')
                    plt.title('Jumlah Kejadian per Bulan dan per Tahun')
                    plt.xlabel('Bulan')
                    plt.ylabel('Jumlah Kejadian')
                    plt.legend(title='Tahun', bbox_to_anchor=(1.05, 1), loc='upper left')
                    st.pyplot(plt)

                elif exploration_option == "Aktor":
                    if 'actor1' in data.columns:
                        plt.figure(figsize=(10,6))
                        sns.countplot(y='actor1', data=data, order=data['actor1'].value_counts().index)
                        plt.title('Distribusi Aktor dalam Peristiwa')
                        plt.xlabel('Jumlah Kejadian')
                        plt.ylabel('Aktor Utama')
                        st.pyplot(plt)
                    else:
                        st.error("Kolom 'actor1' tidak ditemukan dalam dataset.")

                elif exploration_option == "Plot Tren Tahunan":
                    # Plot tren tahunan
                    fig, ax = plt.subplots()
                    data.groupby('year')['fatalities'].sum().plot(ax=ax)
                    ax.set_title('Jumlah Fatalitas per Tahun')
                    ax.set_xlabel('Tahun')
                    ax.set_ylabel('Jumlah Fatalitas')
                    st.pyplot(fig)

                elif exploration_option == "Clustering":
                    # Clustering Peristiwa
                    st.subheader("Clustering Peristiwa")
                    features = data[['latitude', 'longitude']]
                    kmeans = KMeans(n_clusters=5, random_state=42).fit(features)
                    data['cluster'] = kmeans.labels_

                    # Plot cluster geografis
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.scatterplot(data=data, x='longitude', y='latitude', hue='cluster', palette='tab10')
                    ax.set_title('Cluster Peristiwa')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    st.pyplot(fig)

                elif exploration_option == "Insight Realtime":
                    data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')

                    st.subheader("Insight Realtime")
                    st.write(f"Jumlah kejadian: {len(data)}")
                    st.write(f"Total fatalities: {data['fatalities'].sum()}")
                    recent_data = data[data['event_date'] > (datetime.now() - pd.Timedelta(days=30))]
                    st.write(f"Jumlah kejadian terakhir 30 hari: {len(recent_data)}")
                    st.write(f"Fatalities dalam 30 hari terakhir: {recent_data['fatalities'].sum()}")

        # Footer aplikasi
        st.markdown("---")
        st.markdown("© 2024 Alfonso Tuanahope")

    elif page == "Model Machine Learning":
        st.write("Analisis Potensi Tindak Kekerasan di Indonesia")


                    # Menampilkan judul aplikasi dan garis pemisah
        st.markdown("<h1 style='text-align: center; color: #2a3d4f;'>Model Machine Learning</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Memuat data dengan cache_data
        @st.cache_data
        def load_data(uploaded_file):
            data = pd.read_csv(uploaded_file)  # Membaca dataset yang diunggah
            return data

        # Upload file dataset sekali
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

        if uploaded_file is not None:
            data = load_data(uploaded_file)  # Memanggil fungsi untuk memuat data
            st.write(data.head())  # Menampilkan beberapa baris data untuk memastikan dataset terunggah

        # Sidebar untuk navigasi
        st.sidebar.title("Dasbor Analisis")
        main_option = st.sidebar.selectbox(
            "Pilih Menu",
            ("Model Machine Learning")
        )

        # Menu Machine Learning
        if main_option == "Model Machine Learning":
            machine_learning_option = st.sidebar.selectbox(
                "Pilih Submenu Model Machine Learning",
                ("Logistic Regression", "Model Klasifikasi", "Prediksi Fatalitas", "Random Forest", "Support Vector Machine")
            )    

            if uploaded_file is not None:
                # Preprocessing data
                X = data.drop(columns=['fatalities'])  # Sesuaikan dengan kolom target
                y = (data['fatalities'] > 0).astype(int)  # Target biner, fatalities > 0

                # Identifikasi kolom kategorikal dan numerik
                categorical_cols = ['event_id_cnty', 'sub_event_type', 'actor1', 'region', 'country', 
                                    'admin1', 'admin2', 'location', 'source', 'source_scale', 'iso3']
                numerical_cols = ['year', 'latitude', 'longitude']

                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                        ('num', 'passthrough', numerical_cols)
                    ]
                )

                # Split data menjadi latih dan uji
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if machine_learning_option == "Logistic Regression":
                    st.subheader("Model Logistic Regression")
                    pipeline_lr = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
                    ])
                    pipeline_lr.fit(X_train, y_train)
                    y_pred_lr = pipeline_lr.predict(X_test)
                    
                    # Akurasi dan laporan klasifikasi
                    st.write(f"Accuracy Logistic Regression: {accuracy_score(y_test, y_pred_lr)}")
                    st.text(f"Classification Report:\n{classification_report(y_test, y_pred_lr)}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_test, y_pred_lr)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Prediksi 0', 'Prediksi 1'], yticklabels=['Sebenarnya 0', 'Sebenarnya 1'])
                    plt.xlabel('Prediksi')
                    plt.ylabel('Sebenarnya')
                    plt.title('Confusion Matrix - Logistic Regression')
                    st.pyplot(fig)
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_test, pipeline_lr.predict_proba(X_test)[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic - Logistic Regression')
                    plt.legend(loc='lower right')
                    st.pyplot(plt)

                elif machine_learning_option == "Prediksi Fatalitas":
                    # Model Machine Learning untuk Prediksi Fatalitas
                    st.subheader("Model Prediksi Fatalitas")
                    from sklearn.linear_model import LinearRegression
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import mean_squared_error, r2_score

                    # Misalnya kita menggunakan 'fatalities' sebagai target dan 'year', 'latitude', 'longitude' sebagai fitur
                    X = data[['year', 'latitude', 'longitude']]
                    y = data['fatalities']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluasi model
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"Mean Squared Error: {mse}")
                    st.write(f"R-squared: {r2}")

                # Model Random Forest
                elif machine_learning_option == "Random Forest":
                    st.subheader("Model Random Forest")
                    pipeline_rf = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
                    ])
                    pipeline_rf.fit(X_train, y_train)
                    y_pred_rf = pipeline_rf.predict(X_test)
                    st.write(f"Accuracy Random Forest: {accuracy_score(y_test, y_pred_rf)}")



                # Model Support Vector Machine
                elif machine_learning_option == "Support Vector Machine":
                    st.subheader("Model Support Vector Machine")
                    pipeline_svm = Pipeline(steps=[
                        ('preprocessor', preprocessor),
                        ('classifier', SVC(kernel='linear', random_state=42))
                    ])
                    pipeline_svm.fit(X_train, y_train)
                    y_pred_svm = pipeline_svm.predict(X_test)
                    st.write(f"Accuracy SVM: {accuracy_score(y_test, y_pred_svm)}")

                elif machine_learning_option == "Model Klasifikasi":
                    # Implementasi model klasifikasi
                    st.subheader("Model Klasifikasi")
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.metrics import accuracy_score

                    X = data[['year', 'latitude', 'longitude']]
                    y = data['event_type']  # Misalnya kita klasifikasikan berdasarkan 'event_type'
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluasi model
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Accuracy: {accuracy}")

         # Footer aplikasi
        st.markdown("---")
        st.markdown("© 2024 Alfonso Tuanahope")


    elif page == "Benchmarking Algoritma":
        st.write("Analisis Potensi Tindak Kekerasan di Indonesia")

                # Menampilkan judul aplikasi dan garis pemisah
        st.markdown("<h1 style='text-align: center; color: #2a3d4f;'>Benchmarking Algoritma</h1>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Memuat data dengan cache_data
        @st.cache_data
        def load_data(uploaded_file):
            data = pd.read_csv(uploaded_file)  # Membaca dataset yang diunggah
            return data

        # Upload file dataset sekali
        uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

        if uploaded_file is not None:
            data = load_data(uploaded_file)  # Memanggil fungsi untuk memuat data
            st.write(data.head())  # Menampilkan beberapa baris data untuk memastikan dataset terunggah

        # Sidebar untuk navigasi
        st.sidebar.title("Dasbor Analisis")
        main_option = st.sidebar.selectbox(
            "Pilih Menu",
            ("Benchmarking Algoritma")
        )

        # Menu Benchmarking Algoritma
        if main_option == "Benchmarking Algoritma":
            benchmarking_option = st.sidebar.selectbox(
                "Pilih Metode Evaluasi",
                ("Akurasi", "Presisi", "Recall", "F-1 Score", "AUC-ROC")
            )
            
            if uploaded_file is not None:
                # Preprocessing data
                X = data.drop(columns=['fatalities'])  # Sesuaikan dengan kolom target
                y = (data['fatalities'] > 0).astype(int)  # Target biner, fatalities > 0

                # Identifikasi kolom kategorikal dan numerik
                categorical_cols = ['event_id_cnty', 'sub_event_type', 'actor1', 'region', 'country', 
                                    'admin1', 'admin2', 'location', 'source', 'source_scale', 'iso3']
                numerical_cols = ['year', 'latitude', 'longitude']

                # Menangani kolom tanggal jika ada
                if 'event_date' in data.columns:
                    data['event_date'] = pd.to_datetime(data['event_date'], errors='coerce')

                # Preprocessing untuk kolom numerik dan kategorikal
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                        ('num', Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='mean')),  # Imputasi untuk fitur numerik
                            ('scaler', StandardScaler())  # Normalisasi fitur numerik
                        ]), numerical_cols)
                    ]
                )
                # Split data menjadi latih dan uji
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Menambahkan preprocessing dan SVM dalam pipeline
                pipeline_svm = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('classifier', SVC(kernel='linear', random_state=42, probability=True))
                ])
                
                # Melatih model
                pipeline_svm.fit(X_train, y_train)
                
                # Prediksi
                y_pred = pipeline_svm.predict(X_test)

                # Evaluasi berdasarkan menu yang dipilih
                if benchmarking_option == "Akurasi":
                    accuracy = accuracy_score(y_test, y_pred)
                    st.write(f"Akurasi: {accuracy}")

                elif benchmarking_option == "Presisi":
                    precision = precision_score(y_test, y_pred)
                    st.write(f"Presisi: {precision}")

                elif benchmarking_option == "Recall":
                    recall = recall_score(y_test, y_pred)
                    st.write(f"Recall: {recall}")

                elif benchmarking_option == "F-1 Score":
                    f1 = f1_score(y_test, y_pred)
                    st.write(f"F1-Score: {f1}")

                elif benchmarking_option == "AUC-ROC":
                    auc = roc_auc_score(y_test, pipeline_svm.predict_proba(X_test)[:, 1])
                    st.write(f"AUC-ROC: {auc}")

        # Footer aplikasi
        st.markdown("---")
        st.markdown("© 2024 Alfonso Tuanahope")

    # Tombol Logout (untuk keluar dari aplikasi)
    if st.button("Logout"):
        st.session_state.logged_in = False  # Menghapus status login
        del st.session_state.username
        st.rerun()  # Refresh halaman ke halaman login