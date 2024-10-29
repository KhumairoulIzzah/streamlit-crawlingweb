import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Contoh Dataset Internal
data = {
    "teks": [
        "Pemerintah kota Jombang melakukan revitalisasi alun-alun.",
        "Nasionalisme harus dijaga agar Indonesia tetap bersatu.",
        "Festival budaya di Jombang meriah dengan banyak peserta.",
        "Kebijakan ekonomi nasional akan difokuskan pada peningkatan investasi.",
        "Bupati Jombang meresmikan pasar rakyat baru.",
        "Presiden membahas kebijakan luar negeri dengan negara tetangga."
    ],
    "kategori": ["Jombangku", "Nasional", "Jombangku", "Nasional", "Jombangku", "Nasional"]
}

# Memisahkan Fitur dan Target
X = data["teks"]
y = data["kategori"]

# Split Dataset Menjadi Training dan Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi TF-IDF dan Transformasi Data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Latih Model Logistic Regression
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Prediksi Data Testing dan Hitung Akurasi Model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Judul Aplikasi
st.title("Klasifikasi Kategori Berita Dari Website Warga Jombang")

# Tampilkan Akurasi Model
st.write(f"**Akurasi Model:** {accuracy:.2f}")

# Input Teks Berita
input_text = st.text_area("Masukkan Teks Berita")

# Tombol Prediksi
if st.button("Prediksi Kategori"):
    if input_text.strip():
        # Transformasi TF-IDF dan Prediksi Probabilitas
        input_tfidf = vectorizer.transform([input_text])
        probabilities = model.predict_proba(input_tfidf)[0]

        # Ambil Kategori dan Probabilitas Terkait
        kategori_prediksi = model.classes_[probabilities.argmax()]
        probabilitas_prediksi = probabilities.max()

        # Tampilkan Hasil Prediksi dan Probabilitas
        st.success(f"Hasil Prediksi: **{kategori_prediksi}**")
        st.write(f"**Tingkat Keyakinan:** {probabilitas_prediksi:.2f}")
    else:
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
