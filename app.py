import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Chatbot Rekomendasi Wisata",
    page_icon="🤖",
    layout="wide"
)

# --- Fungsi untuk Memuat dan Memproses Data (Tetap Sama) ---
@st.cache_data # Cache data agar tidak perlu load ulang setiap kali ada interaksi
def load_and_prepare_data(filepath):
    """
    Memuat data dari file CSV, membersihkan data, dan membuat kolom fitur gabungan.
    """
    try:
        df = pd.read_csv(filepath)
        # Membersihkan data yang kosong (NaN) agar tidak error
        df['Description'] = df['Description'].fillna('')
        df['Category'] = df['Category'].fillna('')
        df['City'] = df['City'].fillna('')
        df['Place_Name'] = df['Place_Name'].fillna('')

        # KUNCI UTAMA: Membuat kolom 'combined_features'
        df['combined_features'] = (
            df['Place_Name'] + ' ' +
            df['Category'] + ' ' +
            df['City'] + ' ' +
            df['Description']
        )
        return df
    except FileNotFoundError:
        st.error(f"Error: File `{filepath}` tidak ditemukan. Pastikan file tersebut berada di direktori yang sama.")
        return None

# --- Fungsi untuk Mendapatkan Rekomendasi (Dioptimalkan) ---
@st.cache_resource # Cache model TF-IDF agar lebih efisien
def create_tfidf_model(corpus):
    """Membuat dan melatih model TF-IDF."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(corpus)
    return tfidf, tfidf_matrix

def get_recommendations(query, df, tfidf, tfidf_matrix, top_n=5):
    """
    Menghitung kemiripan dan mengembalikan rekomendasi.
    """
    # Mengubah kueri pengguna menjadi vektor
    query_vec = tfidf.transform([query])
    # Menghitung Cosine Similarity
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    # Mengambil indeks dari 'top_n' skor kemiripan tertinggi
    top_indices = cosine_sim.argsort()[:-top_n-1:-1]
    # Mengembalikan DataFrame hasil rekomendasi
    return df.iloc[top_indices]


# --- Tampilan Utama Aplikasi Chatbot Streamlit ---
st.title("🤖 Chatbot Rekomendasi Wisata Indonesia")
st.markdown("Saya dapat membantu Anda menemukan destinasi wisata. Coba ketik sesuatu seperti `wisata alam di bandung` atau `taman budaya di yogyakarta`.")

# Muat data dan siapkan model
tourism_df = load_and_prepare_data('tourism_with_id.csv')

if tourism_df is not None:
    # Buat model TF-IDF sekali dan cache hasilnya
    tfidf_vectorizer, tfidf_matrix = create_tfidf_model(tourism_df['combined_features'])

    # Inisialisasi riwayat chat di session state jika belum ada
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu untuk merencanakan liburan Anda di Indonesia?"}
        ]

    # Tampilkan semua pesan dari riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Jika konten adalah DataFrame (hasil rekomendasi), tampilkan secara khusus
            if isinstance(message["content"], pd.DataFrame):
                recommendations = message["content"]
                st.markdown(f"Tentu, ini **{len(recommendations)} rekomendasi teratas** yang paling cocok untuk Anda:")
                # Menampilkan hasil rekomendasi
                for index, row in recommendations.iterrows():
                    st.markdown("---")
                    col1, col2 = st.columns([1, 2.5])
                    with col1:
                        # Placeholder gambar yang dinamis
                        st.image(f"https://placehold.co/400x300/a3e635/000000?text={row['Place_Name'].replace(' ', '+')}", use_column_width=True)
                    with col2:
                        st.subheader(row['Place_Name'])
                        st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                        st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                        st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                        st.info(f"**Deskripsi:** {row['Description'][:150]}...") # Tampilkan potongan deskripsi
            else:
                # Jika hanya teks biasa
                st.markdown(message["content"])

    # Terima input dari pengguna menggunakan st.chat_input
    if prompt := st.chat_input("Apa yang Anda cari?"):
        # 1. Tambahkan pesan pengguna ke riwayat dan tampilkan
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Proses input dan hasilkan respons dari asisten (bot)
        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari tempat terbaik untuk Anda..."):
                recommendations_df = get_recommendations(prompt, tourism_df, tfidf_vectorizer, tfidf_matrix, top_n=5)

                if not recommendations_df.empty:
                    # Jika ditemukan, tampilkan dan simpan DataFrame ke riwayat
                    st.markdown(f"Tentu, ini **{len(recommendations_df)} rekomendasi teratas** yang paling cocok untuk Anda:")
                    for index, row in recommendations_df.iterrows():
                        st.markdown("---")
                        col1, col2 = st.columns([1, 2.5]) # Atur rasio kolom
                        with col1:
                            st.image(f"https://placehold.co/400x300/a3e635/000000?text={row['Place_Name'].replace(' ', '+')}", use_column_width=True)
                        with col2:
                            st.subheader(row['Place_Name'])
                            st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                            st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                            st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                            st.info(f"**Deskripsi:** {row['Description']}...") # Tampilkan potongan deskripsi
                    
                    # Simpan hasil rekomendasi (DataFrame) ke session state
                    st.session_state.messages.append({"role": "assistant", "content": recommendations_df})
                else:
                    # Jika tidak ditemukan
                    response = "Maaf, saya tidak dapat menemukan tempat yang cocok dengan deskripsi Anda. Silakan coba kata kunci yang lain."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

