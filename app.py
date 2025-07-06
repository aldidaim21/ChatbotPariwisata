import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Rekomendasi Wisata Indonesia",
    page_icon="🏞️",
    layout="wide"
)

# --- Fungsi untuk Memuat dan Memproses Data ---
@st.cache_data # Cache data agar tidak perlu load ulang setiap kali ada interaksi
def load_and_prepare_data(filepath):
    """
    Memuat data dari file CSV, membersihkan data, dan membuat kolom fitur gabungan.
    """
    df = pd.read_csv(filepath)
    
    # Membersihkan data yang kosong (NaN) agar tidak error
    df['Description'] = df['Description'].fillna('')
    df['Category'] = df['Category'].fillna('')
    df['City'] = df['City'].fillna('')
    df['Place_Name'] = df['Place_Name'].fillna('')

    # KUNCI UTAMA: Membuat kolom 'combined_features'
    # Kolom ini menggabungkan semua informasi teks penting menjadi satu.
    # Ini memungkinkan pencarian yang kuat berdasarkan nama, kategori, kota, dan deskripsi secara bersamaan.
    df['combined_features'] = (
        df['Place_Name'] + ' ' + 
        df['Category'] + ' ' + 
        df['City'] + ' ' + 
        df['Description']
    )
    return df

def get_recommendations(query, df, top_n=5):
 
    tfidf = TfidfVectorizer(stop_words='english')

    # 2. Membuat matriks TF-IDF dari fitur gabungan
    # Ini adalah representasi matematis dari setiap tempat wisata.
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    # 3. Mengubah kueri pengguna menjadi vektor menggunakan vectorizer yang sama
    query_vec = tfidf.transform([query])

    # 4. Menghitung Cosine Similarity
    # Mengukur kemiripan sudut antara vektor kueri dan semua vektor tempat wisata.
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 5. Mengambil indeks dari 'top_n' skor kemiripan tertinggi
    # argsort() mengurutkan dari yang terkecil, jadi kita ambil dari belakang.
    top_indices = cosine_sim.argsort()[:-top_n-1:-1]

    # 6. Mengembalikan DataFrame hasil rekomendasi
    return df.iloc[top_indices]


# --- Tampilan Utama Aplikasi Streamlit ---
st.title("🏞️ Sistem Rekomendasi Wisata Indonesia")
st.markdown("Masukkan deskripsi wisata yang Anda inginkan. Sistem akan mencari tempat yang paling cocok berdasarkan **kategori, kota, dan deskripsi** secara bersamaan.")

# Muat data menggunakan fungsi yang sudah dibuat
try:
    tourism_df = load_and_prepare_data('tourism_with_id.csv')

    # Input dari pengguna
    user_input = st.text_input(
        "Contoh: `wisata alam di bandung` atau `taman bermain untuk keluarga di jakarta`",
        "Rekomendasi tempat wisata Alam Di Bandung"
    )

    if st.button("🔍 Cari Rekomendasi"):
        if user_input:
            with st.spinner('Mencari tempat wisata terbaik untuk Anda...'):
                recommendations = get_recommendations(user_input, tourism_df, top_n=5)
            
            st.success(f"Berikut adalah **{len(recommendations)} rekomendasi teratas** untuk Anda:")

            # Menampilkan hasil rekomendasi
            for index, row in recommendations.iterrows():
                st.markdown("---")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(f"https://placehold.co/400x300?text={row['Place_Name'].replace(' ', '+')}", use_column_width=True)
                
                with col2:
                    st.subheader(row['Place_Name'])
                    st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                    st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                    st.markdown(f"**Koordinat:** `{row['Lat']}, {row['Long']}`")
                    st.info(f"**Deskripsi:** {row['Description']}")

        else:
            st.warning("Mohon masukkan deskripsi pencarian.")

except FileNotFoundError:
    st.error("File `tourism_with_id.csv` tidak ditemukan. Pastikan file tersebut berada di direktori yang sama dengan aplikasi ini.")