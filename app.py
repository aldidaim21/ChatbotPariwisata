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

# --- Fungsi untuk Memuat dan Memproses Data ---
@st.cache_data
def load_and_prepare_data(filepath):
    """
    Memuat data dari file CSV, membersihkan data, dan membuat kolom fitur gabungan.
    """
    try:
        df = pd.read_csv(filepath)
        df['Description'] = df['Description'].fillna('')
        df['Category'] = df['Category'].fillna('')
        df['City'] = df['City'].fillna('')
        df['Place_Name'] = df['Place_Name'].fillna('')
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

# --- Fungsi untuk Mendapatkan Rekomendasi ---
@st.cache_resource
def create_tfidf_model(corpus):
    """Membuat dan melatih model TF-IDF."""
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(corpus)
    return tfidf, tfidf_matrix

def get_recommendations(query, df, tfidf, tfidf_matrix, top_n=5):
    """
    Menghitung kemiripan dan mengembalikan rekomendasi.
    """
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[:-top_n-1:-1]
    return df.iloc[top_indices]

# --- Tampilan Utama Aplikasi Chatbot Streamlit ---
st.title("🤖 Chatbot Rekomendasi Wisata Indonesia")
st.markdown("Saya dapat membantu Anda menemukan destinasi wisata. Coba ketik sesuatu seperti `wisata alam di bandung` atau `taman budaya di yogyakarta`.")

tourism_df = load_and_prepare_data('tourism_with_id.csv')

if tourism_df is not None:
    tfidf_vectorizer, tfidf_matrix = create_tfidf_model(tourism_df['combined_features'])

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu untuk merencanakan liburan Anda di Indonesia?"}
        ]
    # Tampilkan semua pesan dari riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], pd.DataFrame):
                recommendations = message["content"]
                st.markdown(f"Tentu, ini **{len(recommendations)} rekomendasi teratas** yang paling cocok untuk Anda:")
                for index, row in recommendations.iterrows():
                    st.markdown("---")
                    st.subheader(row['Place_Name'])
                    st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                    st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                    st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                    st.info(f"**Deskripsi:** {row['Description'][:150]}...")
                    # --- PENAMBAHAN TOMBOL NAVIGASI ---
                    maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                    st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)
                    # ------------------------------------            
            else:
                st.markdown(message["content"])

    # Terima input dari pengguna
    if prompt := st.chat_input("Apa yang Anda cari?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Sedang mencari tempat terbaik untuk Anda..."):
                recommendations_df = get_recommendations(prompt, tourism_df, tfidf_vectorizer, tfidf_matrix, top_n=5)

                if not recommendations_df.empty:
                    st.markdown(f"Tentu, ini **{len(recommendations_df)} rekomendasi teratas** yang paling cocok untuk Anda:")
                    for index, row in recommendations_df.iterrows():
                        st.markdown("---")
                        st.subheader(row['Place_Name'])
                        st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                        st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                        st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                        st.info(f"**Deskripsi:** {row['Description'][:150]}...")
                        # --- PENAMBAHAN TOMBOL NAVIGASI (DI SINI JUGA) ---
                        maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                        st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)
                        # ---------------------------------------------

                    st.session_state.messages.append({"role": "assistant", "content": recommendations_df})
                else:
                    response = "Maaf, saya tidak dapat menemukan tempat yang cocok dengan deskripsi Anda. Silakan coba kata kunci yang lain."
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

