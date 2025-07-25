import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re # Import regex for more flexible pattern matching
import io

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
    query_vec = tfidf.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[:-top_n-1:-1]
    return df.iloc[top_indices]

# --- Definisi Intents ---
intents = [
    {
        "tag": "get_head_data",
        "patterns": ['Saya ingin melihat data teratas dataset rekomendasi tempat wisata','cek data'],
        "action": "display_head_data"
    },
    {
        "tag": "dataset_info",
        "patterns": ["informasi pada dataset ini!"],
        "responses": ['Baiklah, informasi tentang dataset akan ditampilkan 😊'],
        "action": "display_dataset_info"
    },
    {
        "tag": "checking_missing_values",
        "patterns": ["tampilkan missing values!", "tampilkan missing values pada dataset ini!"],
        "responses": ['Baik, berikut akan saya tampilkan missing values / null values 😊'],
        "action": "display_missing_values"
    },
    {
        "tag": "eliminating_missing_values",
        "patterns": ['hapus semua missing values'],
        "responses": "Baiklah, missing values akan saya hapus 😊",
        "action": "eliminate_missing_values"
    },
    {
        "tag": "salam",
        "patterns": ["Assalamualaikum", 'Assalamualaikum'],
        "responses": ["Wa'alaikumussalam, ada yang bisa dibantu?", "Wa'alaikumussalam, apa yang ingin anda tanyakan?"]
    },
    {
        "tag": "greeting",
        "patterns": ["Hai chatbot", "Hai", "Halo chatbot", "Halo", 'Hallo'],
        "responses": ["Halo, apa yang ingin anda tanyakan?", "Halo, apa yang ingin anda tanyakan?", "Hai, apa yang ingin anda tanyakan?"]
    },
    {
        "tag": "introduction_chatbot",
        "patterns": ["is your name?", "are you?"],
        "responses": ["I'm Chatbot"]
    },
    {
        "tag": "goodbye",
        "patterns": ["Sampai jumpa", "Selamat tinggal"],
        "responses": ["Sampai jumpa kembali", "Selamat tinggal juga"]
    },
    {
        "tag": "thanks",
        "patterns": ["Terima kasih atas penjelasannya 😊", "Terima kasih banyak", "Terima kasih ya"],
        "responses": ["Sama-sama 😊", "Tidak masalah", "Senang bisa membantu 😊"]
    },
    {
        "tag": 'asking',
        "patterns": ['Saya ada pertanyaan lagi', 'saya ingin bertanya lagi?', 'Saya ingin bertanya lagi'],
        "responses": ['Silahkan, apa yang ingin anda tanyakan?', 'Boleh, apa yang bisa saya bantu?']
    },
    {
        "tag": "about_identity",
        "patterns": ["kamu?", "anda?"],
        "responses": ["Saya adalah chatbot", "Saya adalah Chatbot yang bisa membantu anda"]
    },
    {
        "tag": "chatbot_capability",
        "patterns": ["Apa yang bisa anda lakukan?"],
        "responses": [
            "Saya bisa menjawab pertanyaan dan memberikan bantuan tentang informasi seperti berikut ini: \n",
            "1. Menjawab pertanyaan sapaan sederhana seperti **Hai Chatbot**, **Halo Chatbot!**, dan salam seperti **Assalamualaikum** \n",
            "2. Menampilkan dataset teratas rekomendasi tempat wisata dengan perintah **Saya ingin melihat data teratas dataset rekomendasi tempat wisata** \n",
            "3. Menampilkan informasi dataset dengan **tampilkan informasi dataset ini!** \n",
            "4. Menampilkan missing values dengan **tampilkan missing values pada dataset ini!** \n",
            "5. Menghapus missing values dengan **hapus semua missing values** \n",
            "6. Merekomendasikan tempat wisata dengan kategori Budaya dengan cara **berikan rekomendasi tempat wisata di Bandung dengan kategori Budaya**"
        ]
    },
    {
        "tag": "help",
        "patterns": ["Tolong", "Saya butuh bantuan", "yang harus saya lakukan?"],
        "responses": ["Tentu, apa yang anda butuhkan?", "Saya disini untuk membantu, apa kendala anda?", "Apa yang bisa saya bantu?"]
    },
    {
        "tag": "age",
        "patterns": ["umur anda?", "usia anda sekarang?"],
        "responses": ["Saya tidak punya umur, saya hanyalah chatbot", "Saya terlahir di dunia digital, umur hanyalah angka"]
    },
    {
        "tag": "weather",
        "patterns": ["cuaca hari ini?", "cuaca hari ini baik?"],
        "responses": ["Maaf, saya tidak bisa memberikan informasi cuaca hari ini", "Anda bisa melihat nya pada website atau aplikasi lain"]
    },
    {
        "tag": "random_input", # Catch-all for irrelevant inputs
        "patterns": ["abcdef", "skdjafnoasf", "asjfdnkajfiuaf", "121uhui***66JHAFHSF"], # Example patterns, you might make this more robust
        "responses": ["Maaf, saya tidak mengerti apa yang anda maksud. Bisakah Anda mengulanginya dengan cara lain atau bertanya tentang rekomendasi wisata?", "Saya tidak memahami pertanyaan Anda. Coba tanyakan tentang rekomendasi wisata atau hal lain yang terdaftar dalam kemampuan saya."]
    },
    {
        "tag": "Budaya",
        "patterns": ['rekomendasi tempat wisata di Bandung dengan kategori Budaya'],
        "responses": ["Berikut rekomendasi tempat wisata yang anda cari"],
        "action": "recommend_bandung_budaya" # Add a specific action for this intent
    }
]

# Function to find intent based on user input
def find_intent(user_input):
    user_input_lower = user_input.lower()
    for intent in intents:
        for pattern in intent['patterns']:
            # Using regex for more flexible matching, especially for patterns that might be partial
            # or contain special characters
            if re.search(r'\b' + re.escape(pattern.lower()) + r'\b', user_input_lower):
                return intent
    return None # No intent found

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
                    st.info(f"**Deskripsi:** {row['Description']}...")
                    maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                    st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)          
            else:
                st.markdown(message["content"])

    # Terima input dari pengguna
    if prompt := st.chat_input("Apa yang Anda cari?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Check for specific intents first
            matched_intent = find_intent(prompt)

            if matched_intent:
                # Handle responses for matched intents
                if "responses" in matched_intent and matched_intent["responses"]:
                    response = matched_intent["responses"][0] if isinstance(matched_intent["responses"], list) else matched_intent["responses"]
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                # Handle specific actions based on intent
                if "action" in matched_intent:
                    if matched_intent["action"] == "display_head_data":
                        st.markdown("Berikut adalah 5 baris pertama dari dataset:")
                        st.dataframe(tourism_df.head())
                        st.session_state.messages.append({"role": "assistant", "content": tourism_df.head()})
                    elif matched_intent["action"] == "display_dataset_info":
                        buffer = io.StringIO()
                        tourism_df.info(buf=buffer)
                        s = buffer.getvalue()
                        st.text(s)
                        st.session_state.messages.append({"role": "assistant", "content": s})
                    elif matched_intent["action"] == "display_missing_values":
                        missing_values = tourism_df.isnull().sum()
                        missing_values_df = missing_values[missing_values > 0]
                        if not missing_values_df.empty:
                            st.write("Jumlah missing values per kolom:")
                            st.dataframe(missing_values_df.rename("Missing Values").to_frame())
                            st.session_state.messages.append({"role": "assistant", "content": missing_values_df.rename("Missing Values").to_frame()})
                        else:
                            st.markdown("Tidak ada missing values dalam dataset.")
                            st.session_state.messages.append({"role": "assistant", "content": "Tidak ada missing values dalam dataset."})
                    elif matched_intent["action"] == "eliminate_missing_values":
                        initial_rows = tourism_df.shape[0]
                        tourism_df.dropna(inplace=True)
                        rows_after_drop = tourism_df.shape[0]
                        st.markdown(f"Missing values telah dihapus. Jumlah baris data setelah penghapusan: {rows_after_drop} (sebelumnya {initial_rows}).")
                        st.session_state.messages.append({"role": "assistant", "content": f"Missing values telah dihapus. Jumlah baris data setelah penghapusan: {rows_after_drop} (sebelumnya {initial_rows})."})
                        # Re-create TF-IDF model if data changes
                        tfidf_vectorizer, tfidf_matrix = create_tfidf_model(tourism_df['combined_features'])
                    elif matched_intent["action"] == "recommend_bandung_budaya":
                        # Filter for specific recommendations
                        bandung_budaya_recommendations = tourism_df[
                            (tourism_df['City'].str.contains('Bandung', case=False)) &
                            (tourism_df['Category'].str.contains('Budaya', case=False))
                        ]
                        if not bandung_budaya_recommendations.empty:
                            st.markdown("Berikut beberapa rekomendasi tempat wisata Budaya di Bandung:")
                            for index, row in bandung_budaya_recommendations.head(5).iterrows(): # Show top 5
                                st.markdown("---")
                                st.subheader(row['Place_Name'])
                                st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                                st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                                st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                                st.info(f"**Deskripsi:** {row['Description'][:150]}...")
                                maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                                st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)
                            st.session_state.messages.append({"role": "assistant", "content": bandung_budaya_recommendations.head(5)}) # Store the displayed portion
                        else:
                            response = "Maaf, saya tidak menemukan rekomendasi wisata Budaya di Bandung."
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                # If no specific intent is matched, fall back to TF-IDF recommendations
                with st.spinner("Sedang mencari tempat terbaik untuk Anda..."):
                    recommendations_df = get_recommendations(prompt, tourism_df, tfidf_vectorizer, tfidf_matrix, top_n=5)

                    if not recommendations_df.empty:
                        st.markdown(f"Tentu, ini **{len(recommendations_df)} rekomendasi** yang paling cocok untuk Anda:")
                        for index, row in recommendations_df.iterrows():
                            st.markdown("---")
                            st.subheader(row['Place_Name'])
                            st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                            st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                            st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                            st.info(f"**Deskripsi:** {row['Description'][:150]}...")
                            maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                            st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)

                        st.session_state.messages.append({"role": "assistant", "content": recommendations_df})
                    else:
                        # Fallback for general recommendations if TF-IDF also finds nothing
                        response = "Maaf, saya tidak dapat menemukan tempat yang cocok dengan deskripsi Anda. Silakan coba kata kunci yang lain."
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})

