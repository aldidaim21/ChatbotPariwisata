import streamlit as st
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset wisata
@st.cache_data
def load_data():
    df = pd.read_csv("tourism_with_id.csv")  # pastikan file ada
    df.dropna(subset=['Description', 'Category'], inplace=True)
    return df

tourism_df = load_data()

# Preprocessing untuk SVM
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tourism_df['Description'])
y = tourism_df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

valid_categories = tourism_df['Category'].unique()

# Intents AIML-style
intents = [
    {
        "tag": "salam",
        "patterns": ["Assalamualaikum","hallo"],
        "responses": ["Wa'alaikumussalam, ada yang bisa dibantu?"]
    },
    {
        "tag": "greeting",
        "patterns": ["Hai chatbot", "Hai", "Halo chatbot", "Halo"],
        "responses": ["Halo, apa yang ingin Anda tanyakan?"]
    },
    {
        "tag": "thanks",
        "patterns": ["Terima kasih", "Makasih", "Thank you"],
        "responses": ["Sama-sama 😊", "Senang bisa membantu!"]
    },
    {
        "tag": "get_head_data",
        "patterns": ["Saya ingin melihat data teratas dataset rekomendasi tempat wisata"],
        "responses": ["Berikut adalah data teratas dari dataset:"]
    },
    {
        "tag": "dataset_info",
        "patterns": ["informasi pada dataset ini", "info dataset"],
        "responses": ["Berikut informasi lengkap dari dataset ini:"]
    },
    {
        "tag": "checking_missing_values",
        "patterns": ["tampilkan missing values", "missing data"],
        "responses": ["Berikut adalah missing values dalam dataset:"]
    },
    {
        "tag": "eliminating_missing_values",
        "patterns": ["hapus semua missing values"],
        "responses": ["Baik, saya akan menghapus missing values sekarang."]
    },
    {
        "tag": "random_input",
        "patterns": ["abcdef", "sdfjhsd", "xzyyzzz", "clean","jokowi"],
        "responses": ["Maaf, saya tidak mengerti apa yang Anda maksud 😅"]
    }
]

def match_intent(user_input):
    user_input = user_input.lower()
    for intent in intents:
        for pattern in intent["patterns"]:
            if re.search(pattern.lower(), user_input):
                return random.choice(intent["responses"]), intent["tag"]
    return None, None

def get_svm_recommendations(user_input, model, df, categories, top_n=5):
    vect_input = vectorizer.transform([user_input])
    predicted_category = model.predict(vect_input)[0]
    if predicted_category in categories:
        recommendations = df[df['Category'] == predicted_category].head(top_n)
        return predicted_category, recommendations
    return None, pd.DataFrame()

# UI Chatbot
st.title("🗺️ Chatbot Rekomendasi Wisata")
st.caption("Berbasis SVM + Pola Percakapan")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan destinasi atau info data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang diproses..."):
            response_text, intent_tag = match_intent(prompt)

            if intent_tag == "get_head_data":
                st.markdown(response_text)
                st.dataframe(tourism_df.head())
            elif intent_tag == "dataset_info":
                st.markdown(response_text)
                st.text(str(tourism_df.info()))
            elif intent_tag == "checking_missing_values":
                st.markdown(response_text)
                st.dataframe(tourism_df.isnull().sum())
            elif intent_tag == "eliminating_missing_values":
                st.markdown(response_text)
                tourism_df.dropna(inplace=True)
            elif response_text:
                st.markdown(response_text)
            else:
                predicted_cat, recommendations_df = get_svm_recommendations(prompt, svm_model, tourism_df, valid_categories)
                if predicted_cat and not recommendations_df.empty:
                    st.markdown(f"Kategori yang cocok: **{predicted_cat}**")
                    for _, row in recommendations_df.iterrows():
                        st.markdown("---")
                        st.subheader(row['Place_Name'])
                        st.markdown(f"**Kategori:** `{row['Category']}` | **Kota:** `{row['City']}`")
                        st.markdown(f"**Harga Tiket:** `Rp {row['Price']:,}`")
                        st.success(f"⭐ **Rating:** {row['Rating']} / 5")
                        st.info(f"**Deskripsi:** {row['Description']}")
                        maps_url = f"https://www.google.com/maps/dir/?api=1&destination={row['Lat']},{row['Long']}"
                        st.link_button("🗺️ Buka Navigasi di Google Maps", maps_url)
                else:
                    st.markdown("😕 Maaf, saya tidak menemukan tempat wisata yang sesuai. Coba kata lain ya!")
