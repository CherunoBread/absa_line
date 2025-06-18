
import streamlit as st
import pandas as pd
from huggingface_hub import login
from google_play_scraper import Sort, reviews as gp_reviews
from preprocessing import predict_topic, predict_sentiment

# Login ke HF (ambil token dari secrets.toml)
login(token=st.secrets.huggingface.token, add_to_git_credential=False)

st.set_page_config(page_title="ABSA LINE Reviews", layout="wide")
st.title("Analisis Sentimen Berbasis Aspek — LINE Reviews")

# Sidebar
n = st.sidebar.selectbox("Jumlah ulasan:", [10,50,100,500,1000], index=2)
if st.sidebar.button("Jalankan Analisis"):
    @st.cache_data
    def scrape(n):
        data, _ = gp_reviews("jp.naver.line.android", lang="id", country="id",
                             sort=Sort.NEWEST, count=n)
        return pd.DataFrame(data)[["content"]]

    df = scrape(n)
    if df.empty:
        st.error("Gagal ambil ulasan.")
        st.stop()

    # Inference
    df["topic"]     = df["content"].apply(predict_topic)
    df["sentiment"] = df["content"].apply(predict_sentiment)

    # Agregasi
    agg = (df.groupby(["topic","sentiment"])
             .size()
             .unstack(fill_value=0)
             .reindex(index=["Topic 1","Topic 2","Topic 3"], fill_value=0))

    st.subheader("Tabel Agregasi")
    st.dataframe(agg, use_container_width=True)

    st.subheader("Bar Chart per Topik")
    st.bar_chart(agg)

    st.subheader("Distribusi Sentimen Keseluruhan")
    overall = df["sentiment"].value_counts().reindex(["Positif","Negatif","Netral"], fill_value=0)
    st.pyplot(overall.plot.pie(autopct="%1.1f%%", ylabel="").get_figure())
