import streamlit as st

# Membaca file HTML dari folder pages
with open("pages/about.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Menampilkan HTML di Streamlit
st.markdown(html_content, unsafe_allow_html=True)

