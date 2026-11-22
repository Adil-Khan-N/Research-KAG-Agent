"""
A/B Dashboard page for Streamlit multi-page interface.
"""
import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="A/B Pipeline Dashboard", page_icon="📊", layout="wide")
st.title("📊 A/B Pipeline Comparison Dashboard")
st.markdown("Compares **vector_only** vs **hybrid** pipeline performance from logged queries.")

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Refresh Data"):
        st.rerun()

try:
    r = requests.get(f"{API_URL}/ab-summary", timeout=10)
    if r.status_code == 200:
        data = r.json()
        stats = data.get("stats_by_variant", [])
        if stats:
            st.subheader("Summary by Variant")
            st.dataframe(stats, use_container_width=True)
        paired = data.get("paired_comparisons", [])
        if paired:
            st.subheader("Paired Comparisons")
            st.dataframe(paired, use_container_width=True)
    else:
        st.info("No A/B summary data available yet. Run some queries first.")
except Exception as e:
    st.error(f"Cannot connect to API at {API_URL}: {e}")
