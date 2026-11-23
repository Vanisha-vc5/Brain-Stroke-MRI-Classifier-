"""Streamlit-cached loaders shared by every page."""
import streamlit as st

from src.model import load_model
from src.retrieval import load_or_build_gallery


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def get_gallery(_model):
    return load_or_build_gallery(_model)
