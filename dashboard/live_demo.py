"""Streamlit live-webcam detection + segmentation demo.

Status: scaffold stub. Populated in Phase 6.
"""

from __future__ import annotations


def main() -> None:
    try:
        import streamlit as st
    except ImportError:
        print("Install the [ui] extras: pip install -e '.[dev,ui]'")
        return

    st.set_page_config(page_title="edge-vision — Live", layout="wide")
    st.title("edge-vision — live detection + segmentation")
    st.info("This demo is a Phase 6 deliverable.")


if __name__ == "__main__":
    main()
