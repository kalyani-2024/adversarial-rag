"""Streamlit entry point kept at the repo root so `streamlit run app.py` keeps working.

The UI lives in ui/streamlit_app.py.
"""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "ui" / "streamlit_app.py"), run_name="__main__")
