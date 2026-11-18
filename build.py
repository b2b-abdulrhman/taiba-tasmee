"""Build index.html from app.src.html + taiba_data.json (run: python3 build.py)."""
import pathlib
here = pathlib.Path(__file__).parent
src = (here / "app.src.html").read_text(encoding="utf-8")
data = (here / "taiba_data.json").read_text(encoding="utf-8")
(here / "index.html").write_text(src.replace("/*DATA*/null", data), encoding="utf-8")
print("built index.html")
