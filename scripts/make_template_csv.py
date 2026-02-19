import json
from pathlib import Path
import pandas as pd

def main():
    repo = Path(__file__).resolve().parents[1]  # C:\ARARAT_MVP
    meta_path = repo / "inference" / "models" / "v1_prostatex" / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cols = meta["features"]

    out = repo / "inference_template_features.csv"
    df = pd.DataFrame([{c: "" for c in cols}])  # 1 linha vazia
    df.to_csv(out, index=False, encoding="utf-8")

    print("OK ->", out)
    print("Colunas:", len(cols))

if __name__ == "__main__":
    main()