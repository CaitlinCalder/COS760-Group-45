#Phase 1 dataset preparation
#Loads human text from HuggingFace and MGT from all three LLMs, merges into merged_dataset.csv. Run before baseline.py.

import os
import re
import pandas as pd
from datasets import load_dataset

print("imports ok")

BASE_PATH      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE_PATH, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

print(f"raw path      : {RAW_PATH}")
print(f"processed path: {PROCESSED_PATH}")


def load_vukuzenzele(lang_code: str, hf_code: str) -> pd.DataFrame:
    ds       = load_dataset("dsfsi/vukuzenzele-monolingual", hf_code)
    frames   = [pd.DataFrame(ds[split]) for split in ds]
    combined = pd.concat(frames, ignore_index=True)

    if "text" in combined.columns:
        combined.rename(columns={"text": "Text_Generated"}, inplace=True)

    combined["Language_Code"]    = lang_code
    combined["Model_Identifier"] = "human"
    combined["Label"]            = 0

    return combined[["Text_Generated", "Language_Code", "Model_Identifier", "Label"]]


print("\nloading Vukuzenzele human text...")
zulu_human    = load_vukuzenzele("zu", "zul");  print(f"  isiZulu  : {len(zulu_human)}")
xhosa_human   = load_vukuzenzele("xh", "xho");  print(f"  isiXhosa : {len(xhosa_human)}")
siswati_human = load_vukuzenzele("ss", "ssw");  print(f"  Siswati  : {len(siswati_human)}")

human_df = pd.concat([zulu_human, xhosa_human, siswati_human], ignore_index=True)
print(f"\ntotal human: {len(human_df)}")

MGT_FILES = {
    "Claude"  : os.path.join(RAW_PATH, "mgt_claude.csv"),
    "ChatGPT" : os.path.join(RAW_PATH, "mgt_chatgbt.csv"),
    "Gemini"  : os.path.join(RAW_PATH, "mgt_gemini.csv"),
}

mgt_frames = []
for model_name, filepath in MGT_FILES.items():
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"MGT file for {model_name} not found: {filepath}")
    df_tmp = pd.read_csv(filepath)
    print(f"\n{model_name}: {len(df_tmp)} records | {df_tmp['Language_Code'].value_counts().to_dict()}")
    mgt_frames.append(df_tmp)

machine_df = pd.concat(mgt_frames, ignore_index=True)
machine_df["Label"] = 1
machine_df = machine_df[["Text_Generated", "Language_Code", "Model_Identifier", "Label"]]

#Remove the ChatGPT numeric tracking artefact that appears in every ChatGPT row and zero human rows.
def clean_text(text: str) -> str:
    text = re.sub(r'ngenombolo\s+\d+-\d+-\d+', '', text)
    text = re.sub(r'\b\d+-\d+-\d+\b', '', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

before = machine_df["Text_Generated"].str.contains(r'\d+-\d+-\d+', regex=True).sum()
machine_df["Text_Generated"] = machine_df["Text_Generated"].apply(clean_text)
after  = machine_df["Text_Generated"].str.contains(r'\d+-\d+-\d+', regex=True).sum()
print(f"\nartefact rows cleaned: {before} -> {after}")

print(f"\ntotal MGT: {len(machine_df)}")
print(machine_df["Language_Code"].value_counts())
print(machine_df["Model_Identifier"].value_counts())

machine_df.to_csv(os.path.join(PROCESSED_PATH, "mgt_text.csv"), index=False)

combined_df = pd.concat([human_df, machine_df], ignore_index=True)
combined_df.dropna(subset=["Text_Generated"], inplace=True)
combined_df["Text_Generated"] = combined_df["Text_Generated"].astype(str).str.strip()
combined_df = combined_df[combined_df["Text_Generated"] != ""]

print(f"\ntotal after cleaning: {len(combined_df)}")
print(combined_df["Language_Code"].value_counts())
print(combined_df["Label"].value_counts())

output_path = os.path.join(PROCESSED_PATH, "merged_dataset.csv")
combined_df.to_csv(output_path, index=False)
print(f"\nsaved: {output_path}  shape: {combined_df.shape}")
