#Phase 1 dataset preparation, loads human text from HuggingFace and MGT from all three LLMs, merges into merged_dataset.csv. Run before baseline.py.
import os
import re
import pandas as pd
from datasets import load_dataset

print("imports ok")

BASE_PATH= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH= os.path.join(BASE_PATH, "data", "raw")
PROCESSED_PATH= os.path.join(BASE_PATH, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

print(f"raw path: {RAW_PATH}")
print(f"processed path: {PROCESSED_PATH}")


def load_vukuzenzele(lang_code: str, hf_code: str) -> pd.DataFrame:
    ds= load_dataset("dsfsi/vukuzenzele-monolingual", hf_code)
    frames= [pd.DataFrame(ds[split]) for split in ds]
    combined= pd.concat(frames, ignore_index=True)

    if "text" in combined.columns:
        combined.rename(columns={"text": "Text_Generated"}, inplace=True)

    combined["Language_Code"]= lang_code
    combined["Model_Identifier"]= "human"
    combined["Label"]= 0

    return combined[["Text_Generated", "Language_Code", "Model_Identifier", "Label"]]


print("\nloading Vukuzenzele human text...")
zulu_human= load_vukuzenzele("zu", "zul");  print(f"  isiZulu  : {len(zulu_human)}")
xhosa_human= load_vukuzenzele("xh", "xho");  print(f"  isiXhosa : {len(xhosa_human)}")
siswati_human= load_vukuzenzele("ss", "ssw");  print(f"  Siswati  : {len(siswati_human)}")

human_df= pd.concat([zulu_human, xhosa_human, siswati_human], ignore_index=True)
print(f"\ntotal human: {len(human_df)}")

MGT_FILES = {
    "Claude": os.path.join(RAW_PATH, "machine_generated_claude.csv"),
    "ChatGPT": os.path.join(RAW_PATH, "machine_generated_chatgpt-4o.csv"),
    "Gemini": os.path.join(RAW_PATH, "machine_generated_gemini-2.5-pro.csv"),
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

#remove the ChatGPT numeric tracking artefact that appears in every ChatGPT row and zero human rows.
def clean_text(text: str) -> str:
    text= re.sub(r'ngenombolo\s+\d+-\d+-\d+', '', text)
    text= re.sub(r'\b\d+-\d+-\d+\b', '', text)
    text= re.sub(r' {2,}', ' ', text)
    return text.strip()

before= machine_df["Text_Generated"].str.contains(r'\d+-\d+-\d+', regex=True).sum()
machine_df["Text_Generated"] = machine_df["Text_Generated"].apply(clean_text)
after = machine_df["Text_Generated"].str.contains(r'\d+-\d+-\d+', regex=True).sum()
print(f"\nArtefact rows cleaned: {before} -> {after}")

#Remove first sentence from MGT texts that start with formulaic patterns as they appear in 25-34% of MGT texts but <1% of human texts, creating easy detection
FORMULAIC_STARTS = ["Ngaphezulu","Abahlali","Emaphandleni","Umasipala","Ekhuluma","Ngemuva"]

def remove_formulaic_first_sentence(text: str, is_mgt: bool) -> str:
    if not is_mgt:
        return text
    
    #Check if text starts with any formulaic pattern
    starts_with_formula = any(text.startswith(pattern) for pattern in FORMULAIC_STARTS)
    
    if not starts_with_formula:
        return text
    
    #Find the first sentence boundary (. ! ?)
    match = re.search(r'[.!?](?=\s|$)', text)
    
    if match:
        #Remove everything up to and including the first sentence boundary
        remaining = text[match.end():].strip()
        if len(remaining) > 100:  # Only remove if there's substantial text remaining
            return remaining
    
    #If no sentence boundary found or remaining text too short, return original
    return text

before_removal = machine_df["Text_Generated"].apply(
    lambda t: any(t.startswith(p) for p in FORMULAIC_STARTS)
).sum()

machine_df["Text_Generated"] = machine_df.apply(
    lambda row: remove_formulaic_first_sentence(row["Text_Generated"], is_mgt=True),
    axis=1
)

after_removal = machine_df["Text_Generated"].apply(
    lambda t: any(t.startswith(p) for p in FORMULAIC_STARTS)
).sum()

print(f"\nformulaic first sentences removed: {before_removal} -> {after_removal}")

print(f"\ntotal MGT: {len(machine_df)}")
print(machine_df["Language_Code"].value_counts())
print(machine_df["Model_Identifier"].value_counts())

machine_df.to_csv(os.path.join(PROCESSED_PATH, "mgt_text.csv"), index=False)

combined_df = pd.concat([human_df, machine_df], ignore_index=True)
combined_df.dropna(subset=["Text_Generated"], inplace=True)
combined_df["Text_Generated"] = combined_df["Text_Generated"].astype(str).str.strip()
combined_df = combined_df[combined_df["Text_Generated"] != ""]

#truncate all texts to a target character length, cutting at the nearest sentence boundary
def truncate_to_sentence(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    
    #Find the last sentence boundary in the truncated portion
    import re
    boundaries = list(re.finditer(r'[.!?](?=\s|$)', truncated))
    
    if boundaries:
        #Cut at the last sentence boundary found
        last_boundary = boundaries[-1].end()
        return text[:last_boundary].strip()
    else:
        #No sentence boundary found — fall back to character truncation
        return truncated.strip()

TARGET_LENGTH = 800
print(f"\ntruncating texts to ~{TARGET_LENGTH} chars at sentence boundaries...")
before_human = combined_df.loc[combined_df["Label"]==0, "Text_Generated"].str.len().mean()
before_mgt   = combined_df.loc[combined_df["Label"]==1, "Text_Generated"].str.len().mean()

combined_df["Text_Generated"] = combined_df["Text_Generated"].apply(
    lambda t: truncate_to_sentence(t, TARGET_LENGTH)
)

after_human= combined_df.loc[combined_df["Label"]==0, "Text_Generated"].str.len().mean()
after_mgt= combined_df.loc[combined_df["Label"]==1, "Text_Generated"].str.len().mean()

print(f"human avg: {before_human:.0f} -> {after_human:.0f} chars")
print(f"MGT avg  : {before_mgt:.0f} -> {after_mgt:.0f} chars")

print(f"\ntotal after cleaning: {len(combined_df)}")
print(combined_df["Language_Code"].value_counts())
print(combined_df["Label"].value_counts())

output_path = os.path.join(PROCESSED_PATH, "merged_dataset.csv")
combined_df.to_csv(output_path, index=False)
print(f"\nsaved: {output_path}  shape: {combined_df.shape}")
