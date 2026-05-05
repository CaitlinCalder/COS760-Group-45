#phase 1 dataset preparation for COS760 group 45
#this script loads the human text from huggingface and the machine generated text,
#merges them into one clean dataset and saves it to data/processed/merged_dataset.csv
#run this script before baseline.py

#run from repo root:
#pip install datasets pandas
#python code/dataset_prep.py

import os
import pandas as pd
from datasets import load_dataset

print("imports ok")

#build paths relative to the repo root so the script works from any machine
BASE_PATH      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH       = os.path.join(BASE_PATH, "data", "raw")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

print(f"raw data path      : {RAW_PATH}")
print(f"processed data path: {PROCESSED_PATH}")


def load_vukuzenzele(lang_code: str, hf_code: str) -> pd.DataFrame:
    """load all splits of the vukuzenzele monolingual dataset for one language
    and return a standardised dataframe with the four columns we need.

    lang_code is the short code used internally e.g. 'zu', 'xh', 'ss'
    hf_code is the huggingface config name e.g. 'zul', 'xho', 'ssw'
    """
    ds = load_dataset("dsfsi/vukuzenzele-monolingual", hf_code)

    #combine all available splits (train, test, validation) into one dataframe
    frames   = [pd.DataFrame(ds[split]) for split in ds]
    combined = pd.concat(frames, ignore_index=True)

    #the huggingface dataset uses 'text' but our mgt files use 'Text_Generated'
    #so we rename here to keep everything consistent
    if "text" in combined.columns:
        combined.rename(columns={"text": "Text_Generated"}, inplace=True)

    combined["Language_Code"]    = lang_code
    combined["Model_Identifier"] = "human"
    combined["Label"]            = 0  #0 means human authored

    return combined[["Text_Generated", "Language_Code", "Model_Identifier", "Label"]]


#load human text for all three languages
#isiZulu and isiXhosa are used for training, Siswati is held out for zero-shot testing
print("\nloading vukuzenzele human text from huggingface...")

print("  loading isiZulu...")
zulu_human    = load_vukuzenzele("zu", "zul")
print(f"    {len(zulu_human)} records")

print("  loading isiXhosa...")
xhosa_human   = load_vukuzenzele("xh", "xho")
print(f"    {len(xhosa_human)} records")

print("  loading Siswati...")
siswati_human = load_vukuzenzele("ss", "ssw")
print(f"    {len(siswati_human)} records")

human_df = pd.concat([zulu_human, xhosa_human, siswati_human], ignore_index=True)

print(f"\ntotal human records: {len(human_df)}")
print(human_df["Language_Code"].value_counts())

#swap this to mgt_text.csv once all group members have added their generated files
#and they have been merged together
#MACHINE_FILE = os.path.join(PROCESSED_PATH, "mgt_text.csv")  #full dataset when ready
MACHINE_FILE = os.path.join(RAW_PATH, "mgt_claude.csv")  #claude only for now

print(f"\nloading machine generated text from:\n  {MACHINE_FILE}")

machine_df          = pd.read_csv(MACHINE_FILE)
machine_df["Label"] = 1  #1 means machine generated
machine_df          = machine_df[["Text_Generated", "Language_Code", "Model_Identifier", "Label"]]

print(f"\nmachine generated records: {len(machine_df)}")
print(machine_df["Language_Code"].value_counts())
print(machine_df["Model_Identifier"].value_counts())

#combine human and machine records into one dataframe
combined_df = pd.concat([human_df, machine_df], ignore_index=True)

#drop any rows where the text is missing or empty after stripping whitespace
combined_df.dropna(subset=["Text_Generated"], inplace=True)
combined_df["Text_Generated"] = combined_df["Text_Generated"].astype(str).str.strip()
combined_df = combined_df[combined_df["Text_Generated"] != ""]

print(f"\ntotal records after cleaning: {len(combined_df)}")
print(f"\nby language:")
print(combined_df["Language_Code"].value_counts())
print(f"\nby label (0=human, 1=machine):")
print(combined_df["Label"].value_counts())

#quick preview of how the data will split between training languages and the held-out language
#the actual train/test split happens inside baseline.py
train_pool = combined_df[combined_df["Language_Code"].isin(["zu", "xh"])]
siswati    = combined_df[combined_df["Language_Code"] == "ss"]

print(f"\ntraining pool (isiZulu + isiXhosa) : {len(train_pool)} records")
print(train_pool["Label"].value_counts())
print(f"\nsiswati held-out (zero-shot)        : {len(siswati)} records")
print(siswati["Label"].value_counts())

#save the final merged dataset so baseline.py can load it directly
output_path = os.path.join(PROCESSED_PATH, "merged_dataset.csv")
combined_df.to_csv(output_path, index=False)

print(f"\nsaved merged dataset to:\n  {output_path}")
print(f"shape: {combined_df.shape}")
