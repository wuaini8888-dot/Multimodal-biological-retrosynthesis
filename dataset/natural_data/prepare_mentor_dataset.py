import pandas as pd
import re
import os

# 1. Your senior's ancestral tokenization regex (ensures the model can understand)
SMILES_TOKENIZER_PATTERN = r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
tokenizer_re = re.compile(SMILES_TOKENIZER_PATTERN)


def tokenize_smiles(smiles):
    """Standard tokenization function"""
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    # Replace manually written " + " with the multi-molecule separator " . " recognized by the model
    smiles = smiles.replace(" + ", " . ")
    tokens = [token for token in tokenizer_re.findall(smiles)]
    return " ".join(tokens)


print("=" * 50)
print(" Directly extracting the ultimate natural product test set from Excel")
print("=" * 50)

# 2. Directly read the complete Excel file
excel_file = "SMILES_EC.xlsx"

try:
    # Read all sheets
    xls = pd.ExcelFile(excel_file)
except FileNotFoundError:
    print(f" File {excel_file} not found, please ensure the script and the spreadsheet are in the same directory!")
    exit()

all_src = []
all_tgt = []

# Iterate through all sheets (steroids, flavonoids, terpenes, phenolic acids, etc.)
for sheet_name in xls.sheet_names:
    print(f"Parsing: {sheet_name} ...")
    df = pd.read_excel(xls, sheet_name=sheet_name)

    # Dynamically match column names (prevents issues with invisible spaces in column names across different sheets)
    try:
        sub_col = [c for c in df.columns if '底物SMILES' in str(c)][0]
        prod_col = [c for c in df.columns if '产物SMILES' in str(c)][0]
    except IndexError:
        print(f" Warning: Columns containing '底物SMILES' or '产物SMILES' not found in {sheet_name}, skipping.")
        continue

    for idx, row in df.iterrows():
        reactants_smi = str(row[sub_col]).strip()
        products_smi = str(row[prod_col]).strip()

        # Exclude empty data or nan
        if reactants_smi == 'nan' or products_smi == 'nan' or not reactants_smi or not products_smi:
            continue

        # [Core logic]: Retrosynthesis! Model input (Src) is the product, prediction target (Tgt) is the substrate
        src_tokenized = tokenize_smiles(products_smi)
        tgt_tokenized = tokenize_smiles(reactants_smi)

        all_src.append(src_tokenized)
        all_tgt.append(tgt_tokenized)

# 3. Write to the final test set files
out_dir = "mentor_testset"
os.makedirs(out_dir, exist_ok=True)

with open(f"{out_dir}/mentor_test_src.txt", "w") as f:
    f.write("\n".join(all_src) + "\n")
with open(f"{out_dir}/mentor_test_tgt.txt", "w") as f:
    f.write("\n".join(all_tgt) + "\n")

print("=" * 50)
print(f" Processing complete! Extracted {len(all_src)} high-quality retrosynthesis reactions from {len(xls.sheet_names)} categories.")
print(f" Your second test set has been saved in the {out_dir}/ directory!")
print("=" * 50)