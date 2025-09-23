import pandas as pd

# Read the Excel file
file_path = 'nri17_csv_layout_050521.xlsx'

# Read the main sheet
df = pd.read_excel(file_path, sheet_name='2017 NRI layout')

# Find the row with broad code description
broad_desc_row = df[df.iloc[:, 4].astype(str).str.contains('Broad Cover/Use for 1982', na=False)]
if not broad_desc_row.empty:
    print("BROAD Code Definition found:")
    print("=" * 80)
    desc = broad_desc_row.iloc[0, 4]
    print(desc)
    print("=" * 80)

# Also check the "IV & V Land Cover Uses" sheet which likely has the code definitions
land_use_df = pd.read_excel(file_path, sheet_name='IV & V Land Cover Uses')
print("\n=== Land Cover/Use Codes ===")
print(land_use_df.to_string())