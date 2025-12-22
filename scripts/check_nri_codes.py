import pandas as pd

# Read the Excel file
file_path = 'nri17_csv_layout_050521.xlsx'

# Try to read all sheets
xl_file = pd.ExcelFile(file_path)
print(f"Available sheets: {xl_file.sheet_names}")

# Look for BROAD code definitions
for sheet_name in xl_file.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Search for BROAD in column names or values
    if any('BROAD' in str(col).upper() for col in df.columns):
        print(f"\n=== Sheet: {sheet_name} ===")
        print(f"Columns with BROAD: {[col for col in df.columns if 'BROAD' in str(col).upper()]}")
        print(df.head())

    # Search for BROAD in the data
    for col in df.columns:
        if df[col].astype(str).str.contains('BROAD', case=False, na=False).any():
            print(f"\n=== Sheet: {sheet_name}, Column: {col} ===")
            broad_rows = df[df[col].astype(str).str.contains('BROAD', case=False, na=False)]
            print(broad_rows[[col] + [c for c in df.columns if c != col][:3]])

# Also look for forest, land use, or code 7
for sheet_name in xl_file.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Search for rows containing both '7' and 'forest'
    for col in df.columns:
        try:
            mask = (df[col].astype(str).str.contains('forest', case=False, na=False) |
                   df[col].astype(str) == '7')
            if mask.any():
                relevant_rows = df[mask]
                if len(relevant_rows) > 0 and len(relevant_rows) < 20:
                    print(f"\n=== Sheet: {sheet_name} - Forest/Code 7 references ===")
                    print(relevant_rows)
        except:
            pass