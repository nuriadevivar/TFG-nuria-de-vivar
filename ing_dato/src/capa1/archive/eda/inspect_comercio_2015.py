import pandas as pd
from config import RAW_COMERCIO_DIR

file_path = RAW_COMERCIO_DIR / "comercio_electronico2015.xlsx"

print("Ruta del archivo:")
print(file_path)
print("")

xls = pd.ExcelFile(file_path)

print("Hojas disponibles:")
print(xls.sheet_names)
print("")

for sheet in xls.sheet_names:
    print(f"--- Hoja: {sheet} ---")
    df = pd.read_excel(file_path, sheet_name=sheet, header=None)
    print("Dimensiones:", df.shape)
    print(df.head(15))
    print("")