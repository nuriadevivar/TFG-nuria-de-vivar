import pandas as pd
from config import DATA_PROCESSED

inventory = pd.DataFrame([
    {
        "dataset": "processed/contexto/contexto_digitalizacion_clean.csv",
        "fuente": "DataReportal + Eurostat",
        "periodo": "2020-2025",
        "frecuencia": "anual",
        "unidad_analisis": "anio",
        "descripcion": "Uso de RRSS y compra online en España",
        "rol_capa1": "contexto digital"
    },
    {
        "dataset": "processed/eurostat/eurostat_moda_mensual_clean.csv",
        "fuente": "Eurostat",
        "periodo": "2015-2023",
        "frecuencia": "mensual",
        "unidad_analisis": "fecha",
        "descripcion": "Volumen de ventas retail de moda en España, índice base 2015=100",
        "rol_capa1": "contexto sectorial moda"
    },
    {
        "dataset": "processed/eurostat/eurostat_retail_total_mensual_clean.csv",
        "fuente": "Eurostat",
        "periodo": "2015-2023",
        "frecuencia": "mensual",
        "unidad_analisis": "fecha",
        "descripcion": "Volumen de ventas retail total en España, índice base 2015=100",
        "rol_capa1": "contexto retail general"
    },
    {
        "dataset": "processed/eurostat/eurostat_online_empresas_clean.csv",
        "fuente": "Eurostat",
        "periodo": "2015-2023",
        "frecuencia": "anual",
        "unidad_analisis": "anio",
        "descripcion": "Porcentaje de la facturación empresarial procedente de ventas online",
        "rol_capa1": "adopcion online empresarial"
    },
    {
        "dataset": "processed/comercio_electronico/comercio_electronico_core_std.csv",
        "fuente": "INE / Encuesta TIC y Comercio Electrónico",
        "periodo": "2015-2023",
        "frecuencia": "anual",
        "unidad_analisis": "anio-indicador-tamano_empresa",
        "descripcion": "Indicadores clave de comercio electrónico por tamaño de empresa",
        "rol_capa1": "contexto ecommerce empresarial"
    },
])

output_path = DATA_PROCESSED / "capa1_inventory.csv"
inventory.to_csv(output_path, index=False)

print("Inventario guardado en:")
print(output_path)
print("")
print(inventory)