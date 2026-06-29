from datasets import load_dataset

def cargar_datos():
    datos = load_dataset("Tobi-Bueck/customer-support-tickets")
    datos = datos["train"]
    tabla = datos.to_pandas()
    return tabla

def limpiar_datos(tabla):
    tabla = tabla[tabla["language"] == "en"].copy()

    tabla["subject"] = tabla["subject"].fillna("")
    tabla["body"] = tabla["body"].fillna("")
    tabla["queue"] = tabla["queue"].fillna("unknown")

    tabla["texto"] = tabla["subject"].astype(str) + ". " + tabla["body"].astype(str)
    tabla = tabla[tabla["texto"].str.len() > 10]

    return tabla
