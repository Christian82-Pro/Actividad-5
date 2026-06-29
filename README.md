# Actividad-5
Actividad 5. Design thinking, definiendo e ideando
# Descripción
Este proyecto usa machine learning para clasificar tickets de soporte según el área que debe atenderlos.
Se trabajó con el dataset `Tobi-Bueck/customer-support-tickets` de Hugging Face. La variable que se predice es `queue`, porque representa la categoría o área responsable del ticket.
# Modelos usados
Se compararon dos modelos:
- Regresión Logística
- Random Forest
# Dataset
Fuente del dataset:
https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets
El dataset se descarga desde Python usando:
```python
from datasets import load_dataset
datos = load_dataset("Tobi-Bueck/customer-support-tickets")
