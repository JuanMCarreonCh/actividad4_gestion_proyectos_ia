# Actividad 4: Gestión de Proyectos de IA - Análisis Comparativo de Modelos de NLP

## 1. Información del Notebook

* **Archivo del Notebook:** [`GestionProyectosIA_Actividad4.ipynb`](file:///c:/Users/juani/Desktop/MasterIA/Curso5.%20Gestion%20de%20Proyectos/Actividad4/actividad4_gestion_proyectos_ia/notebooks/GestionProyectosIA_Actividad4.ipynb)
* **Rama de IA:** Procesamiento de Lenguaje Natural (NLP - *Natural Language Processing*).
* **Caso de Uso:** Clasificación de sentimientos en texto (reseñas binarias: **Positivo** / **Negativo**) mediante modelos preentrenados basados en la arquitectura *Transformer*.
* **Dataset:** [`stanfordnlp/imdb`](https://huggingface.co/datasets/stanfordnlp/imdb) (conjunto de prueba `test`), extrayendo una muestra reproducible de **1,000 registros balanceados** (semilla aleatoria `seed=42`).
* **Modelos Evaluados:**
  1. **DistilBERT:** `distilbert-base-uncased-finetuned-sst-2-english` (modelo ligero y destilado, diseñado para alta velocidad).
  2. **BERT-base:** `textattack/bert-base-uncased-imdb` (modelo estándar bidireccional afinado específicamente en IMDB).
  3. **RoBERTa:** `aychang/roberta-base-imdb` (variante optimizada de BERT entrenada con más datos y sin objetivo de predicción de la siguiente oración).

---

## 2. ¿Qué se hace en el Notebook?

El cuaderno implementa un *benchmark* comparativo integral orientado a la toma de decisiones en proyectos de IA:

1. **Inspección del Entorno de Cómputo:** Detección automática y asignación del dispositivo de aceleración por hardware (`cuda` para GPU NVIDIA o `cpu` como respaldo).
2. **Autenticación en Hugging Face Hub:** Conexión segura mediante token de acceso para la descarga de modelos y artefactos.
3. **Carga y Muestreo de Datos:** Descarga del dataset IMDB desde la librería `datasets`, aplicando un muestreo pseudoaleatorio balanceado para optimizar el tiempo de inferencia sin perder significancia estadística.
4. **Construcción del Pipeline de Inferencia:** Creación de pipelines unificados de Hugging Face (`sentiment-analysis`) configurados con truncamiento a 512 tokens y procesamiento por lotes (*batch size* de 16).
5. **Evaluación de Rendimiento y Latencia:** 
   - Medición rigurosa de tiempos de inferencia total y cálculo de la **latencia promedio por secuencia** (en milisegundos).
   - Estandarización de las etiquetas emitidas por cada modelo (`POSITIVE`/`NEGATIVE` o `LABEL_1`/`LABEL_0`) a valores binarios homogéneos (`1` y `0`).
   - Cálculo de métricas de clasificación estándar con `scikit-learn`: **Accuracy**, **Precision**, **Recall** y **F1-Score**.
6. **Consolidación de Resultados:** Visualización tabular estructurada mediante `tabulate` y redacción de conclusiones sobre el *trade-off* latencia vs. rendimiento.

---

## 3. ¿Qué se necesita para ejecutar?

### Requisitos de Hardware y Software
* **Python:** Versión `3.9` o superior.
* **Acelerador de Hardware:** Se recomienda disponer de una **GPU con soporte CUDA** (por ejemplo, NVIDIA T4 en Google Colab o GPU dedicada local) para una evaluación ágil. El código cuenta con soporte de respaldo para **CPU**, aunque con una latencia significativamente mayor.
* **Acceso a Hugging Face:** Una cuenta en [Hugging Face](https://huggingface.co/) y un **User Access Token** con rol de lectura (*Read*).

### Dependencias Principales
Definidas en el archivo [`requirements.txt`](file:///c:/Users/juani/Desktop/MasterIA/Curso5.%20Gestion%20de%20Proyectos/Actividad4/actividad4_gestion_proyectos_ia/requirements.txt):
* `torch`
* `transformers`
* `datasets`
* `evaluate`
* `huggingface_hub`
* `pandas`
* `scikit-learn`
* `tabulate`
* `ipywidgets` / `jupyterlab` / `notebook`

---

## 4. ¿Cómo se ejecuta?

### Opción A: Ejecución en Google Colab (Recomendado)
1. Abra Google Colab y suba el archivo `GestionProyectosIA_Actividad4.ipynb`.
2. Configure el tipo de entorno de ejecución en **GPU T4**:
   * En el menú superior: `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución` > Seleccionar `GPU T4`.
3. Configure su token de Hugging Face como secreto en Colab:
   * Haga clic en el ícono de llave (**Secrets**) en la barra lateral izquierda.
   * Agregue un nuevo secreto con nombre: `hug_face_token` y pegue su token de Hugging Face.
   * Conceda acceso al cuaderno.
4. Ejecute las celdas secuencialmente (`Shift + Enter` o `Entorno de ejecución` > `Ejecutar todas`).

### Opción B: Ejecución en Entorno Local

1. **Clonar o situarse en el directorio del proyecto:**
   ```bash
   cd "c:/Users/juani/Desktop/MasterIA/Curso5. Gestion de Proyectos/Actividad4/actividad4_gestion_proyectos_ia"
   ```

2. **Crear y activar un entorno virtual:**
   * En Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * En Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   *(Opcional: Si cuenta con GPU NVIDIA local, instale la versión de PyTorch con soporte CUDA acorde a sus controladores desde [pytorch.org](https://pytorch.org/get-started/locally/)).*

4. **Autenticación local con Hugging Face:**
   En entorno local, puede autenticarse vía CLI antes de iniciar el notebook:
   ```bash
   huggingface-cli login
   ```
   *Nota:* En la Celda 3 del notebook se usa `google.colab.userdata`. Para ejecución fuera de Colab, reemplace dicha celda por:
   ```python
   from huggingface_hub import login
   import os
   login(token=os.getenv("HF_TOKEN", "TU_TOKEN_AQUI"))
   ```

5. **Iniciar Jupyter:**
   ```bash
   jupyter lab
   # o bien:
   jupyter notebook
   ```
   Abra `notebooks/GestionProyectosIA_Actividad4.ipynb` y ejecute todas las celdas.

---

## 5. Tabla Comparativa de Resultados

Resultados obtenidos en el entorno de evaluación (muestra de 1,000 textos en GPU T4):

| Modelo | Accuracy | Precision | Recall | F1-Score | Latencia (ms/seq) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DistilBERT** | 0.8810 | 0.8917 | 0.8607 | 0.8759 | **17.43** |
| **BERT-base** | 0.9280 | **0.9426** | 0.9078 | 0.9248 | 30.50 |
| **RoBERTa** | **0.9400** | 0.9280 | **0.9508** | **0.9393** | 30.33 |

---

## 6. Conclusiones Parciales del Cuaderno

A continuación se copian las conclusiones extraídas directamente del análisis en el cuaderno:

1. **Rendimiento Computacional (Latencia):**  
   Se observa que **DistilBERT** ofrece la latencia por secuencia más baja (17.43 ms/seq), justificando su arquitectura destilada, lo cual es ideal para entornos de producción con restricciones de tiempo de respuesta.

2. **Desempeño de Clasificación (Métricas):**  
   Modelos más robustos como **BERT-base** o **RoBERTa** tienden a mostrar mejores resultados en F1-Score y Accuracy (RoBERTa alcanza 0.9400 en Accuracy y 0.9393 en F1-Score), ya que capturan mejor el contexto bidireccional y las sutilezas lingüísticas de reseñas largas.

3. **Trade-off:**  
   La elección final dependerá del caso de negocio:
   * Si el sistema de reseñas requiere procesamiento en **tiempo real masivo**, el sacrificio mínimo en Accuracy de DistilBERT compensa su velocidad (un ~43% más rápido en latencia).
   * Si se trata de un procesamiento **batch analítico** donde la precisión es prioritaria sobre el tiempo de respuesta, **RoBERTa** es superior.
