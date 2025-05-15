# Hacia un Diagnóstico Computacional de la Depresión  
_Un enfoque basado en Deep Learning híbrido para el análisis de resonancias magnéticas_

Este proyecto forma parte de un Trabajo Final de Máster del Máster Universitario en Ciencia de Datos de la UOC. Propone el desarrollo de un modelo híbrido CNN-RNN capaz de clasificar secuencias fMRI de sujetos con y sin diagnóstico de depresión. Se incorporan técnicas de deep learning avanzadas como transfer learning, data augmentation y métodos de inteligencia artificial explicable (XAI).


## Resumen del proyecto

Este repositorio acompaña al TFM titulado "**Hacia un Diagnóstico Computacional de la Depresión: Un Enfoque Basado en Deep Learning Híbrido para el Análisis de Resonancias Magnéticas**", cuyo contenido completo puede consultarse en el documento entregado (ver `Hacia un Diagnóstico Computacional de la Depresión: Un enfoque basado en Deep Learning híbrido para el análisis de resonancias magnéticas.pdf`).


## Dataset

El conjunto de datos utilizado en este trabajo proviene del estudio publicado por **Bezmaternykh et al. (2021)**. Se trata de imágenes de resonancia magnética funcional (fMRI) en reposo disponibles en acceso abierto a través de [OpenNeuro]([https://openneuro.org/](https://openneuro.org/datasets/ds002748/versions/1.0.5)). Por razones de derechos de distribución, este repositorio no incluye el dataset original. Puedes obtenerlo siguiendo las instrucciones del repositorio original del estudio.


## Estructura y ejecución

- El proyecto se ha desarrollado íntegramente mediante **Google Colab** y almacenado en **Google Drive**.
- Está organizado en **notebooks secuenciales**, que deben ejecutarse en el orden indicado para reproducir los resultados y los archivos intermedios (preprocesamiento, entrenamiento, validación, XAI...).
- Debido a su gran tamaño, no se han incluido los archivos de splits (`X_train.npy`, `X_test.npy`, etc.) ni los modelos entrenados (`.h5`). Cada notebook incluye instrucciones para volver a generarlos si se desea.
- Es necesario adaptar las rutas y dependencias.


## Dependencias

Las librerías necesarias están listadas en `requirements.txt`, el cual contiene unificado todos los módulos usados a lo largo de los diferentes notebooks.


## Funciones comunes

El archivo `pipeline_util.py` incluye una colección de funciones auxiliares para tareas comunes de:
- Preprocesamiento de datos fMRI.
- Transformación de formato.
- Selección de slices y secuencias.
- Normalización, reconstrucción y otros pasos clave.
- Visualización de resultados de los modelos.

Estas funciones son utilizadas de forma modular por los diferentes pipelines definidos en los notebooks.


## Archivos importantes

| Archivo / carpeta      | Descripción |
|------------------------|-------------|
| `archivos.ipynb`       |Los distintos cuadernos ejecutables del proyecto |
| `pipeline_util.py`     | Funciones comunes para preprocesamiento de datos |
| `requirements.txt`     | Lista completa de dependencias |
| `Hacia un Diagnóstico Computacional de la Depresión: Un enfoque basado en Deep Learning híbrido para el análisis de resonancias magnéticas.pdf`              | Documento final del proyecto |

## 📋 Créditos

- **Autor:** Alejandro Francisco Chivite Bermúdez  
- **Tutor:** Raúl Parada Medina  
- **Programa:** Máster Universitario en Ciencia de Datos (UOC)  
- **Fecha:** Mayo de 2025  
- **Licencia:** Creative Commons BY-NC-ND 3.0 España
