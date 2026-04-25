# Multi-Scale FCOS for Query-Conditioned Detection

Implementación de FCOS (Fully Convolutional One-Stage Object Detection) con Feature Pyramid Network para detección de objetos condicionada por sketches en documentos históricos.

## 🎯 Características

- **Multi-Scale Detection**: Detecta objetos de diferentes tamaños usando FPN con 4 niveles (P3-P6)
- **Query-Conditioned**: Usa sketches como queries para buscar objetos específicos
- **ViT Backbone**: Utiliza Vision Transformer pre-entrenado (iDoc) como feature extractor
- **Backbone Congelado**: Por defecto, el encoder ViT está congelado (solo se entrenan FPN y FCOS heads)
- **Modern Training**: Mixed precision, gradient accumulation, class balancing
- **Flexible Input**: Maneja imágenes de diferentes tamaños y proporciones

## 📁 Estructura del Proyecto