# iDoc-FCOS: Detección de Objetos en Documentos Históricos
### Documentación técnica del sistema de entrenamiento

---

## Índice

1. [El problema que resolvemos](#1-el-problema-que-resolvemos)
2. [Conceptos fundamentales](#2-conceptos-fundamentales)
3. [La arquitectura completa](#3-la-arquitectura-completa)
4. [Por qué FCOS y no otro método](#4-por-qué-fcos-y-no-otro-método)
5. [El pre-entrenamiento iDoc](#5-el-pre-entrenamiento-idoc)
6. [Decisiones de diseño](#6-decisiones-de-diseño)
7. [El pipeline de entrenamiento](#7-el-pipeline-de-entrenamiento)
8. [Glosario de términos clave](#8-glosario-de-términos-clave)

---

## 1. El problema que resolvemos

Tenemos un conjunto de páginas de manuscritos medievales del dataset **HORAE** y queremos que una red neuronal sea capaz de responder la siguiente pregunta:

> *"Dada una imagen de un manuscrito y un boceto (sketch) dibujado a mano de un símbolo específico, ¿dónde aparece ese símbolo en la página?"*

Este tipo de tarea se llama **detección de objetos condicionada por query**, y es considerablemente más difícil que una detección clásica porque:

- El detector no sabe de antemano qué clase buscar: se lo decimos en tiempo de inferencia mediante el sketch.
- Los objetos tienen escalas extremadamente variables: desde marcas de 5×15 píxeles hasta ilustraciones que ocupan un tercio de la página.
- El dataset es pequeño (544 imágenes, 22 clases), lo que hace que entrenar desde cero sea inviable.

La solución que implementamos tiene tres pilares: un **backbone pre-entrenado y congelado** (iDoc), un **detector anclado en píxeles** (FCOS), y un mecanismo de **condicionamiento por query** (FiLM).

---

## 2. Conceptos fundamentales

Esta sección presenta los bloques de conocimiento necesarios, de lo más simple a lo más técnico.

### 2.1 ¿Qué es una red neuronal convolucional (CNN)?

Una CNN procesa imágenes aplicando filtros que detectan patrones locales: bordes en las primeras capas, formas más complejas en las más profundas. El resultado es un **mapa de características** (*feature map*): una representación comprimida de la imagen donde cada posición resume "qué hay cerca de ese punto". Si la imagen de entrada es de 800×800 px y el filtro se aplica cada 16 píxeles, el mapa resultante es de 50×50 celdas.

### 2.2 ¿Qué es un Vision Transformer (ViT)?

Un **ViT** divide la imagen en parches (*patches*) de tamaño fijo (en nuestro caso 16×16 píxeles) y los trata como una secuencia, igual que un modelo de lenguaje trata palabras. Cada parche se convierte en un vector de 768 números (*embedding*) y pasa por 12 capas de **atención** que le permiten a cada parche "mirar" a todos los demás.

El resultado es que el ViT produce representaciones globalmente coherentes: un parche que contiene una esquina de un símbolo ya "sabe", gracias a la atención, que hay otros parches cercanos que forman ese símbolo completo.

```
Imagen 800×800
    │
    ▼  Dividir en parches 16×16
[p₁][p₂][p₃]...[p₂₅₀₀]   ← 2500 parches
    │
    ▼  Embedding (768-dim cada uno)
    │
    ▼  12 capas de atención (self-attention)
    │
    ▼  2500 vectores de 768 dimensiones
       (uno por parche, con contexto global)
```

### 2.3 ¿Qué es la detección de objetos?

Detectar un objeto significa producir una **bounding box**: un rectángulo `[x₁, y₁, x₂, y₂]` que delimita el objeto en la imagen, junto con una **clase** (¿qué tipo de objeto es?) y un **score** de confianza (¿qué tan seguro está el modelo?).

Los detectores modernos pueden ser de dos tipos:

| Tipo | Descripción | Ejemplos |
|------|-------------|---------|
| **Con anclas** | Proponen rectángulos predefinidos en cada posición y los ajustan | Faster R-CNN, RetinaNet |
| **Sin anclas** | Predicen la caja directamente desde cada punto de la imagen | FCOS, CenterNet |

Nuestro sistema usa el segundo enfoque (FCOS), cuya lógica se explica en la sección 4.

### 2.4 ¿Qué es una Feature Pyramid Network (FPN)?

Un problema central en detección es la **variación de escala**: un símbolo pequeño necesita resolución alta para ser detectado, pero un símbolo grande necesita contexto amplio. La FPN resuelve esto creando representaciones a múltiples escalas simultáneamente.

```
Backbone (ViT)
    │
    ├── Capa 2  → feature map H/4  × W/4  → P2 (objetos muy pequeños, ~5-32px)
    ├── Capa 5  → feature map H/8  × W/8  → P3 (objetos pequeños, ~32-64px)
    ├── Capa 8  → feature map H/16 × W/16 → P4 (objetos medianos, ~64-128px)
    └── Capa 11 → feature map H/32 × W/32 → P5 (objetos grandes, ~128-256px)
                                          → P6 (objetos muy grandes, >256px)
```

Todos los niveles se proyectan a 256 canales mediante capas convolucionales 1×1. Luego se fusionan de arriba hacia abajo (*top-down pathway*): las features del nivel grueso se interpolan y suman a las del nivel fino, combinando contexto global con detalle local.

---

## 3. La arquitectura completa

El sistema está compuesto por cuatro módulos que operan en secuencia:

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE DE DETECCIÓN                       │
│                                                                 │
│  Sketch query ──► Query Encoder ──────────────────────┐        │
│                   (ViT frozen)                         │        │
│                   [B, 768]                             │        │
│                                                        ▼        │
│  Página ────────► Backbone ──► FPN ──────► FCOS Head ──► Dets  │
│  manuscrito       (ViT frozen)  (entrenable) (entrenable)       │
│                   [B,768,H,W]  [B,256,H,W]                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Backbone (ViT-Base, congelado)

El backbone es el **Vision Transformer Base** pre-entrenado con iDoc sobre manuscritos históricos. Sus parámetros están completamente **congelados**: no se actualizan durante el entrenamiento del detector.

Para construir la pirámide de escala se extraen las activaciones de cuatro capas intermedias del ViT (capas 2, 5, 8 y 11 de 12) y se reescalan mediante interpolación bilineal y max-pooling:

| Capa extraída | Stride resultante | Objetos objetivo |
|:---:|:---:|:---|
| 2 | 4px (4x upsample) | `marqeur`, `croix`, `pdp` (~27px mediana) |
| 5 | 8px (2x upsample) | `S`, `T`, `petit_A`, `double_sep` |
| 8 | 16px (nativo ViT) | `losange`, `BP`, `encadrement` |
| 11 | 32px (2x maxpool) | `status`, `obj_31`, `obj_34` |

Esta estrategia se conoce como **Simple Feature Pyramid** y fue propuesta para ViT por He et al. (2022) como alternativa al FPN jerárquico de los modelos Swin.

### 3.2 FPN (Feature Pyramid Network, entrenable)

Recibe los 4 feature maps del backbone (todos con 768 canales) y los proyecta a 256 canales mediante convoluciones laterales 1×1. Luego aplica el *top-down pathway* para fusionar escalas y añade P6 mediante una convolución stride-2 sobre P5.

Todos los parámetros del FPN **se entrenan** desde cero.

### 3.3 Query Encoder (ViT, congelado)

El sketch de query pasa por el mismo ViT del backbone (compartiendo pesos). Se extrae el **token CLS** de la última capa, un vector de 768 dimensiones que resume globalmente el contenido del sketch. Este vector codifica "qué símbolo estoy buscando".

El token CLS es la representación de propósito general del ViT, equivalente al `[CLS]` de BERT en NLP: no corresponde a ningún parche específico sino a un resumen de toda la imagen.

### 3.4 FCOS Head con FiLM conditioning (entrenable)

La cabeza FCOS predice, para cada celda de cada nivel FPN, tres cosas:
- **Clasificación**: ¿hay un objeto aquí? ¿de qué clase?
- **Regresión**: distancias (l, t, r, b) desde este punto hasta los bordes del objeto
- **Centerness**: ¿qué tan centrado está este punto dentro del objeto?

El **condicionamiento FiLM** (*Feature-wise Linear Modulation*) inyecta el embedding del query en la cabeza antes de cada convolución:

```
query_emb [B, 768]
    │
    ▼  Linear → [γ, β]  (scale y shift por canal)
    │
feature_map [B, 256, H, W]
    │
    ▼  out = γ * feature_map + β
    │
  conv → ... → predicciones
```

Esto permite que el detector ajuste su "atención" según qué símbolo está buscando, sin necesitar reentrenar para cada nueva clase.

---

## 4. Por qué FCOS y no otro método

### 4.1 El problema con las anclas (*anchor-based*)

Los detectores clásicos como Faster R-CNN o RetinaNet definen una cuadrícula de **anclas**: rectángulos predefinidos de distintos tamaños y proporciones que se colocan en cada posición de la imagen. El detector entonces aprende a "ajustar" esas anclas para que encajen con los objetos reales.

El problema en nuestro caso: las anclas son hiperparámetros que hay que diseñar manualmente, y nuestro dataset tiene objetos con **aspect ratios extremos** (desde 0.2 hasta 14.76, con mediana 1.73). Diseñar anclas que cubran todo ese rango es complejo, y si las anclas no coinciden con los objetos, el entrenamiento se vuelve inestable.

### 4.2 FCOS: detección sin anclas

**FCOS** (*Fully Convolutional One-Stage Object Detector*, Tian et al., 2019) elimina las anclas. En vez de preguntar "¿este rectángulo predefinido coincide con algún objeto?", pregunta: "**desde este píxel, ¿a qué distancia están los bordes del objeto más cercano?**"

Para cada punto `(x, y)` en el mapa de características, FCOS predice:
- `l`: distancia al borde izquierdo del objeto
- `t`: distancia al borde superior
- `r`: distancia al borde derecho
- `b`: distancia al borde inferior
- `centerness`: √((min(l,r)/max(l,r)) × (min(t,b)/max(t,b)))

El **centerness** es crucial: penaliza las predicciones que vienen de puntos alejados del centro del objeto, reduciendo el número de falsos positivos. Un punto exactamente en el centro de un cuadrado perfecto tiene centerness=1; un punto en la esquina tiene centerness≈0.

### 4.3 Asignación de targets en FCOS

Cada punto de cada nivel FPN se asigna a un GT box siguiendo dos reglas:

1. El punto debe estar **dentro** de la bounding box GT.
2. La distancia máxima `max(l,t,r,b)` debe caer dentro del **rango de regresión** del nivel.

Si un punto puede pertenecer a varios GT boxes (overlapping), se asigna al de **menor área**. Esto es elegante: no requiere calcular IoU con cientos de anclas.

| Nivel | Rango √area |
|:---:|:---:|
| P2 | 0 – 32px |
| P3 | 32 – 64px |
| P4 | 64 – 128px |
| P5 | 128 – 256px |
| P6 | 256px – ∞ |

Estos rangos se calibraron manualmente según el análisis del dataset: `marqeur` tiene mediana 27px → P2; `obj_2` tiene mediana 599px → P6.

### 4.4 Ventajas concretas para este proyecto

| Criterio | Anchor-based | FCOS |
|----------|:---:|:---:|
| Hiperparámetros a definir | Muchos (tamaños, ratios, umbrales IoU) | Pocos (rangos por nivel) |
| Aspect ratios extremos | Problemático | Manejable |
| Objetos muy pequeños (<32px) | Depende del anchor más pequeño | P2 con stride 4 |
| Condicionamiento por query | Complejo de integrar | Natural en la cabeza |
| Dataset pequeño | Sensible a mal diseño de anclas | Más robusto |

---

## 5. El pre-entrenamiento iDoc

### 5.1 ¿Por qué pre-entrenar?

Con solo 544 imágenes no es posible entrenar un ViT-Base (86M parámetros) desde cero: habría *overfitting* severo. La solución es partir de un modelo que ya "entiende" documentos históricos y solo entrenar la capa de detección encima.

### 5.2 El framework iBOT

iDoc fue pre-entrenado usando **iBOT** (*image BERT pre-training with Online Tokenizer*), que combina dos objetivos de auto-supervisión:

**Objetivo DINO** (sobre el token CLS): el modelo aprende representaciones globales comparando la salida de una red *student* con la de una red *teacher* (actualizada por promedio exponencial). El student ve vistas con más augmentación; el teacher ve vistas más "limpias".

**Objetivo MIM** (*Masked Image Modeling*, sobre los patch tokens): se enmascaran parches de la imagen y el model aprende a predecir sus representaciones, análogo al enmascaramiento de palabras en BERT.

```
Imagen histórica
    │
    ├── 2 vistas globales (224px) ──► Teacher ──► CLS target
    │                             └► Student
    └── 10 vistas locales (96px)  ──► Student solo
                                       │
                                  Máscaras BEiT ──► patch targets

Loss = λ₁ × L_DINO(CLS) + λ₂ × L_MIM(patches)
```

### 5.3 LoRA: fine-tuning eficiente

El checkpoint de iDoc usa **LoRA** (*Low-Rank Adaptation*): en vez de guardar la matriz completa de atención `W ∈ ℝ^{768×768}`, guarda el peso base más dos matrices pequeñas:

```
W_merged = W_base + W_lora_B @ W_lora_A
           [768×768]   [768×64] @ [64×768]
```

El rango `r=64` implica que los parámetros de atención se reducen de 768² = 589,824 a 2×64×768 = 98,304 por proyección — una compresión de ~6×. Nuestro código reconstruye `W_merged` al cargar el checkpoint para que sea compatible con el ViT estándar.

### 5.4 ¿Qué ganamos al congelar el backbone?

Al congelar el ViT:
- Los 85.8M parámetros del backbone no se optimizan → menor memoria GPU, mayor estabilidad.
- El conocimiento específico de manuscritos históricos se preserva.
- Solo se entrenan 12.5M parámetros (FPN + FCOS head + LayerNorms), lo que es manejable con 433 imágenes.

---

## 6. Decisiones de diseño

### 6.1 Resolución de entrada

Las páginas HORAE tienen entre 460 y 3852px de ancho. El símbolo más frecuente (`marqeur`) tiene mediana de solo 27px. Si redimensionamos ingenuamente a 224px, ese símbolo queda en ~3px, indetectable.

La solución: **resize manteniendo aspect ratio** con lado corto en 800px y tope en 1333px. Un `marqeur` de 27px en una página de 600px queda en 36px al redimensionar a 800px — pequeño pero detectable en P2 (stride 4).

### 6.2 Función de pérdida

El sistema minimiza tres pérdidas simultáneamente:

```
L_total = λ_cls × L_focal  +  λ_bbox × L_GIoU  +  λ_ctr × L_centerness
```

**Focal Loss** para clasificación: da más peso a los ejemplos difíciles. El factor `(1-p_t)^γ` suprime la contribución de los negativos fáciles (el 99% del fondo). Fundamental cuando hay miles de celdas negativas por cada positiva.

**GIoU Loss** para regresión: a diferencia de la pérdida L1 sobre coordenadas, la GIoU (*Generalized IoU*) mide directamente cuánto se superponen las cajas predicha y real, incluyendo penalización por desplazamiento cuando no hay solapamiento. Funciona bien con aspect ratios extremos.

**BCE** para centerness: pérdida binaria entre la predicción y el target `√((min(l,r)/max(l,r)) × (min(t,b)/max(t,b)))`.

### 6.3 Augmentación

Con solo 433 imágenes de train, la augmentación es crítica:

- **Multi-scale resize**: el lado corto se muestrea entre [640, 720, 800, 900, 1024]px en cada iteración.
- **Flip horizontal**: con probabilidad 0.5.
- **Color jitter**: brillo ±0.3, contraste ±0.3 — moderado para no destruir la textura del pergamino.
- **Copy-paste**: objetos de las clases más pequeñas (`marqeur`, `croix`) se recortan y pegan en otras imágenes del train set. Esto multiplica artificialmente las instancias de las clases raras.

---

## 7. El pipeline de entrenamiento

### 7.1 Split del dataset

El dataset se divide por `page_id` (no por sample) para evitar *data leakage*: si la misma página apareciera en train y val, el modelo podría memorizar esa página específica en vez de generalizar.

| Split | Páginas | Muestras |
|:---:|:---:|:---:|
| Train | 336 | 433 |
| Val | 42 | 55 |
| Test | 42 | 56 |

### 7.2 Optimizador y schedule

```
AdamW  (lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))
Schedule: coseno con warmup de 5 épocas
    Época 0-4:  lr sube linealmente de 2e-5 a 1e-4
    Época 5-50: lr baja siguiendo cos(π×t) hasta 1e-6
```

El warmup evita que los gradientes grandes al inicio del entrenamiento dañen las features del FPN recién inicializado.

### 7.3 Métricas de evaluación

La métrica principal es **mAP@0.5**: *mean Average Precision* con umbral IoU=0.5. Para cada clase se calcula la curva Precision-Recall y su área (AP), luego se promedia sobre las 22 clases. Un mAP de 0.5 significa que en promedio el modelo detecta correctamente la mitad de los objetos con buen solapamiento.

Se monitorea también el AP por clase para identificar si clases específicas (como las más raras o las más pequeñas) mejoran o se estancan.

---

## 8. Glosario de términos clave

| Término | Definición |
|---------|-----------|
| **Backbone** | Red neuronal que extrae features de la imagen. En nuestro caso, ViT-Base congelado. |
| **Bounding box** | Rectángulo `[x₁, y₁, x₂, y₂]` que delimita un objeto en la imagen. |
| **CLS token** | Vector especial del ViT que resume el contenido global de la imagen. |
| **Centerness** | Escalar en [0,1] que indica cuán centrado está un punto dentro de su objeto. |
| **Feature map** | Representación intermedia de la imagen: tensor `[B, C, H, W]` donde C son canales (features) y H,W la resolución espacial. |
| **FiLM** | *Feature-wise Linear Modulation*. Condiciona un feature map aplicando escala y offset aprendidos desde un embedding externo (el query). |
| **Focal Loss** | Variante de cross-entropy que reduce el peso de ejemplos fáciles para enfocarse en los difíciles. |
| **FPN** | *Feature Pyramid Network*. Fusiona features de múltiples escalas del backbone. |
| **FCOS** | *Fully Convolutional One-Stage Object Detector*. Detecta objetos prediciendo distancias a los bordes desde cada píxel, sin anclas. |
| **GIoU** | *Generalized Intersection over Union*. Métrica y función de pérdida para comparar bounding boxes que funciona incluso sin solapamiento. |
| **iBOT** | Framework de pre-entrenamiento auto-supervisado que combina DINO (CLS) y MIM (patches). |
| **LoRA** | *Low-Rank Adaptation*. Representa matrices de pesos grandes como producto de dos matrices pequeñas de rango bajo. |
| **mAP** | *mean Average Precision*. Métrica estándar en detección de objetos: promedio de AP sobre todas las clases. |
| **MIM** | *Masked Image Modeling*. Objetivo de pre-entrenamiento donde el modelo predice parches enmascarados. |
| **NMS** | *Non-Maximum Suppression*. Post-proceso que elimina detecciones duplicadas conservando solo la de mayor score por región. |
| **Patch** | Región cuadrada de la imagen (16×16px en ViT-Base) que se procesa como una unidad. |
| **Self-attention** | Mecanismo que permite a cada token del ViT "consultar" a todos los demás para construir su representación. |
| **Simple FPN** | Variante del FPN para ViT que obtiene multi-escala extrayendo features de capas intermedias y reescalando. |
| **Stride** | Cuántos píxeles de la imagen original corresponden a un píxel del feature map. Stride 4 → resolución alta; stride 32 → baja. |
| **Student / Teacher** | En iBOT, dos redes con la misma arquitectura. El student se entrena con gradientes; el teacher se actualiza por EMA del student. |
| **Token** | Vector que representa un parche de imagen en el ViT. Hay N_patches + 1 tokens (el extra es el CLS token). |
| **Top-down pathway** | En el FPN, el flujo de información desde el nivel más grueso (stride 32) hacia el más fino (stride 4), fusionando contexto con detalle. |
| **ViT** | *Vision Transformer*. Arquitectura que procesa imágenes divididas en parches mediante mecanismos de atención. |
| **Warmup** | Período inicial donde el learning rate sube gradualmente desde un valor bajo hasta el valor objetivo. |
| **Weight decay** | Regularización L2: penaliza pesos grandes para evitar overfitting. |

---

*Documentación generada para el proyecto iDoc-FCOS. Dataset: HORAE. Backbone: ViT-Base pre-entrenado con iBOT + LoRA. Detector: FCOS con FiLM conditioning.*
