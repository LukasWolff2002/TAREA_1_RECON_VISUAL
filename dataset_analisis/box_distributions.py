#!/usr/bin/env python3
"""
BBox Analysis Pro - Versión Final Paper
Genera:
1. fig1_violin_areas.pdf: Distribución de áreas con escala log y sin cortes.
2. fig2_heatmap_dimensions.pdf: Heatmap suavizado (Gaussiano) con escala logarítmica.
3. analysis_report.txt: Estadísticas descriptivas detalladas.

Uso:
    python3 box_distributions.py dataset.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from matplotlib.colors import LogNorm
import os
import sys

# ============================================================================
# 1. CONFIGURACIÓN DE ESTILO CIENTÍFICO
# ============================================================================
sns.set_style("whitegrid", {'axes.grid': True, 'grid.color': '.95'})
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
})

OUTPUT_DIR = 'output_paper_final'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================
if len(sys.argv) < 2:
    print("Error: Proporciona la ruta al archivo JSON.")
    sys.exit(1)

dataset_path = sys.argv[1]
try:
    with open(dataset_path, 'r') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    sys.exit(1)

print(f"Procesando dataset: {dataset_path}")
bbox_list = []
class_counts = defaultdict(int)

for sample in data['samples']:
    c_name = sample['class_name']
    for box in sample['boxes']:
        x1, y1, x2, y2 = box['bbox_xyxy']
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: continue
        
        area = w * h
        bbox_list.append({
            'Clase': c_name,
            'Width': w,
            'Height': h,
            'Area': area,
            'AR': w / h
        })
        class_counts[c_name] += 1

df = pd.DataFrame(bbox_list)
total_bboxes = len(df)

# Seleccionar Top 10 clases para el Violin Plot
top_classes = df['Clase'].value_counts().nlargest(10).index.tolist()
df_top = df[df['Clase'].isin(top_classes)].copy()
# Actualizar etiquetas para incluir N
df_top['Clase_Label'] = df_top['Clase'].apply(lambda x: f"{x}\n(n={class_counts[x]})")

print(f"BBoxes procesadas: {total_bboxes} | Clases totales: {len(class_counts)}")

# ============================================================================
# 3. FIGURA 1: VIOLIN PLOT DE ÁREAS (REFINADO)
# ============================================================================
print("Generando Figura 1: Violin Plot...")
fig1, ax1 = plt.subplots(figsize=(12, 4))

# Paleta y Violines (cut=0 evita que el gráfico se "invente" colas fuera de los datos)
palette = sns.color_palette("husl", len(top_classes))
sns.violinplot(
    data=df_top, x='Clase_Label', y='Area', ax=ax1,
    hue='Clase_Label', palette=palette, inner=None, 
    linewidth=1.2, edgecolor='black', saturation=0.7, 
    legend=False, cut=0
)

# Boxplot minimalista superpuesto para mostrar cuartiles
sns.boxplot(
    data=df_top, x='Clase_Label', y='Area', ax=ax1,
    width=0.1, color='black', linewidth=1.0, showfliers=False,
    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 2, 'alpha': 0.6},
    whiskerprops={'zorder': 2}, showcaps=False
)

# Punto de mediana blanco brillante
for i, label in enumerate(df_top['Clase_Label'].unique()):
    median_val = df_top[df_top['Clase_Label'] == label]['Area'].median()
    ax1.scatter(i, median_val, marker='o', color='white', s=35, 
                zorder=3, edgecolor='black', linewidth=0.8)

# Configuración de escalas y límites
ax1.set_yscale('log')
y_min, y_max = df['Area'].min(), df['Area'].max()
ax1.set_ylim(y_min * 0.5, y_max * 2)

# Líneas de referencia COCO
#ax1.axhline(32**2, color='#e74c3c', linestyle='--', alpha=0.6, zorder=1)
#ax1.axhline(96**2, color='#e67e22', linestyle='--', alpha=0.6, zorder=1)
#ax1.text(len(top_classes)-0.5, 32**2*1.2, 'Small (32²)', color='#c0392b', fontsize=9, ha='right')
#ax1.text(len(top_classes)-0.5, 96**2*1.2, 'Medium (96²)', color='#d35400', fontsize=9, ha='right')

#ax1.set_title('Distribución de Áreas por Clase (Top 10)', fontweight='bold', pad=15)
ax1.set_ylabel('Área (px²)', fontweight='bold')
ax1.set_xlabel('')
plt.xticks(rotation=15, ha='right')

fig1.savefig(os.path.join(OUTPUT_DIR, 'fig1_bbox_area_dist.pdf'))
fig1.savefig(os.path.join(OUTPUT_DIR, 'fig1_bbox_area_dist.png'))

# ============================================================================
# 4. FIGURA 2: HEATMAP 2D (SUAVIZADO E INTERPOLADO)
# ============================================================================
print("Generando Figura 2: Heatmap Suavizado...")
fig2, ax2 = plt.subplots(figsize=(12, 4))

# Filtrar outliers extremos para la visualización (Percentil 99.5)
w_limit = np.percentile(df['Width'], 99.5)
h_limit = np.percentile(df['Height'], 99.5)

# Crear histograma 2D de alta resolución
bins = 150
heatmap, xedges, yedges = np.histogram2d(
    df['Width'], df['Height'], bins=bins, 
    range=[[0, w_limit], [0, h_limit]]
)

# Dibujar Heatmap con interpolación Gaussiana para suavizar colores
im = ax2.imshow(
    heatmap.T, origin='lower',
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    aspect='auto', cmap='magma',
    norm=LogNorm(vmin=1, vmax=heatmap.max() if heatmap.max() > 1 else 10),
    interpolation='gaussian'
)

# Superponer contornos de densidad (KDE)
sns.kdeplot(
    data=df, x='Width', y='Height', ax=ax2, 
    levels=6, color='white', linewidths=0.6, alpha=0.3,
    clip=(0, w_limit)
)

# Guías de Aspect Ratio
max_dim = max(w_limit, h_limit)
ax2.plot([0, max_dim], [0, max_dim], color='white', linestyle='-', alpha=0.4, label='AR 1:1')
ax2.plot([0, max_dim], [0, max_dim/2], color='white', linestyle='--', alpha=0.3, label='AR 2:1')
ax2.plot([0, max_dim/2], [0, max_dim], color='white', linestyle=':', alpha=0.3, label='AR 1:2')

ax2.set_xlim(0, w_limit)
ax2.set_ylim(0, h_limit)
ax2.set_xlabel('Ancho (px)', fontweight='bold')
ax2.set_ylabel('Alto (px)', fontweight='bold')
#ax2.set_title('Relación Ancho vs Alto (Densidad Log)', fontweight='bold', pad=15)

# Colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Número de Objetos', rotation=270, labelpad=15)

#ax2.legend(loc='upper right', frameon=True, facecolor='black', framealpha=0.6, labelcolor='white')

fig2.savefig(os.path.join(OUTPUT_DIR, 'fig2_bbox_dimensions_heatmap.pdf'))
fig2.savefig(os.path.join(OUTPUT_DIR, 'fig2_bbox_dimensions_heatmap.png'))

# ============================================================================
# 5. GENERAR REPORTE DE TEXTO
# ============================================================================
report_path = os.path.join(OUTPUT_DIR, 'analysis_report.txt')
with open(report_path, 'w') as f:
    f.write("REPORTE TÉCNICO DE BOUNDING BOXES\n")
    f.write("="*40 + "\n\n")
    f.write(f"Total de objetos: {total_bboxes}\n")
    f.write(f"Clases únicas: {len(class_counts)}\n\n")
    
    f.write("ESTADÍSTICAS DE ÁREA (px²):\n")
    f.write(df['Area'].describe().to_string() + "\n\n")
    
    f.write("CATEGORIZACIÓN COCO:\n")
    f.write(f"Pequeños (<32²): {sum(df['Area'] < 32**2)} ({sum(df['Area'] < 32**2)/total_bboxes*100:.2f}%)\n")
    f.write(f"Medianos: {sum((df['Area'] >= 32**2) & (df['Area'] < 96**2))} ({sum((df['Area'] >= 32**2) & (df['Area'] < 96**2))/total_bboxes*100:.2f}%)\n")
    f.write(f"Grandes (>=96²): {sum(df['Area'] >= 96**2)} ({sum(df['Area'] >= 96**2)/total_bboxes*100:.2f}%)\n\n")
    
    f.write("ASPECT RATIO (W/H):\n")
    f.write(f"Mediana: {df['AR'].median():.2f}\n")
    f.write(f"Horizontal (AR > 1.2): {sum(df['AR'] > 1.2)/total_bboxes*100:.1f}%\n")
    f.write(f"Vertical (AR < 0.8): {sum(df['AR'] < 0.8)/total_bboxes*100:.1f}%\n")

print(f"\n✓ Proceso finalizado exitosamente.")
print(f"✓ Archivos generados en: {OUTPUT_DIR}/")
plt.close('all')