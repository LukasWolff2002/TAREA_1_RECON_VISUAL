import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

# Configuración de iDoc
sys.path.append(os.path.abspath('./iDoc'))
import models

## 1. FUNCIONES DE PROCESAMIENTO

def get_dynamic_patches(image_path, patch_size=16):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    new_w, new_h = (w // patch_size) * patch_size, (h // patch_size) * patch_size
    img = img.crop((0, 0, new_w, new_h))
    
    img_tensor = transforms.ToTensor()(img)
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    grid_h, grid_w = patches.shape[1], patches.shape[2]
    
    patches = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    return patches, (grid_h, grid_w), img

def load_model(ckpt_path, device):
    model = models.__dict__['vit_base'](patch_size=16, num_classes=0)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace("module.", "").replace("encoder.model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()

## 2. CAPA DE DETECCIÓN (ESTILO FCOS)

def get_fcos_bounding_box(heatmap, threshold=0.6):
    """
    Simula la salida de FCOS: identifica el centro de masa de la activación 
    y genera una caja delimitadora (bbox).
    """
    # 1. Binarizar el heatmap basado en un umbral
    mask = heatmap > (heatmap.max() * threshold)
    coords = np.argwhere(mask) # Obtener coordenadas (y, x) donde hay activación
    
    if len(coords) == 0:
        return None

    # 2. Definir los límites (min/max) para la caja
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return [x_min, y_min, x_max, y_max]

## 3. EJECUCIÓN PRINCIPAL

def main():
    # --- CONFIGURACIÓN ---
    PATCH_DESEADO = 128     
    BATCH_SIZE = 128        
    INVERTIR_MAPA = True   
    UMBRAL_DETECCION = 0.7 # Qué tan estricto es FCOS para la caja
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT = "idoc_pretrained.pth"
    IMG_PATH = "DocExplore_images/page3.jpg"
    SK_PATH = "Sketches/sketch_1_DocExplore_bateau_1349_4x_1349.jpeg"

    model = load_model(CKPT, DEVICE)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 1. Inferencia
    img_patches, grid_dim, img_cropped = get_dynamic_patches(IMG_PATH, patch_size=16)
    img_patches_norm = torch.stack([norm(p) for p in img_patches]).to(DEVICE)
    sk_img = Image.open(SK_PATH).convert("RGB").resize((224, 224))
    sk_tensor_norm = norm(transforms.ToTensor()(sk_img)).unsqueeze(0).to(DEVICE)

    img_features_list = []
    with torch.no_grad():
        for i in range(0, len(img_patches_norm), BATCH_SIZE):
            batch = img_patches_norm[i : i + BATCH_SIZE]
            feats = model(batch)
            feats = feats[0] if isinstance(feats, tuple) else feats
            img_features_list.append(feats.cpu())
        
        img_features = torch.cat(img_features_list, dim=0).to(DEVICE)
        sk_features = model(sk_tensor_norm)
        sk_features = sk_features[0] if isinstance(sk_features, tuple) else sk_features

    # 2. Similitud
    img_features = F.normalize(img_features, p=2, dim=1)
    sk_features = F.normalize(sk_features, p=2, dim=1)
    sim_scores = torch.mm(sk_features, img_features.t()).cpu()
    
    heatmap = sim_scores.reshape(1, 1, grid_dim[0], grid_dim[1])
    
    # Redimensionar al tamaño de patch visual deseado
    if PATCH_DESEADO != 16:
        scale = PATCH_DESEADO // 16
        heatmap = F.avg_pool2d(heatmap, kernel_size=scale, stride=scale)

    heatmap = heatmap.squeeze().numpy()

    # 3. Corrección e Inversión
    if INVERTIR_MAPA:
        heatmap = -heatmap 
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # 4. CAPA DE DETECCIÓN (FCOS)
    # Obtenemos la caja en coordenadas de "parches"
    bbox_patches = get_fcos_bounding_box(heatmap, threshold=UMBRAL_DETECCION)
    
    # --- VISUALIZACIÓN ---
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    
    # Imagen con Bounding Box
    ax[0].imshow(img_cropped)
    if bbox_patches:
        # Escalar coordenadas de parches a píxeles
        x1, y1, x2, y2 = [c * PATCH_DESEADO for c in bbox_patches]
        # Añadir margen de un parche para que la caja no sea tan ajustada
        rect = plt.Rectangle((x1, y1), x2-x1 + PATCH_DESEADO, y2-y1 + PATCH_DESEADO, 
                             fill=False, color='red', linewidth=3, label='Detección FCOS')
        ax[0].add_patch(rect)
        ax[0].legend()
    ax[0].set_title("Detección de Objeto (BBox)")
    ax[0].axis('off')

    # Heatmap
    ax[1].imshow(img_cropped)
    ax[1].imshow(heatmap, cmap='jet', alpha=0.6, 
               extent=(0, img_cropped.size[0], img_cropped.size[1], 0), 
               interpolation='nearest')
    ax[1].set_title(f"Heatmap (Patch: {PATCH_DESEADO})")
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig("deteccion_fcos.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()