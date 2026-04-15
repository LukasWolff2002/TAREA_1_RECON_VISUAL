## 1. IMPORTACIÓN Y CONFIGURACIÓN
import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Añadir la ruta de los modelos (Asegúrate de que la carpeta contenedora esté en el PATH)
sys.path.append(os.path.abspath('./iDoc')) 
from models.vision_transformer_lora import vit_lora # Tu nuevo import

## 2. FUNCIONES DE PROCESAMIENTO

def get_dynamic_patches(image_path, patch_size=16):
    """
    Carga la imagen original y la divide en patches sin resize.
    Ajusta las dimensiones para que sean divisibles por patch_size.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Ajustar dimensiones para que sean múltiples del patch_size (16)
    new_w = (w // patch_size) * patch_size
    new_h = (h // patch_size) * patch_size
    img = img.crop((0, 0, new_w, new_h)) # Recorte mínimo para ajustar rejilla
    
    img_tensor = transforms.ToTensor()(img) # [3, H, W]
    
    # Descomponer en patches: [C, H, W] -> [Num_Patches, C, P, P]
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    grid_h, grid_w = patches.shape[1], patches.shape[2]
    
    patches = patches.contiguous().view(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
    return patches, (grid_h, grid_w), img

def load_model_lora(ckpt_path, device):
    """Carga el modelo ViT con LoRA basado en tu configuración"""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Nota: Si obtienes un error de "Missing key" o "Unexpected key", 
    # descomenta la siguiente línea para limpiar los prefijos (muy común en PyTorch):
    # state_dict = {k.replace("module.", "").replace("encoder.model.", ""): v for k, v in state_dict.items()}

    model = vit_lora().to(device)
    # strict=False es recomendable al usar LoRA por si el state_dict tiene pesos extra
    model.load_state_dict(state_dict, strict=False) 
    return model.eval()

## 3. EJECUCIÓN PRINCIPAL

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rutas actualizadas
    CKPT = "idoc_pretrained.pth" 
    IMG_PATH = "DocExplore_images/page1.jpg"
    SK_PATH = "Sketches/sketch_1_DocExplore_bateau_1349_4x_1349.jpeg"

    print("Cargando modelo ViT-LoRA...")
    model = load_model_lora(CKPT, DEVICE)
    print("Modelo cargado y en modo evaluación.")
    
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # --- PROCESAR IMAGEN (Por Patches - Local) ---
    img_patches, grid_dim, img_cropped = get_dynamic_patches(IMG_PATH)
    img_patches_norm = torch.stack([norm(p) for p in img_patches]).to(DEVICE)

    # --- PROCESAR SKETCH (Global) ---
    # El sketch se redimensiona a 224x224 para obtener un vector global estándar
    sk_img = Image.open(SK_PATH).convert("RGB").resize((224, 224))
    sk_tensor_norm = norm(transforms.ToTensor()(sk_img)).unsqueeze(0).to(DEVICE)

    # --- INFERENCIA ---
    print("Realizando inferencia...")
    with torch.no_grad():
        # Embeddings de los N patches de la imagen
        img_features = model(img_patches_norm)
        img_features = img_features[0] if isinstance(img_features, tuple) else img_features
        
        # Embedding global del sketch
        sk_features = model(sk_tensor_norm)
        sk_features = sk_features[0] if isinstance(sk_features, tuple) else sk_features

    # --- CÁLCULO DE SIMILITUD ---
    img_features = F.normalize(img_features, p=2, dim=1)
    sk_features = F.normalize(sk_features, p=2, dim=1)
    
    # Similitud Coseno: resultado de forma [1, Num_Patches]
    sim_scores = torch.mm(sk_features, img_features.t()).cpu().numpy()
    # Redimensionar a la rejilla original de la imagen (H_patches, W_patches)
    heatmap = sim_scores.reshape(grid_dim[0], grid_dim[1])

    # --- VISUALIZACIÓN ---
    plt.figure(figsize=(15, 7))
    
    # Mostrar Imagen Original
    plt.subplot(1, 2, 1)
    plt.imshow(img_cropped)
    plt.title(f"Documento Original ({img_cropped.size[0]}x{img_cropped.size[1]})")
    plt.axis('off')

    # Mostrar Heatmap superpuesto
    plt.subplot(1, 2, 2)
    plt.imshow(img_cropped)
    plt.imshow(heatmap, cmap='jet', alpha=0.6, extent=(0, img_cropped.size[0], img_cropped.size[1], 0), interpolation='bilinear')
    plt.title("Localización del Sketch (Heatmap)")
    plt.colorbar(label="Similitud Coseno")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("resultado_lora_sin_resize.png", dpi=300)
    print(f"Proceso completado con éxito. Revisa 'resultado_lora_sin_resize.png'")

if __name__ == "__main__":
    main()