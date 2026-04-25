"""
Configuración de entrenamiento para Multi-Scale FCOS con Query Conditioning
"""
import torch
import os

class Config:
    # ======================== PATHS ========================
    # Base path (directorio home)
    BASE_PATH = os.path.expanduser("~")  # /home/rvdl_2

    DATASET_JSON = os.path.join(BASE_PATH, "detection_dataset_sketches.json")
    IMAGES_DIR = os.path.join(BASE_PATH, "DocExplore_images")
    SKETCHES_DIR = os.path.join(BASE_PATH, "Sketches")
    CHECKPOINT_DIR = os.path.join(BASE_PATH, "fcos_training", "checkpoints")
    LOG_DIR = os.path.join(BASE_PATH, "fcos_training", "logs")

    BACKBONE_CKPT = os.path.join(BASE_PATH, "idoc_pretrained.pth")
    BACKBONE_TYPE = "vit_base"
    PATCH_SIZE = 16
    FEATURE_DIM = 768  # ViT-Base output dimension
    
    # 🔒 IMPORTANTE: Congelar backbone
    FREEZE_BACKBONE = True  # NUEVO: Congela el encoder ViT
    
    # FPN Configuration
    FPN_OUT_CHANNELS = 256
    FPN_LEVELS = 4  # P3, P4, P5, P6
    FPN_STRIDES = [8, 16, 32, 64]
    
    # Object size ranges for each FPN level (in pixels)
    FPN_SIZE_RANGES = [
        (0, 64),        # P3: small objects
        (64, 128),      # P4: medium-small objects
        (128, 256),     # P5: medium-large objects
        (256, 999999)   # P6: large objects
    ]
    
    # ======================== TRAINING ========================
    # Image sizes
    TRAIN_SIZES = [800, 1024, 1333]  # Random choice during training
    MAX_SIZE = 1600
    TEST_SIZE = 1024
    
    # Sketch size (fixed)
    SKETCH_SIZE = 224
    
    # Optimization
    BATCH_SIZE = 2  # Per GPU
    ACCUMULATION_STEPS = 8  # Effective batch size = 2 * 8 = 16
    NUM_EPOCHS = 50
    
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 2
    LR_DECAY_EPOCHS = [30, 40]
    LR_DECAY_GAMMA = 0.1
    
    # Loss weights
    LOSS_WEIGHT_CLS = 1.0
    LOSS_WEIGHT_REG = 1.0
    LOSS_WEIGHT_CTR = 1.0
    
    # FCOS specific
    CENTER_SAMPLING_RADIUS = 1.5
    IOU_LOSS_TYPE = "giou"  # "iou", "giou", "diou", "ciou"
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    
    # ======================== DATA ========================
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Score threshold for filtering noisy boxes
    BOX_SCORE_THRESHOLD = 0.6
    
    # Data augmentation
    USE_AUGMENTATION = True
    HORIZONTAL_FLIP_PROB = 0.5
    COLOR_JITTER = True
    
    # Class balancing
    USE_CLASS_BALANCING = True
    
    # ======================== EVALUATION ========================
    EVAL_INTERVAL = 2  # Evaluate every N epochs
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    
    # NMS
    NMS_THRESHOLD = 0.6
    SCORE_THRESHOLD = 0.05
    MAX_DETECTIONS_PER_IMAGE = 100
    
    # mAP calculation
    IOU_THRESHOLDS = [0.5, 0.75]  # For AP@50, AP@75
    
    # ======================== SYSTEM ========================
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # Mixed precision training
    USE_AMP = True
    
    # Gradient clipping
    CLIP_GRAD_NORM = 35.0
    
    # ======================== LOGGING ========================
    PRINT_FREQ = 50  # Print every N iterations
    WANDB_PROJECT = "fcos-query-detection"
    USE_WANDB = False  # Set to True if you want to use Weights & Biases
    
    @classmethod
    def display(cls):
        """Pretty print configuration"""
        print("=" * 70)
        print("CONFIGURATION".center(70))
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:.<50} {value}")
        print("=" * 70)