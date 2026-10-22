# AmbiguousTextDiff: Text-Guided Diffusion for Medical Image Segmentation

Medical image segmentation often suffers from ambiguity due to unclear boundaries, expert inconsistencies, and varying interpretation standards. **AmbiguousTextDiff** introduces a novel text-guided diffusion model that generates diverse and plausible segmentation proposals reflecting the ambiguity observed in medical imaging. By combining text-conditional diffusion models with ambiguity-aware training, our approach generates multiple valid segmentations for a single input image, guided by descriptive text prompts that include anatomical and diagnostic attributes.

## Key Features

- **Text-Guided Segmentation**: Uses descriptive medical text prompts to guide segmentation generation
- **Ambiguity-Aware**: Generates multiple plausible segmentations instead of single deterministic outputs
- **Medical Domain Expertise**: Trained on LIDC-IDRI dataset with expert radiologist annotations
- **Comprehensive Evaluation**: Evaluated using specialized metrics for diversity and clinical validity
- **BioClinicalBERT Integration**: Leverages medical domain-specific language understanding

## Method Overview


AmbiguousTextDiff employs a conditional diffusion framework that:

1. **Encodes medical images** through a visual pathway
2. **Processes text prompts** using BioClinicalBERT for medical context
3. **Generates segmentations** via a text-guided U-Net denoiser with cross-attention
4. **Produces diverse outputs** through stochastic sampling and varied text conditioning

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/siddharthdhara/AmbiguousTextDiff.git
cd AmbiguousTextDiff

# Install dependencies
pip install -r requirements.txt
```

### Download Datasets

1. **LIDC-IDRI Dataset**: 
   - Download from [Kaggle](https://www.kaggle.com/datasets/zhangweiled/lidcidri)
   - Or from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/)

2. **Text Annotations**:
   - Available on [Hugging Face](https://huggingface.co/datasets/siddharthdhara17/lidc-idri-text-annotations)

3. **Pre-trained Model Checkpoints**:
   - Available on [Google Drive](https://drive.google.com/drive/folders/1u6JsPTwecPdl14MHhhRQSdKyVDDDmM-U?usp=sharing)

## Project Structure

```
AmbiguousTextDiff/
├── guided_diffusion/              # Core diffusion model implementation
│   ├── unet.py                   # Text-guided U-Net architecture
│   ├── text_encoder.py           # BioClinicalBERT text encoder
│   ├── cross_attention.py        # Cross-modal attention mechanisms
│   ├── gaussian_diffusion.py     # Diffusion process implementation
│   ├── dataset.py                # Data loading and preprocessing
│   └── ...
├── scripts/                       # Training and inference scripts
│   ├── train_ambiguous_textdiff.py      # Model training
│   ├── sample_ambiguous_textdiff.py     # Generate segmentations
│   ├── evaluate_ambiguous_textdiff.py   # Model evaluation
│   └── *.sh                      # Shell scripts for execution
├── data/                          # Dataset directory
│   ├── training/                  # Training data
│   │   ├── lidc_annotations_text_augmented_train.csv  # Training text annotations
│   │   ├── train_list.txt         # Training file list
│   │   └── LIDC-IDRI-*/          # Patient folders
│   └── testing/                   # Test data
│       ├── lidc_annotations_text_augmented_test.csv   # Test text annotations
│       ├── test_list.txt          # Test file list
│       └── LIDC-IDRI-*/          # Patient folders
├── results/                       # Experimental results
│   ├── ambiguous_textdiff/       # Model outputs
├── lidc_annotations_text_augmented.csv     # Text annotations for LIDC-IDRI
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### 1. Data Preparation

Organize your LIDC-IDRI dataset in the following structure:

```
data/
├── training/
│   ├── lidc_annotations_text_augmented_train.csv  # Training text annotations
│   ├── train_list.txt         # Training file list
│   ├── LIDC-IDRI-0001/
│   │   ├── nodule_1/
│   │   │   ├── images/
│   │   │   │   ├── slice_001.png
│   │   │   │   └── ...
│   │   │   └── masks/
│   │   │       ├── slice_001.png
│   │   │       └── ...
│   │   └── nodule_2/
│   └── ...
└── testing/
    ├── lidc_annotations_text_augmented_test.csv   # Test text annotations
    ├── test_list.txt          # Test file list
    └── LIDC-IDRI-*/          # Patient folders (similar structure)
```

### 2. Training

```bash
# Using shell script
bash scripts/train_ambiguous_textdiff.sh
```

### 3. Inference/Sampling

```bash
# Using shell script
bash scripts/sample_ambiguous_textdiff.sh
```

### 4. Evaluation

```bash
# Using shell script
bash scripts/evaluate_ambiguous_textdiff.sh
```

## Text Annotations Format

The text annotations provide rich semantic information for each nodule:

```csv
file_name,Text
LIDC-IDRI-0001,"The annotation id is 84. The subtlety is 5. The internalStructure is 1. The calcification is 6. The sphericity is 3. The margin is 3. The lobulation is 3. The spiculation is 4. The texture is 5. The malignancy is 5. The Subtlety is Obvious. The InternalStructure is Soft Tissue. The Calcification is Absent. The Sphericity is Ovoid. The Margin is Medium Margin. The Lobulation is Medium Lobulation. The Spiculation is Near Marked Spiculation. The Texture is Solid. The Malignancy is Highly Suspicious. The diameter mm is 32.21. The volume mm3 is 7869.37..."
```

Each annotation includes:
- **Radiological characteristics**: subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy
- **Quantitative measurements**: diameter, volume, slice indices, bounding box
- **Natural language descriptions**: Converted numerical ratings to clinical terminology

**Note on prompts:** the annotation files also store diameter, volume, slice indices, and bounding boxes, but the text encoder truncates prompts at 128 tokens, so these trailing fields are never seen by the model (the numeric ratings and categorical labels up to spiculation are). The leading annotation id is an arbitrary database index, not a clinical descriptor.

Supplementary material (pseudocode, hyperparameters, extra results, baseline training details): [docs/supplementary.pdf](docs/supplementary.pdf).

## Evaluation Metrics

Our evaluation framework uses specialized metrics for ambiguous segmentation:

- **Combined Sensitivity**: Measures overall detection capability across multiple annotations
- **Diversity Agreement**: Quantifies how well generated diversity matches expert variability  
- **Generalized Energy Distance (GED)**: Assesses distributional similarity between generated and ground truth segmentations
- **Collective Insight (CI) Score**: Comprehensive measure combining accuracy and uncertainty modeling


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


