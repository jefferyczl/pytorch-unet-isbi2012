# pytorch-unet-isbi2012
This repository is aim to recurrent the paper of the unet with pytorch.And our dataset is isbi2012.
## 1. Introduction to Project File Structure

This project adopts a modular design with clear file functions and a well-structured layout, as follows:

```plaintext
project/
├── isbi2012/                 # Dataset directory (binary cell segmentation)
│   ├── train/
│   │   ├── imgs/             # Original training set images (PNG format)
│   │   └── labels/           # Training set segmentation labels (PNG format)
│   └── test/
│       ├── imgs/             # Original test set images (PNG format)
│       └── labels/           # Test set segmentation labels (PNG format)
├── outputs/                  # Output directory
│   └── unet_xxx.pth          # Model weight files generated during training
├── dataset.py                # Data loading and preprocessing module
├── u_transforms.py           # Data augmentation function
├── model.py                  # UNet model definition module (supports precision switching)
├── train.py                  # Model training and validation module (core experiment script)
├── metrics.py                # Evaluation metric calculation module (accuracy, IoU, etc.)
└── predict.py                # Model prediction and result visualization module
```

## 2. Deployment Process

### 2.1 Environment Requirements

- **Hardware**: NVIDIA GPU (supporting CUDA, recommended RTX 3090 or higher, with FP16 computing capability)
- **Operating System**: Ubuntu 20.04.6 LTS 64-bit
- Software Dependencies:
  - Python 3.12.4
  - PyTorch 2.4.0
  - torchvision 0.19.0
  - numpy 1.26.4
  - matplotlib 3.7.5
  - Pillow 10.0.0
  - CUDA 11.8+ (must be compatible with the PyTorch version)

### 2.2 Environment Configuration Steps

1. Create a virtual environment(conda is recommended):

   ```bash
   conda create -n unet_precision python=3.12.4
   conda activate unet_precision
   ```

2. Install PyTorch and dependencies:

   ```bash
   # Install PyTorch (with CUDA support)
   pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118
   # Install other dependencies
   pip install numpy==1.26.4 matplotlib==3.7.5 Pillow==10.0.0
   ```

3. Verify the environment:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"  # Outputs True if GPU is available
   python -c "import torch; print(torch.backends.cudnn.version())"  # Verify CUDA configuration
   ```

## 3. Experiment Process

### 3.1 Dataset Preparation

1. Download the dataset:
   - Download the 2012 cell membrane segmentation dataset (ISBI2012) from the ISBI official website, which includes 30 training images and 30 test images in TIFF format.
2. Data preprocessing:
   - Convert TIFF format to PNG format (batch conversion can be done using Python scripts or tools).
   - Store according to the file structure requirements:
     - Training images: `isbi2012/train/imgs/`
     - Training labels: `isbi2012/train/labels/`
     - Test images: `isbi2012/test/imgs/`
     - Test labels: `isbi2012/test/labels/`

### 3.2 Experiment Configuration and Execution

This project can experiments through the `train.py` script, with core configurations specified via command-line parameters such as training/testing precision, encoder/decoder precision, etc.

#### Explanation of Experiment Parameters

| Parameter        | Meaning                                           | Optional Values           |
| ---------------- | ------------------------------------------------- | ------------------------- |
| `--data_root`    | Root directory of the dataset                     | e.g., `./ISBI2012`        |
| `--batch_size`   | Training batch size                               | 2, 4, 8, etc.             |
| `--total_epoch`  | Total number of training epochs                   | e.g., 1500                |
| `--lr`           | Learning rate                                     | e.g., 2e-3                |
| `--loss_type`    | Loss function                                     | eg., binary_cross_entropy |
| `--lr_policy     |                                                   | eg., CosineAnnealingLR    |
| `--train_enc_fp` | Encoder training precision (overrides `train_fp`) | 32, 16                    |
| `--train_dec_fp` | Decoder training precision (overrides `train_fp`) | 32, 16                    |

#### Training command

```bash
python ./train.py \
--data_root ./ \
--batch_size 2 \
--save_val_results \
--total_epoch 1501 \
--model unet \
--dataset isbi2012 \
--lr 2e-3 \
--loss_type binary_cross_entropy \
--lr_policy CosineAnnealingLR 
```

### 3.3 Saving of Experiment Results

- Model weights: Saved in the `outputs/` directory, named in the format `best_unet_isbi2012_os16.pth`
- Evaluation metrics: Output in the console, including loss per epoch, accuracy, MIoU, training/inference time

## 4. Reproduction Process

### 4.1 Complete Reproduction Steps

1. **Environment deployment**: Configure the dependent environment according to "II. Deployment Process".
2. **Data preparation**: Download the ISBI2012 dataset, convert it to PNG format, and store it according to the directory structure.
3. **Model training**: Run the commands in "III. Experiment Process" in sequence and record the results of each experiment.
4. **Result verification**: Use `predict.py` to load the trained model and verify the segmentation effect.

### 4.2 Usage of Prediction Script

1. Validate all validation sets:

   ```bash
   python ./predict.py
   ```

### 4.3 Result

1. **Reference conclusions**:

   | **Experiment No.** | **learning  rate** | **weight  decay** | **momentum** | **Accuracy (%)** | **Class Accuracy (%)** | **IoU (%)** |
   | ------------------ | ------------------ | ----------------- | ------------ | ---------------- | ---------------------- | ----------- |
   | 1                  |                    |                   |              |                  |                        |             |
   | 2                  |                    |                   |              |                  |                        |             |
   | 3                  |                    |                   |              |                  |                        |             |
   | 4                  |                    |                   |              |                  |                        |             |
   | 5                  |                    |                   |              |                  |                        |             |
   | 6                  |                    |                   |              |                  |                        |             |

## 5. Notes

1. **Precision compatibility**: Ensure that the GPU supports FP16 (e.g., RTX 3090), otherwise, FP16 experiments may report errors.
2. **Hyperparameter adjustment**: If the model does not converge, adjust the learning rate (e.g., 2e-3) or batch size (e.g., 2).
3. **Data format**: ISBI2012 labels must be binarized images (0 = background, 1 = cell) to avoid format errors.
4. **Log recording**: It is recommended to redirect the experiment output to a file (e.g., `python train.py > experiment1.log`) for subsequent analysis.
