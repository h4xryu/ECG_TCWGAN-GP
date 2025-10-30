# ECG_TCWGAN-GP
ECG Generator using TCWGAN-GP

![nn](https://i.ifh.cc/Y2H9l5.gif)

TransGAN for ECG generation 

## Key Changes

### Signal Length: 200-1800 samples
- 200 samples (~0.56 seconds @ 360Hz)
- Enables learning of longer-range ECG patterns

### Generator Architecture
```python
# TransGAN.py
Generator(output_length=200)

```

### Discriminator Architecture
```python
# TransGAN.py
Discriminator(signal_length=200)
```

### Sample Saving Method
Original approach saved 10 samples per epoch in a single 2x5 grid image.

New approach saves 20 individual samples per epoch:

```
generated_samples/
└── S_epochs/
    ├── epoch_0001.png/
    ├── epoch_0002.png/
    ├── epoch_0003.png/
    ├── epoch_0004.png/
    ├── epoch_0005.png/
    │   ...
    ├── epoch_0006.png/
    ├── epoch_0007.png/
    └── epoch_0008.png/
        
```

## File Structure

```
TransGAN.py                      # Generator/Discriminator for 1800 samples
train_TransGAN.py                # Training script
```

## Usage

### Step 1: Build Dataset
```bash
python gen_dataset_inter_patient_200s.py
```

Output files:
- `dataset/mitbih_interpatient/mitbih_test.npz`
- `dataset/mitbih_interpatient/mitbih_test.npz`

### Step 2: Train GAN


Training outputs:
- Model checkpoints: `log/model_S/models/S_epoch_*.pth`
- Generated samples: `generated_samples/S_epochs/epoch_*/`
- TensorBoard logs: `log/model_S/logs/`

### Step 3: Apply GAN Augmentation
Configure in `gen_dataset_inter_patient_200s.py`:
```python
GAN_AUGMENT_CONFIG = {
    'S': {
        'enabled': True,
        'checkpoint': 'log/model_S/models/S_epoch_50.pth',
        'multiplier': 4
    },
    'V': {
        'enabled': True,
        'checkpoint': 'log/model_V/models/V_epoch_50.pth',
        'multiplier': 2
    },
    'F': {
        'enabled': True,
        'checkpoint': 'log/model_F/models/F_epoch_50.pth',
        'multiplier': 4
    }
}
```

## Data Extraction Window

### Original (200 samples)
```python
pre = int(round(90 * fs / 360.0))   # 90 samples before R-peak
post = int(round(110 * fs / 360.0)) # 110 samples after R-peak
# Total: 200 samples
```

## Training Output Example

```
[INIT] Using TransWGAN-GP (lambda_gp=10, n_critic=5)
[INIT] Target label = S
[INIT] Signal length = 200 samples 
[DATA] Target S: 472 train / 162 val samples

Epoch 1: Loss_D=-2.3451, Loss_G=1.8234, GP=0.1234
  -> Saved 20 samples to: generated_samples_1800/S_epochs/epoch_0001/
     Signal stats - shape: (200,), range: [-2.345, 2.123]
```

Sample file structure:
```
generated_samples/F_epochs/epoch_0050/
├── sample_01.png  (
├── sample_01.npy  
├── sample_02.png
├── sample_02.npy
...
├── sample_20.png
└── sample_20.npy
```

## Model Parameters

### Generator
- `noise_dim`: 256
- `embed_dim`: 1024
- `output_length`: 200

### Discriminator
- `signal_length`: 200
- Convolution layers: 6 (stride=2 downsampling)

### Training
- `batch_size`: 64
- `learning_rate`: 0.00001 (Adam for WGAN-GP)
- `lambda_gp`: 10
- `n_critic`: 5


## References

- TransGAN for ECG Generation
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- WGAN-GP: "Improved Training of Wasserstein GANs"
