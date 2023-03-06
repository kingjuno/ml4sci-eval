# Specific Test VI. Image Super-resolution 

**Task:** Train a deep learning-based super resolution algorithm of your choice to upscale low-resolution strong lensing images using the provided high-resolution samples as ground truths. Please implement your approach in PyTorch or Keras and discuss your strategy.

**Dataset Description:** The dataset comprises strong lensing images with no substructure at multiple resolutions: high-resolution (HR) and low-resolution (LR).

**Evaluation Metrics:** MSE (Mean Squared Error), SSIM (Structural Similarity Index), PSNR (Peak Signal-to-Noise Ratio)

# Solution
## Data Preparation
I didnt do any data augmentation for this task.
## Model 
I used two models for this task.
### SRCNN
The SRCNN is a deep convolutional neural network that learns end-to-end mapping of low resolution to high resolution images. I started using this since this is less complex than SRGAN and can be trained faster. 
For this task, I used the pytorch implementation by [yjn870](https://github.com/yjn870/SRCNN-pytorch). I tried several implementation like using UpsampleBlock, using ConvTranspose2d, but training on resized image with a skip connection gave me better results.

The model is defined as follows:
```python
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=8, padding=5 // 2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=2, padding=5 // 2)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = self.conv3(x2)
        return x3+x
```

I have used Adam optimizer with learning rate 0.001 and MSE loss function. The model was trained on a Kaggle GPU P100 with 16GB VRAM for faster computation. StepLR with step size 3 and gamma = 0.1 is used to adjust the learning rate. The model was trained for 30 epochs.

### FSRCNN
FSRCNN (Fast Super-Resolution Convolutional Neural Network) is an improvement over SRCNN in terms of both accuracy. FSRCNN replaces the bicubic interpolation step in SRCNN with a deconvolution layer and introduces a shrinking and expanding stage to reduce the number of parameters in the model.


Model definition:
```python
class FSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=1, d=64, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

    def forward(self, x):
        x1 = self.first_part(x)
        x2 = self.mid_part(x1)
        x3 = self.last_part(x2)
        return x3
```

I have used Adam optimizer with learning rate 0.001 and MSE loss function. The model was trained on a Kaggle GPU P100 with 16GB VRAM for faster computation. StepLR with step size 3 and gamma = 0.1 is used to adjust the learning rate. The model was trained for 10 epochs.

### SRGAN
SRGAN is a generative adversarial network for single image super-resolution. For implementation I have used a highly customizable pytorch GAN library that I developed and maintain [GANETIC](https://github.com/kingjuno/ganetic). The library is still in development and I am working on adding more features to it. The generator had 15 residual blocks with no last layer activation(for original srgan there are 5 residual blocks and uses (tanh(x)+1)/2 as the last layer activation).

I have used Adam optimizer with learning rate 0.0001 learning rate for both Discriminator and Generator. I have trained the model for 30 epochs. The model was trained on a Kaggle GPU P100 with 16GB VRAM for faster computation. StepLR with step size 15 and gamma = 0.1 is used to adjust the learning rate. MSELoss was used for the generator and BCELoss was used for the discriminator.

## Results
| Model | PSNR | MSE | SSIM |
| --- | --- | --- | --- |
| Resize - Bilinear Interpolation | 41.506622314453125 | 0.000071 | 0.9701901078224182 |
| SRCNN | 42.21138000488281 | 0.000060 | 0.9741216897964478 |
| FSRCNN | 42.2563591003418 | 0.000060 | 0.9742863178253174 |
| SRGAN | 42.09272003173828 | 0.000062 | 0.9736785888671875 |

## Result - Visualizations
Visualizations are done in 3 rows and can be found at the end of respective notebooks. First row is the input image, second row is the ground truth and third row is the output of the model.