import subprocess
import numpy as np
from skimage import io, img_as_ubyte
from scipy.interpolate import interp2d


tiff_filename = 'input.tiff'

# Step 2: Python Initials
raw_image = io.imread(tiff_filename)
bits_per_pixel = raw_image.dtype.itemsize * 8
height, width = raw_image.shape
raw_image = raw_image.astype(np.float64) / (2**bits_per_pixel - 1)

# Step 3: Linearization
black_level = 512
white_level = 16383
linear_image = (raw_image - black_level) / (white_level - black_level)
linear_image = np.clip(linear_image, 0, 1)

# Step 4: Identifying the Correct Bayer Pattern
bayer_pattern = 'rggb'  # Example Bayer pattern

# Step 5: White Balancing
r_scale = 2.525858
g_scale = 1.0
b_scale = 1.265026

white_world_balanced = linear_image / [r_scale, g_scale, b_scale]
gray_world_balanced = linear_image / np.mean(linear_image)
preset_balanced = linear_image * [r_scale, g_scale, b_scale]

# Step 6: Demosaicing
demosaiced_image = interp2d(np.arange(0, width, 2), np.arange(0, height, 2), linear_image[bayer_pattern[0::2, 0::2]])

# Step 7: Color Space Correction
MsRGB_to_cam = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

M_cam_to_sRGB = np.linalg.inv(MsRGB_to_cam)
corrected_image = np.dot(linear_image, M_cam_to_sRGB.T)

# Step 8: Brightness Adjustment and Gamma Encoding
brightness_factor = 0.25
adjusted_image = corrected_image * brightness_factor
adjusted_image = np.clip(adjusted_image, 0, 1)
gamma_encoded_image = np.where(adjusted_image <= 0.0031308,
                               12.92 * adjusted_image,
                               (1 + 0.055) * np.power(adjusted_image, 1 / 2.4) - 0.055)

# Step 9: Compression
compressed_image_png = img_as_ubyte(gamma_encoded_image)
compressed_image_jpeg = img_as_ubyte(gamma_encoded_image)

io.imsave('output.png', compressed_image_png)
io.imsave('output.jpg', compressed_image_jpeg, quality=95)