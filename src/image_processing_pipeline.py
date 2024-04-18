import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from scipy.interpolate import RegularGridInterpolator

# Constants from dcraw recon
BLACK_LEVEL = 0
WHITE_LEVEL = 16383
R_SCALE = 1.628906
G_SCALE = 1.0
B_SCALE = 1.386719

# Read and normalize the TIFF image
image_path = '../data/baby.tiff'
raw_image = imread(image_path).astype(np.float64)
print(f"Image shape: {raw_image.shape}, Data type: {raw_image.dtype}, Max value: {np.max(raw_image)}")

# Linearization
image = (raw_image - BLACK_LEVEL) / (WHITE_LEVEL - BLACK_LEVEL)
image = np.clip(image, 0, 1)

# Testing different Bayer patterns and White Balancing
# testing = "rggb"

# if (testing == "rggb"):
#     r = image[0::2, 0::2] * R_SCALE
#     g1 = image[0::2, 1::2] * G_SCALE
#     g2 = image[1::2, 0::2] * G_SCALE
#     b = image[1::2, 1::2] * B_SCALE
#     green = (g1 + g2) / 2
# elif (testing == "bggr"):
#     b = image[0::2, 0::2] * B_SCALE
#     g1 = image[0::2, 1::2] * G_SCALE
#     g2 = image[1::2, 0::2] * G_SCALE
#     r = image[1::2, 1::2] * R_SCALE
#     green = (g1 + g2) / 2
# elif (testing == "grbg"):
#     g1 = image[0::2, 0::2] * G_SCALE
#     r = image[0::2, 1::2] * R_SCALE
#     b = image[1::2, 0::2] * B_SCALE
#     g2 = image[1::2, 1::2] * G_SCALE
#     green = (g1 + g2) / 2
# elif (testing == "gbrg"):
#     g1 = image[0::2, 0::2] * G_SCALE
#     b = image[0::2, 1::2] * B_SCALE
#     r = image[1::2, 0::2] * R_SCALE
#     g2 = image[1::2, 1::2] * G_SCALE
#     green = (g1 + g2) / 2

# # White World
# r = image[0::2, 0::2]
# g1 = image[0::2, 1::2]
# g2 = image[1::2, 0::2]
# b = image[1::2, 1::2]

# avg_r = np.mean(r)
# avg_g = (np.mean(g1) + np.mean(g2)) / 2
# avg_b = np.mean(b)

# # Compute scaling factors
# max_avg = max(avg_r, avg_g, avg_b)
# scale_r = max_avg / avg_r
# scale_g = max_avg / avg_g
# scale_b = max_avg / avg_b

# # Apply scales
# r *= scale_r
# g1 *= scale_g
# g2 *= scale_g
# b *= scale_b
# green = g1

# # Gray World
# r = image[0::2, 0::2]
# g1 = image[0::2, 1::2]
# g2 = image[1::2, 0::2]
# b = image[1::2, 1::2]

# # Calculate averages for each channel
# avg_r = np.mean(r)
# avg_g = (np.mean(g1) + np.mean(g2)) / 2
# avg_b = np.mean(b)

# # Compute overall average
# overall_avg = (avg_r + avg_g + avg_b) / 3

# # Compute scaling factors
# scale_r = overall_avg / avg_r
# scale_g = overall_avg / avg_g
# scale_b = overall_avg / avg_b

# # Apply scales
# r *= scale_r
# g1 *= scale_g
# g2 *= scale_g
# b *= scale_b
# green = g1

# Custom White Balancing
r = image[0::2, 0::2] * R_SCALE
g1 = image[0::2, 1::2] * G_SCALE
g2 = image[1::2, 0::2] * G_SCALE
b = image[1::2, 1::2] * B_SCALE
green = g1

# Adjust the range of full_x and full_y to match the interpolated coordinates
height, width = raw_image.shape
x = np.arange(0, width, 2)
y = np.arange(0, height, 2)
full_x = np.linspace(0, width - 1, width)  # Ensuring coordinates are within bounds
full_y = np.linspace(0, height - 1, height)
full_x, full_y = np.meshgrid(full_x, full_y, indexing='xy')

# Create RegularGridInterpolator instances
red_interp = RegularGridInterpolator((y, x), r, bounds_error=False, fill_value=None)
green_interp = RegularGridInterpolator((y, x), green, bounds_error=False, fill_value=None)
blue_interp = RegularGridInterpolator((y, x), b, bounds_error=False, fill_value=None)

# Use the interpolators to interpolate data
coords = np.stack([full_y.ravel(), full_x.ravel()], axis=-1)  # Flatten the full_x and full_y grids and stack
red = red_interp(coords).reshape(height, width)
green = green_interp(coords).reshape(height, width)
blue = blue_interp(coords).reshape(height, width)

rgb_image = np.stack((red, green, blue), axis=-1)

# Color space correction
M_cam_to_sRGB = np.linalg.inv([[0.4124564, 0.3575761, 0.1804375],
                              [0.2126729, 0.7151522, 0.0721750],
                              [0.0193339, 0.1191920, 0.9503041]])
rgb_image = np.dot(rgb_image, M_cam_to_sRGB.T)

# Brightness adjustment and gamma encoding
grayscale = rgb2gray(rgb_image)
scale_factor = 0.25 / np.mean(grayscale)
rgb_image *= scale_factor
rgb_image = np.clip(rgb_image, 0, 1)
rgb_image = np.where(rgb_image <= 0.0031308, 12.92 * rgb_image, 1.055 * np.power(rgb_image, 1/2.4) - 0.055)

# Compression
rgb_image_uint8 = (rgb_image * 255).astype(np.uint8)
imsave('../data/output.png', rgb_image_uint8)
imsave('../data/output.jpg', rgb_image_uint8, quality=95)

plt.imshow(rgb_image)
plt.title('Processed Image')
plt.axis('off')
plt.show()