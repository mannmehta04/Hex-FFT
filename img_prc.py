import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import hexjl  

def process_image(image_path):
    print("Loading and preparing image...")
    # 1. Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    img_data = np.array(img, dtype=float)

    # 2. Ensure dimensions are multiples of 4 
    h, w = img_data.shape
    h = h - (h % 4)
    w = w - (w % 4)
    img_data = img_data[:h, :w]

    # 3. Map to Hexagonal Offset Data
    even_rows = img_data[0::2, :]
    odd_rows = img_data[1::2, :]
    hex_data = hexjl.OffsetHexData(odd_rows, even_rows)

    # 4. Perform the Hexagonal FFT
    print("Executing Hexagonal FFT...")
    fft_result = hex_data.fft()

    # 5. Extract and Shift Magnitude Spectrum
    from numpy.fft import fftshift
    shifted_odd = fftshift(fft_result.odd_rows)
    shifted_even = fftshift(fft_result.even_rows)

    mag_odd_log = np.log1p(np.abs(shifted_odd))
    mag_even_log = np.log1p(np.abs(shifted_even))

    spectrum = np.zeros((h, w))
    spectrum[0::2, :] = mag_even_log
    spectrum[1::2, :] = mag_odd_log

    # 6. Apply Low-Pass Filter (Compression)
    print("Applying Low-Pass Filter (Compression)...")
    keep_ratio = 0.10 
    
    rows_odd, cols_odd = shifted_odd.shape
    crow_odd, ccol_odd = rows_odd // 2, cols_odd // 2
    radius_odd = int(min(rows_odd, cols_odd) * keep_ratio)
    
    Y_odd, X_odd = np.ogrid[:rows_odd, :cols_odd]
    dist_odd = np.sqrt((Y_odd - crow_odd)**2 + (X_odd - ccol_odd)**2)
    mask_odd = dist_odd <= radius_odd  

    rows_even, cols_even = shifted_even.shape
    crow_even, ccol_even = rows_even // 2, cols_even // 2
    radius_even = int(min(rows_even, cols_even) * keep_ratio)
    
    Y_even, X_even = np.ogrid[:rows_even, :cols_even]
    dist_even = np.sqrt((Y_even - crow_even)**2 + (X_even - ccol_even)**2)
    mask_even = dist_even <= radius_even

    compressed_shifted_odd = shifted_odd * mask_odd
    compressed_shifted_even = shifted_even * mask_even

    # 7. Unshift and prepare for IFFT
    from numpy.fft import ifftshift
    compressed_odd = ifftshift(compressed_shifted_odd)
    compressed_even = ifftshift(compressed_shifted_even)
    compressed_hex_data = hexjl.OffsetHexData(compressed_odd, compressed_even)

    # 8. Reconstruct the image using the COMPRESSED data
    print("Executing Inverse Hexagonal FFT on Compressed Data...")
    ifft_result_compressed = compressed_hex_data.ifft()
    
    reconstructed_compressed = np.zeros((h, w))
    reconstructed_compressed[0::2, :] = np.real(ifft_result_compressed.even_rows)
    reconstructed_compressed[1::2, :] = np.real(ifft_result_compressed.odd_rows)

    # 9. Calculate Compression Metrics (MSE)
    mse = np.mean((img_data - reconstructed_compressed) ** 2)
    print(f"\n--- Compression Metrics ---")
    print(f"Keep Ratio: {keep_ratio * 100}%")
    print(f"Mean Squared Error (MSE): {mse:.2f}")

    # 10. Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_data, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(spectrum, cmap='gray') 
    axes[1].set_title('Hex FFT Magnitude Spectrum')
    axes[1].axis('off')

    axes[2].imshow(reconstructed_compressed, cmap='gray')
    axes[2].set_title(f'Compressed Reconstruction\n(MSE: {mse:.2f})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_image("f1.jpg")