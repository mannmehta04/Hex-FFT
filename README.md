# HexFFT: Hexagonal Fast Fourier Transform & Image Compression

HexNet is a Python-based pipeline for simulating hexagonal image sampling, computing the Hexagonal Discrete Fourier Transform (HDFT), and performing highly efficient frequency-domain image compression. This repository adapts the logic of `HexFFT.jl` for Python, demonstrating the inherent data dominance of hexagonal grids over standard Cartesian (rectangular) pixel grids.

## 📽️ Video
Phase 1: https://www.youtube.com/watch?v=7Ts3xIU5_zI
Phase 2: https://youtu.be/TH2PuhVISWs?si=wxO4DLKrRiIZ41Am

## 📖 Theoretical Background

Standard digital images are captured and displayed using an orthogonal (rectangular) grid of pixels. However, biological vision systems (like the human fovea) and advanced optical sensors utilize hexagonal lattice structures. 

The mathematical advantage of hexagonal sampling is rooted in the Nyquist-Shannon sampling theorem. For a circularly band-limited signal, the optimal sampling lattice is hexagonal, requiring **13.4% fewer samples** than a rectangular grid to completely recover the signal without aliasing. This foundational theory was established by Peterson and Middleton and later expanded upon by Russell M. Mersereau, who developed the algorithms for the Hexagonal Discrete Fourier Transform (HDFT). Mersereau showed that this optimal sampling density directly translates to a computational savings of 13.4% or more when performing recursive operations. 

By processing images in the hexagonal frequency domain, we can achieve superior compression rates while maintaining the structural integrity of the original image.

## ⚙️ Methodology & Pipeline Procedure

The `img_prc.py` script executes a full end-to-end data processing pipeline:

### 1. Virtual Hexagonal Sampling (Row Shifting)
Since standard image files (like `.jpg` or `.png`) do not natively contain hexagonal data, the script simulates a hexagonal lattice by splitting the Cartesian image array into `odd_rows` and `even_rows`. This staggered approach mimics an offset hexagonal coordinate system. To ensure mathematical symmetry during the nested folding operations of the FFT, the input image dimensions are dynamically cropped to the nearest multiple of 4.

### 2. Hexagonal Discrete Fourier Transform (HFFT)
The spatial image data is transformed into the frequency domain using the `OffsetHexData.fft()` method. Unlike standard 2D FFTs, the HFFT accounts for the staggered phase shifts of the hexagonal lattice, resulting in a frequency spectrum that exhibits natural 6-fold hexagonal symmetry.

### 3. Ideal Low-Pass Filtering (Compression)
In the frequency domain, structural image data (broad shapes) is concentrated at the center (low frequencies), while edges and noise reside at the periphery (high frequencies). The script applies a radial boolean mask to the shifted spectrum:
$$H(u,v) = \begin{cases} 1 & \text{if } D(u,v) \leq D_0 \\ 0 & \text{if } D(u,v) > D_0 \end{cases}$$
Where $D_0$ is the cutoff radius determined by the `keep_ratio` parameter. By setting this ratio to `0.10`, the pipeline zeroes out 90% of the frequency data, achieving extreme compression.

### 4. Inverse Transformation & Reconstruction
The compressed, shifted frequency arrays are unshifted and passed through the Inverse Hexagonal Fast Fourier Transform (IHFFT). The real components of the resulting complex arrays are interlaced back into a single 2D image array.

### 5. Performance Metrics
To quantify the data loss from the compression step, the script calculates the Mean Squared Error (MSE) between the original Cartesian image ($I$) and the compressed hexagonal reconstruction ($K$):
$$MSE = \frac{1}{M N} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} [I(i, j) - K(i, j)]^2$$

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/hexnet.git
   cd hexnet
   ```

2. **Install required dependencies:**
   This project strictly requires `numpy` for array mathematics and `matplotlib`/`Pillow` for image processing and visualization.
   ```bash
   pip install numpy matplotlib Pillow
   ```

## 💻 Usage

To run the compression pipeline on a standard image:

1. Place your target image (e.g., `f1.jpg`) in the root directory.
2. Open `img_prc.py` and modify the execution call at the bottom of the script if you are using a different filename:
   ```python
   if __name__ == "__main__":
       process_image("your_image.jpg") 
   ```
3. Execute the script:
   ```bash
   python img_prc.py
   ```

**Output:**
The script will output the structural MSE to the terminal and launch a `matplotlib` figure displaying:
1. The Original Image.
2. The Hexagonal FFT Magnitude Spectrum (log-scaled and centered).
3. The Compressed Reconstruction (Inverse FFT).

## 📄 File Structure

* `hexjl.py`: The core mathematical library handling the nested Fourier transformations and complex coefficient generation for offset hexagonal grids (adapted from Julia).
* `img_prc.py`: The execution pipeline that handles image loading, grid simulation, frequency masking (compression), error calculation, and plotting.

## 📚 References

1. **Mersereau, R. M. (1979).** "The processing of hexagonally sampled two-dimensional signals." *Proceedings of the IEEE*, 67(6), 930-949. (Establishes the foundational HDFT algorithm and computational efficiency).
2. **Petersen, D. P., & Middleton, D. (1962).** "Sampling and reconstruction of wave-number-limited functions in N-dimensional euclidean spaces." *Information and Control*, 5(4), 279-323. (Proves the optimal 13.4% efficiency of hexagonal sampling for circularly band-limited signals).
3. **HexFFT.jl:** Original Julia implementation of the offset Hexagonal FFT logic utilized for translation in this repository ([https://github.com/gwater/HexFFT.jl](https://github.com/gwater/HexFFT.jl)).
