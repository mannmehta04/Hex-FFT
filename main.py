import numpy as np
import hexjl  # Imports your converted script

def main():
    print("Initializing Hexagonal Data...")
    
    # 1. Create sample hexagonal data
    # Note: Because your fold_half function uses div2 (bitwise right shift), 
    # your matrix dimensions should ideally be even numbers (e.g., 4x4, 8x8).
    shape = (4, 4) 
    
    # Using random data for the sake of the test
    odd_rows = np.random.rand(*shape)
    even_rows = np.random.rand(*shape)
    
    # Instantiate your class
    hex_data = hexjl.OffsetHexData(odd_rows, even_rows)
    
    print("\n--- Original Data (Odd Rows Sample) ---")
    print(hex_data.odd_rows)

    # 2. Perform the Hexagonal FFT
    print("\nExecuting Hexagonal FFT...")
    fft_result = hex_data.fft()
    
    print("\n--- FFT Result (Odd Rows Sample) ---")
    print(fft_result.odd_rows) # This will be an array of complex numbers

    # 3. Perform the Inverse Hexagonal FFT
    print("\nExecuting Inverse Hexagonal FFT...")
    ifft_result = fft_result.ifft()
    
    # 4. Verify the results
    # The inverse FFT of an FFT should return the original data. 
    # Because of floating-point arithmetic, we use np.allclose instead of ==
    is_match = np.allclose(hex_data.odd_rows, ifft_result.odd_rows) and \
               np.allclose(hex_data.even_rows, ifft_result.even_rows)
               
    print("\n--- Verification ---")
    if is_match:
        print("Success! The Inverse FFT perfectly reconstructed the original data.")
    else:
        print("Mismatch! The reconstructed data differs from the original.")

if __name__ == "__main__":
    main()