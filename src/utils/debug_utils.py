import numpy as np

def log_error(message):
    with open("error_log.txt", "a") as log_file:
        log_file.write(message + "\n")

def print_matrix_info(matrix, name):
    print(f"{name} Shape: {matrix.shape}")
    if isinstance(matrix, np.ndarray):
        sample = matrix[:5]
    else:
        sample = matrix[:5].toarray()
    
    print(f"{name} Sample (first 5 rows):")
    print(sample)
