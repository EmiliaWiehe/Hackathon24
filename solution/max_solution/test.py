import time
import numpy as np

def generate_powers_of_two_array(shape):
    """
    Generates a NumPy array of the given shape filled with random powers of two.
    
    Args:
        shape (tuple): The shape of the desired array (e.g., (3, 4)).
    
    Returns:
        np.ndarray: A NumPy array of signed 64-bit integers with powers of two.
    """
    # Define the possible powers of two within the 64-bit signed integer range
    min_exponent = 0  # Smallest power of two for signed 64-bit integers
    max_exponent = 61   # Largest power of two for signed 64-bit integers
    
    # Generate random exponents within the valid range
    random_exponents = np.random.randint(min_exponent, max_exponent + 1, size=shape)
    
    # Calculate powers of two
    powers_of_two = np.power(2, random_exponents, dtype=np.int64)
    
    return powers_of_two

def get_element_list(array):
    """Returns a set of all unique elements in the input array which
    are greater than 2^62."""
    return set(array[array > 2**62])

def get_powers_not_in_set(set):
    """Returns a set of all powers of two not present in the input set."""
    return {2**i for i in range(63) if 2**i not in set}

import numpy as np
from scipy.ndimage import rotate

def rotate_and_accumulate(original_array, initial_power=2, rotation_step=8, total_rotation=360):
    """
    Rotates the input array by increments, scales the ones by powers of two,
    and accumulates the results into a result array.

    Args:
        original_array (np.ndarray): Input 128x128 array with only zeros and ones.
        initial_power (int): The starting power for scaling ones (default is 2).
        rotation_step (int): The step size for rotation in degrees (default is 8).
        total_rotation (int): Total rotation in degrees (default is 360).

    Returns:
        np.ndarray: The resulting array after all rotations and additions.
    """
    # Ensure the array is of the correct size and binary
    assert original_array.shape == (128, 128), "Input array must be 128x128."
    assert np.array_equal(original_array, original_array.astype(bool)), "Array must contain only 0s and 1s."

    result_array = np.zeros_like(original_array, dtype=np.int64)

    num_steps = total_rotation // rotation_step
    for step in range(num_steps):
        # Rotate the array
        rotated_array = rotate(original_array, angle=rotation_step * (step + 1), reshape=False, order=1)

        # Threshold rotated array to make it binary
        rotated_array = (rotated_array >= 0.5).astype(np.int64)

        # Scale the ones by the current power of two
        scaled_array = rotated_array * (2 ** (initial_power + step))

        # Accumulate the result
        result_array += scaled_array

    # Add the original array to the result
    result_array += original_array * (2 ** initial_power)

    return result_array

def bitwise_or_set(set):
    """Performs a bitwise OR operation on all elements in the input set."""
    result = 0
    for element in set:
        result |= element
    return result

def get_powers_not_in_number(number):
    """Returns a set of all powers of two not present in the input number."""
    return {2**i for i in range(63) if not number & 2**i}

def main():

    from timeit import default_timer as timer
    print('import tensorflow')
    start = timer()
    import tensorflow
    end = timer()
    print('Elapsed time: ' + str(end - start))

    # Example usage
    original_array = np.random.choice([0, 1], size=(128, 128))  # Generate a random binary array
    result_array = rotate_and_accumulate(original_array)


    # Example usage:
    array_shape = (128, 128)  # Shape of the desired array
    random_array_1 = result_array

    # Define 5x5 array of random 0 and 2^62 values
    random_array_2 = np.random.choice([0, 2**62], array_shape)

    # Add the two arrays together
    random_array = random_array_1 + random_array_2

    # Start timer
    start_time = time.time()
    
    # Get the unique elements in the resulting array
    unique_elements = get_element_list(random_array)

    # Subtract 2^62 from all elemets of unique_elemets
    unique_elements = {element - 2**62 for element in unique_elements} 
    number = bitwise_or_set(unique_elements)
    print(get_powers_not_in_number(number))

    # Print the time taken
    print(f"Time taken: {time.time() - start_time:.6f} seconds")
"""
    print(random_array_1)
    print(random_array_2)
    # Print the number of elements in unique_elements
    print(len(unique_elements))
    print("a\n")
    print(unique_elements)
    print(get_powers_not_in_set(unique_elements))
"""
    
if __name__ == "__main__":
    main()