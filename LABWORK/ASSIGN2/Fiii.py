def read_matrix_from_file(filename):
    """Read matrix from text file with space-separated values"""
    matrix = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                # Remove whitespace and split by spaces
                line = line.strip()
                if line:  # Skip empty lines
                    row = []
                    numbers = line.split()
                    for num in numbers:
                        row.append(float(num))
                    matrix.append(row)
        return matrix
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return None
    except ValueError:
        print("Error: File contains invalid numbers!")
        return None

# Usage
matrix = read_matrix_from_file('matrix.txt')
if matrix:
    print("Matrix loaded successfully:")
    for row in matrix:
        print(row)
