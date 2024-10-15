import numpy as np

# Generate the energies using geomspace
energies = np.geomspace(10, 1e6, 500)

# Print each energy value
for i, energy in enumerate(energies):
    print(f"Energy {i}: {energy:.10e}")

