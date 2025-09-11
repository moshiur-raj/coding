import os
import jax.numpy as jnp

folder = "./results/comparing-nqs/"
l = 6

for filename in os.listdir(folder):
    if filename.endswith(".npz"):
        filepath = os.path.join(folder, filename)
        data = jnp.load(filepath)
        
        if "Energy" in data:
            energy = data["Energy"][-50:].real.mean() / 4 / (l*l)
        else:
            energy = data["energy"][-50:].real.mean() / 4 / (l*l)
        
        print(f"{filename}: energy={float(energy):.6f}")
