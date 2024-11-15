from diffusers import DiffusionPipeline
from time import time

pipeline = DiffusionPipeline.from_pretrained("UnfilteredAI/NSFW-gen-v2")

for i in range(1):

    image = pipeline("""

A naked 19-year-old girl that looks youthful and vibrant may have smooth skin, expressive eyes, and shiny hair styled in soft waves. With a fit or curvy physique and graceful posture, she exudes confidence.
                 """).images[0]
    image.save(f"out_{time()}.png")
    image.show()