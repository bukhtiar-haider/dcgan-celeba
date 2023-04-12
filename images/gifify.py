import os
from PIL import Image

# Set the directory containing the jpg files
dir_path = os.getcwd()

# Get a list of all the jpg files in the directory
jpg_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.jpg')]

# Sort the list of jpg files in ascending order
jpg_files.sort()

# Set the name of the output gif file
gif_name = 'output.gif'

# Open the first jpg file to get the size
with Image.open(jpg_files[0]) as im:
    width, height = im.size

# Create a new image object for the output gif
out = Image.new('RGB', (width, height))

# Create a list to store the frames
frames = []

# Loop through the sorted list of jpg files and add each as a frame
for jpg in jpg_files:
    with Image.open(jpg) as im:
        frames.append(im.convert('RGB'))

# Save the frames as a gif
frames[0].save(gif_name, save_all=True, append_images=frames[1:], duration=100, loop=0)
