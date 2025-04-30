import os

folder = "datasets"
files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
print(files)