import kagglehub

# Download latest version
path = kagglehub.dataset_download("annemburu/categorical-datasets-for-regression-problem")

print("Path to dataset files:", path)
