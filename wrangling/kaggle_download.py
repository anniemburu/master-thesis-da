import kagglehub

# Download latest version
path = kagglehub.dataset_download("annemburu/categorical-datasets-for-regression-problem")

print("Path to dataset files:", path)

#!mkdir -p ~/.kaggle
#!mv kaggle.json ~/.kaggle/
#!chmod 600 ~/.kaggle/kaggle.json

#!kaggle datasets download annemburu/categorical-datasets-for-regression-problem
#!unzip /content/mammals-image-classification-dataset-45-animals.zip -d /home/mburu/Master_Thesis/master-thesis-da
