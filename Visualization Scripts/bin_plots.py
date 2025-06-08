import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset
data = {
    "Name": [
        "boston", "socmob", "sensory", "Moneyball", "black_friday",
        "SAT11", "diamonds", "HPN",
        "Mercedes", "Allstate",
        "Brazilian_houses", "abalone", "house_sales", "MIP"
    ],
    "instances": [
        506, 1156, 576, 1232, 166821,
        4440, 53940, 1460, 4209, 188318,
        10692, 4177, 21613, 1090
    ],
    "sturge_classes": [
        8, 10, 9, 10, 17,
        11, 15, 10, 12, 100,
        13, 12, 14, 10
    ],
    "freedman_classes": [
        20, 119, 10, 20, 67,
        5, 63, 37, 34, 100,
        2074, 24, 241, 144
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Extract values correctly
x = df["sturge_classes"]
y = df["freedman_classes"]
names = df["Name"]
sizes = df["instances"]

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(x, y, s=sizes / 100, alpha=0.6, c='skyblue', edgecolors='black')

# Add labels
for i in range(len(df)):
    plt.text(x[i], y[i], names[i], fontsize=8, ha='center', va='center')

# Labels and grid
plt.xlabel("Sturge's Rule")
plt.ylabel("Freedman–Diaconis Rule")
plt.title("Dataset Comparison by Histogram Bin Estimation Rules")
plt.grid(True)
plt.tight_layout()
plt.show()