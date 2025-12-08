import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

CSV_PATH = "../results/metrics.csv"

df = pd.read_csv(CSV_PATH)

# Make kernel order consistent
kernel_order = ["Gaussian", "Motion", "Box", "RandomSeed", "KeyImage"]
df["kernel"] = pd.Categorical(df["kernel"], categories=kernel_order, ordered=True)

# ---------------------------------------------------------
# 1. PSNR per kernel per image (line plot)
# ---------------------------------------------------------

plt.figure(figsize=(10,6))
for img in df["image"].unique():
    sub = df[(df["image"] == img) & (df["correct_key"] == True)]
    plt.plot(sub["kernel"], sub["psnr"], marker="o", label=img)

plt.title("PSNR of Correct-Key Recovery for Each Kernel")
plt.xlabel("Kernel")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../results/plots/psnr_per_kernel.png")
plt.close()

# ---------------------------------------------------------
# 2. Correct vs Wrong Key (bar chart)
# ---------------------------------------------------------

df_wrong = df[df["correct_key"].isin([True, False]) & df["kernel"].isin(["RandomSeed", "KeyImage"])].copy()
df_wrong["kernel"] = pd.Categorical(
    df_wrong["kernel"], 
    categories=["RandomSeed", "KeyImage"], 
    ordered=True
)

plt.figure(figsize=(10,6))
sns.barplot(
    data=df_wrong,
    x="kernel",
    y="psnr",
    hue="correct_key",
    errorbar=None
)

plt.title("Correct vs Wrong Key: PSNR Comparison")
plt.xlabel("Kernel Type")
plt.ylabel("PSNR (dB)")
plt.xticks(rotation=0)
plt.legend(title="Correct Key?")
plt.tight_layout()
plt.savefig("../results/plots/correct_vs_wrong.png")
plt.close()

# ---------------------------------------------------------
# 3. MSE Heatmap (image × kernel)
# ---------------------------------------------------------

pivot = df[df["correct_key"] == True].pivot(index="image", columns="kernel", values="mse")

plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, cmap="magma_r", fmt=".3f")
plt.title("MSE Heatmap (Correct-Key Recovery)")
plt.xlabel("Kernel")
plt.ylabel("Image")
plt.tight_layout()
plt.savefig("../results/plots/mse_heatmap.png")
plt.close()