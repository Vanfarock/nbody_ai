import os

from matplotlib import pyplot as plt
from PIL import Image

plt.rcParams.update({"font.size": 14})

# folders = [
#     "simulation_comparisons/true",
#     "simulation_comparisons/ann",
#     "simulation_comparisons/rnn",
#     "simulation_comparisons/cnn",
# ]

# row_labels = ["Verdade", "RNA", "RNN", "CNN"]
# col_labels = [
#     "t=0",
#     "t=15",
#     "t=30",
#     "t=45",
#     "t=60",
#     "t=75",
# ]

folders = [
    "simulation_comparisons/visualization",
]

row_labels = []
col_labels = [
    "t=0",
    "t=10",
    "t=20",
    "t=30",
]

images = []
for folder in folders:
    image_files = [
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(("jpg"))
    ]
    images.append([Image.open(img_file) for img_file in sorted(image_files)])

num_rows = len(images)
num_cols = len(images[0])
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 6))
plt.subplots_adjust(hspace=-0.2, wspace=0.1)

print("-------------------------------------------------")

# for i in range(num_rows):
#     for j in range(num_cols):
#         axes[i, j].imshow(images[i][j])
#         axes[i, j].tick_params(
#             left=False, bottom=False, labelleft=False, labelbottom=False
#         )
#         axes[i, j].set_aspect("equal")

# for j, row_label in enumerate(row_labels):
#     axes[j, 0].set_ylabel(row_label, size="large")

# for j, col_label in enumerate(col_labels):
#     axes[0, j].set_title(col_label)

print("--------------------------------------------------")

for i in range(num_cols):
    axes[i].imshow(images[0][i])
    axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[i].set_aspect("equal")

for j, row_label in enumerate(row_labels):
    axes[j, 0].set_ylabel(row_label, size="large")

for j, col_label in enumerate(col_labels):
    axes[j].set_title(col_label)

# arrow_ax = fig.add_axes([0.1, 0.05, 0.8, 0.1], frameon=False)
# arrow_ax.set_xlim(0, 1)
# arrow_ax.set_ylim(0, 1)
# arrow_ax.axis("off")
# arrow = patches.FancyArrowPatch(
#     (0, 0.5),
#     (1, 0.5),
#     mutation_scale=50,
#     lw=10,
#     color="#1182E6",
#     arrowstyle="-|>",
# )
# arrow_ax.add_patch(arrow)
# arrow_ax.text(
#     0.5,
#     0.9,
#     "Probabilidade",
#     ha="center",
#     va="center",
#     color="#1182E6",
#     fontsize=20,
# )


plt.tight_layout()

plt.show()
