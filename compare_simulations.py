import os

from matplotlib import pyplot as plt
from PIL import Image

# folders = [
#     "simulation_comparisons/true",
#     "simulation_comparisons/ann",
#     "simulation_comparisons/rnn",
#     "simulation_comparisons/cnn",
# ]

# row_labels = ["Verdade", "ANN", "RNN", "CNN"]
# col_labels = [
#     "t=0",
#     "t=1",
#     "t=10",
#     "t=20",
#     "t=30",
#     "t=40",
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

print(len(axes))
for i in range(num_cols):
    axes[i].imshow(images[0][i])
    axes[i].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axes[i].set_aspect("equal")

# for j, row_label in enumerate(row_labels):
#     axes[j, 0].set_ylabel(row_label, size="large")

for j, col_label in enumerate(col_labels):
    axes[j].set_title(col_label)


plt.tight_layout()

plt.show()
