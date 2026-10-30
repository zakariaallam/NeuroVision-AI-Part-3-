import gdown
import numpy as np
from pathlib import Path
import zipfile
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
from collections import Counter


file_id = "1JbFzlo4GTYtSu8oPsSFHv9FFzg-fGDgB"
zip_path = "data.zip"

url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url,zip_path,quiet=False)


data_dir = Path(zip_path)

if data_dir.is_file() and data_dir.suffix.lower() == ".zip":
        extract_dir = data_dir.with_suffix("")

with zipfile.ZipFile(data_dir,"r") as zip_file:
            zip_file.extractall(extract_dir)
            
            data_dir = extract_dir
            
            
   
def read_yolo_labels(label_path):
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            class_id, xc, yc, w, h = values

            annotations.append({
                "class_id": int(class_id),
                "x_center": float(xc),
                "y_center": float(yc),
                "width": float(w),
                "height": float(h)
            })

    return annotations

def yolo_to_pixels(annotation, image_width, image_height):
    xc = annotation["x_center"] * image_width
    yc = annotation["y_center"] * image_height

    w = annotation["width"] * image_width
    h = annotation["height"] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    return x1, y1, x2, y2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    p for p in data_dir.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    and p.parent.name == "images"
]

print("Number of images:", len(image_paths))
print("Examples:", image_paths[:5])


def get_label_path(image_path):
    return (
        image_path.parent.parent
        / "labels"
        / (image_path.stem + ".txt")
    )

class_names = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary"
}


valid_images = [
    p for p in image_paths
    if get_label_path(p).exists()
]

if not valid_images:
    print("No matching image-label pairs found. Check the folder structure.")
else:
    samples = random.sample(valid_images, min(9, len(valid_images)))

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for ax, image_path in zip(axes, samples):
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        ax.imshow(image)

        annotations = read_yolo_labels(get_label_path(image_path))

        for ann in annotations:
            x1, y1, x2, y2 = yolo_to_pixels(ann, W, H)

            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor="lime",
                facecolor="none"
            )

            ax.add_patch(rect)

            class_name = class_names.get(
                ann["class_id"],
                f"Class {ann['class_id']}"
            )

            ax.text(
                x1,
                max(0, y1 - 5),
                class_name,
                color="white",
                fontsize=9,
                bbox=dict(facecolor="green", alpha=0.8)
            )

        ax.set_title(image_path.name)
        ax.axis("off")

    for ax in axes[len(samples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    
    
    image_counts = Counter()
box_counts = Counter()

for image_path in image_paths:
    label_path = get_label_path(image_path)
    annotations = read_yolo_labels(label_path)

    classes_in_image = set()

    for ann in annotations:
        class_id = ann["class_id"]

        box_counts[class_id] += 1
        classes_in_image.add(class_id)

    for class_id in classes_in_image:
        image_counts[class_id] += 1

all_class_ids = sorted(set(image_counts) | set(box_counts))

distribution = pd.DataFrame([
    {
        "class_id": cid,
        "class_name": class_names.get(cid, f"Class {cid}"),
        "images": image_counts[cid],
        "bounding_boxes": box_counts[cid]
    }
    for cid in all_class_ids
])

display(distribution)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(
    distribution["class_name"],
    distribution["images"],
    color="steelblue"
)
axes[0].set_title("Number of images per class")
axes[0].set_ylabel("Images")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(
    distribution["class_name"],
    distribution["bounding_boxes"],
    color="darkorange"
)
axes[1].set_title("Number of bounding boxes per class")
axes[1].set_ylabel("Bounding boxes")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()



for column in ["images", "bounding_boxes"]:
    counts = distribution[column]

    if len(counts) == 0 or counts.max() == 0:
        print(f"No data available for {column}.")
        continue

    imbalance_ratio = counts.max() / counts[counts > 0].min()

    print(f"\n{column}:")
    print("Most represented class:", distribution.loc[counts.idxmax(), "class_name"])
    print("Imbalance ratio:", round(imbalance_ratio, 2))

    if imbalance_ratio >= 3:
        print("Possible class imbalance detected.")
    else:
        print("No strong imbalance detected using the 3:1 threshold.")            
            