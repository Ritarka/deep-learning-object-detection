from PIL import Image, ImageDraw
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class bbox:
    x: int
    y: int
    w: int
    h: int
    
@dataclass
class anchor:
    x: int
    y: int

train_path = Path("widerface_homework/train")
val_path = Path("widerface_homework/val")


num_faces = 0
curr_image = None
image_height = None
image_width = None

num_images = -1
boxes_per_image = []
paths = []
anchors = []

with open(str(train_path / "label.txt"), "r") as f:
    for line in f:
        line = line.strip()
        arr = line.split()
        
        if line[0] == "#":
            num_images += 1
            if num_images > 3:
                break
            sub_path = arr[1]
            curr_image = train_path / "images" / sub_path
            image_width, image_height = Image.open(curr_image).size
            boxes_per_image.append([])
            anchors.append([])
            paths.append(curr_image)
            continue
        
        num_faces += 1
        
        # Extract the bounding box
        x, y, w, h = map(int, arr[:4])
        area = w * h
        
        if x < 0 or y < 0 or x + w > image_width or y + h > image_height:            
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + w > image_width:
                w = image_width - x
            if y + h > image_height:
                h = image_height - y
                
        boxes_per_image[num_images].append(bbox(x, y, w, h))
        anchors[num_images].append(anchor(arr[4],  arr[5]))
        anchors[num_images].append(anchor(arr[7],  arr[8]))
        anchors[num_images].append(anchor(arr[10],  arr[11]))
        anchors[num_images].append(anchor(arr[13], arr[14]))
        anchors[num_images].append(anchor(arr[16], arr[17]))
        

assert len(boxes_per_image) == 4
assert len(paths) == 4
assert len(anchors) == 4

images = []
for i in range(4):
    image = Image.open(paths[i])
    d = ImageDraw.Draw(image)
    for box in boxes_per_image[i]:
        d.rectangle([box.x, box.y, box.x + box.w, box.y + box.h], width=5, outline="green")
    points = [(float(anch.x), float(anch.y)) for anch in anchors[i]]
    for point in points:
        d.circle(point, radius = 3, width=2, fill="red")
    # images.append(image)
    plt.imshow(image)
    
# fig, axs = plt.subplots(2, 2)
# axs[0,0].imshow(images[0])
# axs[0,1].imshow(images[1])
# axs[1,0].imshow(images[2])
# axs[1,1].imshow(images[3])

    plt.savefig(f"Q1_2_{i}.png")