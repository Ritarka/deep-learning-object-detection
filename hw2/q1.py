from PIL import Image
from tqdm import tqdm
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
    val: int

train_path = Path("widerface_homework/train")
val_path = Path("widerface_homework/val")

combined_paths = [train_path, val_path]

### Q1.1
total_faces = 0

total_height = 0.0
total_width = 0.0
total_area = 0.0

x_vals = []
y_vals = []

outside_bounds = 0

overlapping_boxes = 0
boxes = []

def jaccard(box_a, box_b):
    xl_a, yt_a, xr_a, yb_a = box_a.x, box_a.y, box_a.x + box_a.w, box_a.y + box_a.h
    xl_b, yt_b, xr_b, yb_b = box_b.x, box_b.y, box_b.x + box_b.w, box_b.y + box_b.h
    
    xl = max(xl_a, xl_b)
    xr = min(xr_a, xr_b)
    yt = max(yt_a, yt_b)
    yb = min(yb_a, yb_b)
    
    if xr > xl and yb > yt:
        intersection = (xr - xl) * (yb - yt)
    else:
        intersection = 0
    
    area_a = (xr_a - xl_a) * (yb_a - yt_a)
    area_b = (xr_b - xl_b) * (yb_b - yt_b)
    
    union = area_a + area_b - intersection
    
    if union == 0:
        return 0
    IoU = intersection / union
    return IoU

def num_overlapping(boxes):
    length = len(boxes)
    inter = set()
    for i in range(length):
        if i in inter: continue
        a = boxes[i]
        for j in range(i+1, length):
            if j == i: continue
            b = boxes[j]
            iou = jaccard(a, b)

            if iou > 0:
                inter.add(i)
                inter.add(j)
                break
            
    return len(inter)

for path in combined_paths:
    num_faces = 0
    curr_image = None
    image_height = None
    image_width = None
    with open(str(path / "label.txt"), "r") as f:
        for line in f:
            line = line.strip()
            arr = line.split()

            if arr[0] == "#":

                # Since we are accepting a new image now,
                # we calculate the number of overlapping GTs
                # from the previous image
                overlapping_boxes += num_overlapping(boxes)
                boxes = []
                
                sub_path = arr[1]
                curr_image = path / "images" / sub_path
                image_width, image_height = Image.open(curr_image).size
                continue
            
            num_faces += 1
            
            # Extract the bounding box
            x, y, w, h = map(float, arr[:4])
            area = w * h
            
            if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
                outside_bounds += 1
                
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x + w > image_width:
                    w = image_width - x
                if y + h > image_height:
                    h = image_height - y
                    
            # assert w > 0
            # assert h > 0
                    
            boxes.append(bbox(x, y, w, h))
            
            total_area += area
            total_height += h
            total_width += w
            
            x_vals.append((x + w / 2)/image_width)
            y_vals.append((y + h / 2)/image_height)
            
    overlapping_boxes += num_overlapping(boxes)
    boxes = []

    print(f"Faces found in {str(path)}: {num_faces}")
    total_faces += num_faces
        
print(f"Found {total_faces} faces in total.")

avg_width = total_width / total_faces
avg_height = total_height / total_faces
avg_area = total_area / total_faces


### Q1.2
print(f"Average width: {avg_width:.2f}, Average height: {avg_height:.2f}, Average area: {avg_area:.2f}")


### Q1.3
plt.hist2d(x_vals, y_vals, bins=(50, 50), cmap='viridis')
plt.colorbar()

plt.xlabel('X') 
plt.ylabel('Y') 
plt.title('2-D Histogram') 

plt.plot()
plt.savefig("Q1_3.png")

''' 
Face distribution is not uniform. Most faces are located about 70% down the image 
and halfway across the image (the bottom middle). Please see the heatmat of face locations.
This was normalized by the image width and height to provide results for varying image sizes.
'''

### Q1.4
print(f"Number of datasets with bounds outside the image: {outside_bounds}")
# Strangely no images had bounds outside the image. I added in some code to move the bounding
# boxes inside the image anyways

### Q1.5
print(f"Got {overlapping_boxes} with {total_faces} making for {100.0*float(overlapping_boxes)/total_faces:.2f}% of GTs overlapping")


'''
Faces found in widerface_homework/train: 159424
Faces found in widerface_homework/val: 39708
Found 199132 faces in total.
Average width: 29.00, Average height: 37.44, Average area: 3847.96
Number of datasets with bounds outside the image: 0
Got 24275 with 199132 making for 12.19% of GTs overlapping
'''