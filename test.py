import torch
from rocket_builder import build
from PIL import Image

# --- LOAD IMAGE ---
# Select the image you want to test the Object Detection Model with
image_path = '/Users/heiki/Development/playground/shop.jpg'
# image_path = 'images/shop.jpg'
# image_path = 'images/street.jpg'

img = Image.open(image_path)

# --- LOAD ROCKET ---
# Select the Rocket you want to test
rocket = "igor/retinanet"
# rocket = "igor/retinanet-resnet101-800px"
# rocket = "lucas/yolov3"

model = build().eval()

with torch.no_grad():
    img_tensor = model.preprocess(img)
    out = model(img_tensor)

# --- OUTPUT ---
# Print the output as a JSON
bboxes_out = model.postprocess(out, img)
print(*bboxes_out, sep='\n')

# Display the output over the image
img_out = model.postprocess(out, img, visualize=True)
img_out_path = '/Users/heiki/Development/playground/output.jpg'
img_out.save(img_out_path)