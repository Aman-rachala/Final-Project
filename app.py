import os
import sys
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template, Response
from utilities import transform, filter_bboxes_from_outputs, get_image_region, plot_results
from transformers import (
    DetrForObjectDetection,
    DPTForDepthEstimation,
    DPTFeatureExtractor,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)
from PIL import Image

# Initializing a vision encoder-decoder model for image captioning
ve_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initializing a feature extractor for the Vision Transformer (ViT)
# This will be used to process images and extract relevant features
feature_extractor_vit = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Initializing a tokenizer for text input
# This will convert raw text into tokenized input that the model can understand
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

app = Flask(__name__)
detrmodel = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
dptmodel = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
detrmodel.eval()
dptmodel.eval()


# Home route
@app.route("/")
def home():
    return render_template("index2.html")


@app.route("/upload")
def photo():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict(priority_list=[]):
    # priority_list = []
    file = request.files["data"]
    print("No.of files: ", len(request.files))
    print("priority_list: ",priority_list)
    # if os.path.exists("static/uploaded_image.jpg"):
    #     os.remove("static/uploaded_image.jpg")  # Delete the file
    #     print("Deletion successful!")
    file.save("static/uploaded_image.jpg")
    # Read the image via file.stream
    im = Image.open(file.stream)
    img = transform(im).unsqueeze(0)

    height, width = img.shape[-2], img.shape[-1]
    print("Height:", height)
    print("Width:", width)
    print("Image: ", img)

    outputs = detrmodel(img) 
    print("Outputs: ", outputs)   
    probas_to_keep, bboxes_scaled, final_outputs = filter_bboxes_from_outputs(im, outputs, threshold=0.95)
    plot_results(im, probas_to_keep, bboxes_scaled)
    print('final_outputs: ',final_outputs)
    
    object_list = [[int(item) if isinstance(item, float) else item for item in sublist] for sublist in final_outputs]
    final_outputs = []
    print('final_outputs: ',final_outputs)
    print("Objects_list: ", object_list)


    pixel_values = feature_extractor(im, return_tensors="pt").pixel_values
    with torch.no_grad():
        outputs = dptmodel(pixel_values)
        predicted_depth = outputs.predicted_depth
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=im.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    # cv2.imread('/content/Depth Image', cv2.IMREAD_GRAYSCALE)
    depth_image = formatted
    # Perform object detection or segmentation on the normal image to obtain object bounding boxes and names
    # and store them in a list called 'objects' with elements of the form: (object_name, x, y, w, h)
    objects = object_list
    print("Objects: ",objects)
    # Calculate average depth value for each object
    
    objects_with_depth = []
    print("objects with depth: ",objects_with_depth)
    for idx, obj in enumerate(objects):
        (
            object_name,
            x,
            y,
            w,
            h,
            origin,
        ) = obj  # Object bounding box coordinates and name
        depth_values = depth_image[y : y + h, x : x + w]
        average_depth = np.mean(depth_values)
        objects_with_depth.append(([object_name, idx, origin], average_depth))

    # Sort objects based on depth (lower depth values have higher priority)
    objects_with_depth.sort(key=lambda x: x[1])
    print("objects with depth: ",objects_with_depth)
    # Create a priority list of objects with object names and initial indices
    priority_list = [obj[0] for obj in objects_with_depth]
    print("Priority list:",priority_list)
    priority_list = sorted(priority_list, key=lambda x: x[1])
    print("Priority list:",priority_list)
    print("-" * 20)
    print(f"The priority list is: \n{priority_list}")
    print("-" * 20)

    height, width = img.shape[-2], img.shape[-1]
    print("Height:", height)
    print("Width:", width)
    print("Image: ", img)

    # regions = {
    #     "Top-Left": (-width // 3, height // 3, -width // 6, height // 2),
    #     "Top-Center": (-width // 6, height // 3, width // 6, height // 2),
    #     "Top-Right": (width // 6, height // 3, width // 3, height // 2),
    #     "Middle-Left": (-width // 3, -height // 6, -width // 6, height // 3),
    #     "Middle-Center": (-width // 6, -height // 6, width // 6, height // 3),
    #     "Middle-Right": (width // 6, -height // 6, width // 3, height // 3),
    #     "Bottom-Left": (-width // 3, -height // 3, -width // 6, -height // 6),
    #     "Bottom-Center": (-width // 6, -height // 3, width // 6, -height // 6),
    #     "Bottom-Right": (width // 6, -height // 3, width // 3, -height // 6),
    # }

    regions = {
        "Top-Left": (-960, 180, -280, 920),
        "front but at a high level": (-280, 180, 280, 920),
        "Top-Right": (280, 180, 960, 920),
        "Left": (-960, -180, -280, 180),
        "Front": (-280, -180, 280, 180),
        "Right": (280, -180, 960, 180),
        "Bottom-Left": (-960, -920, -280, -180),
        "Front but at Low Level": (-280, -920, 280, -180),
        "Bottom-Right": (280, -920, 960, -180),
    }


    for i in priority_list:
        center = i[2]
        i.insert(3, get_image_region(center, height, width, regions))

    

    grp = {}
    for k in regions.keys():
        grp[k] = []
    l = len(priority_list)
    delim = l if l < 5 else 5
    for i in priority_list[:delim]:
        grp[i[3]].append(i[0])

    answer = "The most Priority objects are :"
    for k, v in grp.items():
        text = ""
        if len(v):
            for i in v:
                text += i
                text += " "
            text += " at " + k
            answer += text + ", "
    print(answer)

    # if os.path.exists("static/uploaded_image.jpg"):
    #     os.remove("static/uploaded_image.jpg")  # Delete the file
    #     print("Deletion successful!")
    # if os.path.exists("static/images/image.jpg"):
    #     os.remove("static/images/image.jpg")  # Delete the file
    #     print("Deletion successful!")
        # print(f"File {static/uploaded_image.jpg} deleted successfully.")

    return Response(answer)

@app.route('/restart', methods=['POST'])
def restart_server():
    # Optional: Perform any cleanup or save data before restarting
    os.execl(sys.executable, sys.executable, *sys.argv)

app.run(debug=True)
