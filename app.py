from flask import Flask, render_template, request
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import io
import base64

app = Flask(__name__)

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return render_template('index.html', error='No image uploaded')
        
        file = request.files['image']
        
        # Check if the file has a valid extension
        if file.filename == '':
            return render_template('index.html', error='No image selected')
        
        if file and allowed_file(file.filename):
            # Save the uploaded image
            file_path = 'uploads/' + file.filename
            file.save(file_path)

            # Generate captions using the uploaded image
            captions = predict_step([file_path])
            
            # Display the image using PIL
            image = Image.open(file_path)
            image_data = io.BytesIO()
            image.save(image_data, format='PNG')
            image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

            return render_template('index.html', image=image_base64, captions=captions)
        else:
            return render_template('index.html', error='Invalid file type')

    return render_template('index.html')

def allowed_file(filename):
    # Add the allowed image file extensions here
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, num_return_sequences=3, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

if __name__ == '__main__':
    app.run(debug=True)