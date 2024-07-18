import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import urllib.request
import ollama
import base64
import os
from datetime import datetime
from tqdm import trange

app = Flask(__name__)

# Load ImageNet class names
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
urllib.request.urlretrieve(url, "imagenet_classes.txt")

STYLE_IMG_PATH = os.path.join('imgs', 'candinsky.jpg')

with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Load a pre-trained model
model = models.resnet18(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

stories = []

def classify_image(image):
    image = preprocess(image)       # 이미지 전처리
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return labels[predicted.item()]

def generate_new_story(image_label):
    story_prompt = f"Tell a story about an image containing {image_label}."
    response = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': story_prompt}])
    return response['message']['content']

def generate_continue_story(image_label, previous_story):
    story_prompt = f"Tell a story that continues from the {previous_story} and incorporates this image depicting {image_label}."
    response = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': story_prompt}])
    return response['message']['content']

def summary_story(previous_story):
    story_prompt = f"Summarize this story : {previous_story}."
    response = ollama.chat(model='llama3:latest', messages=[{'role': 'user', 'content': story_prompt}])
    return response['message']['content']

class StoryGANGenerator(nn.Module):
    def __init__(self):
        super(StoryGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def run_gan_model(story):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = StoryGANGenerator().to(device)
    noise = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()
    fake_image = transforms.ToPILImage()(fake_image.squeeze(0))
    return fake_image

def generate_image_with_gan(story):
    generated_image = run_gan_model(story)
    generated_image_base64 = image_to_base64(generated_image)
    return generated_image_base64

def uploaded_user_image(input_image):
    uploaded_image = input_image
    uploaded_image_base64 = image_to_base64(uploaded_image)
    return uploaded_image_base64

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return render_template("upload_story.html", stories=stories)

STYLE_LOSS_WEIGHT = 1e-05
IMG_SIZE = 400
OUTPUT_DIR_PATH = os.path.join('static', 'style_transfer_output', '%s' % datetime.now().strftime('%Y%m%d_%H%M%S'))
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

iT = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

class Extractor(nn.Module):
    features = None

    def __init__(self, layer):
        super().__init__()
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

def get_layers(model, layer_indices):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    selected_layers = [layers[i] for i in layer_indices]
    return selected_layers

def style_transfer(content_image, style_image, model):
    content_tensor = T(content_image).unsqueeze(0).to(device)
    style_tensor = T(style_image).unsqueeze(0).to(device)

    content_layers = get_layers(model, [0])
    content_exts = [Extractor(layer) for layer in content_layers]
    model(content_tensor)
    content_features = [ext.features.clone() for ext in content_exts]

    style_layers = get_layers(model, [0, 1, 2, 3])
    style_exts = [Extractor(layer) for layer in style_layers]
    model(style_tensor)
    style_features = [ext.features.clone() for ext in style_exts]

    input_tensor = content_tensor.clone().requires_grad_()
    optimizer = torch.optim.SGD([input_tensor], lr=100.)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

    def content_loss(y, y_hat):
        loss = 0
        for i in range(len(y_hat)):
            loss += F.mse_loss(y[i], y_hat[i])
        return loss / len(y_hat)

    def gram_matrix(x):
        b, c, h, w = x.size()
        x = x.view(b * c, -1)
        return torch.mm(x, x.t())

    def style_loss(y, y_hat):
        loss = 0
        for i in range(len(y_hat)):
            y_gram = gram_matrix(y[i])
            y_hat_gram = gram_matrix(y_hat[i])
            loss += F.mse_loss(y_gram, y_hat_gram)
        return loss / len(y_hat)

    pbar = trange(500 + 1)
    for i in pbar:
        model(input_tensor)

        current_content_features = [ext.features.clone() for ext in content_exts]
        current_style_features = [ext.features.clone() for ext in style_exts]

        c_loss = content_loss(content_features, current_content_features)
        s_loss = style_loss(style_features, current_style_features) * STYLE_LOSS_WEIGHT
        loss = c_loss + s_loss
        pbar.set_description('Content loss %.6f Style loss %.6f' % (c_loss, s_loss))

        optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=True 추가
        optimizer.step()

        if i % 500 == 0:
            output_img = iT(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255
            Image.fromarray(output_img.astype('uint8')).save(os.path.join(OUTPUT_DIR_PATH, '%s.png' % i))
            scheduler.step()

    final_img = iT(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255
    final_img = Image.fromarray(final_img.astype('uint8'))
    return final_img

@app.route("/upload_story", methods=["POST"])
def upload_story():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image selected!"}), 400

        image_file = request.files['image']
        if not image_file.filename.lower().endswith(('.png', '.jpg', '.gif', '.jpeg')):
            return jsonify({"error": "Invalid image format!"}), 400

        content_image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        style_image = Image.open(STYLE_IMG_PATH).convert('RGB')

        styled_image = style_transfer(content_image, style_image, model)
        styled_image_base64 = image_to_base64(styled_image)

        label = classify_image(content_image)

        story = generate_new_story(label) if len(stories) == 0 else generate_continue_story(label, [item['story'] for item in stories])
        stories.append({'story': story, 'label': label, 'styled_image': styled_image_base64})

        return render_template("upload_story.html", stories=stories)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

