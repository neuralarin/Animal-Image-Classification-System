import streamlit as st
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Animal Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Animal Image Classifier")
st.write("Upload an animal image and the model will predict what it is.")

# -----------------------------
# Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Model Architecture
# -----------------------------
class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------
# Load model
# -----------------------------
model = ClassificationModel().to(device)

model.load_state_dict(
    torch.load(
        "model/animal_classifier.pth",
        map_location=device
    )
)

model.eval()


# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        # Preprocess image
        input_image = transform(image)
        input_image = input_image.unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(input_image)
            probabilities = torch.softmax(output, dim=1)

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            confidence = probabilities[0][predicted_class].item()

        # Your Animal Faces dataset classes
        class_names = ["cat", "dog", "wild"]

        prediction = class_names[predicted_class]

        st.success(f"Prediction: **{prediction.upper()}**")

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )