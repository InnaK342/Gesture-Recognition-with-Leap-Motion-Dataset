import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st

# Function to predict gesture
def predict(image):
    transformed_image = preprocess_image(image)
    prediction = model(transformed_image)
    _, result = torch.max(prediction, 1)
    return CATEGORIES[result][3:]

# Function to preprocess the uploaded image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize image to 64x64
        transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
        transforms.ToTensor(),  # Convert image to tensor
    ])
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension
    return transformed_image

# Define the model architecture
class GestModel(nn.Module):
    def __init__(self, num_classes):
        super(GestModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

# List of gesture categories
CATEGORIES = ["01_palm", '02_l','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_c','10_down']

# Load the trained model
model = GestModel(num_classes=10)
model.load_state_dict(torch.load('gest_model.pth'))
model.eval()

# Streamlit app title and file uploader
st.title('Gesture Recognition')
uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

# If image is uploaded
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction = predict(image)  # Get prediction
    st.write(f"Prediction: {prediction}")  # Display prediction
