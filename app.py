import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

# Define the model class (renamed to FoodClassifierResNet)
class FoodClassifierResNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers except layer4 and fc
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        # Replace the final fully connected layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Define class names from your dataset
class_names = [
    'Baked Potato', 'Crispy Chicken', 'Donut', 'Fries', 'Hot Dog', 'Sandwich', 'Taco', 
    'Taquito', 'apple_pie', 'burger', 'butter_naan', 'chai', 'chapati', 'cheesecake', 
    'chicken_curry', 'chole_bhature', 'dal_makhani', 'dhokla', 'fried_rice', 'ice_cream', 
    'idli', 'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos', 
    'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa', 'sushi'
]
num_classes = len(class_names)

# Define image transformations (matching your training setup)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.438, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to load the model
@st.cache_resource
def load_model(model_path="food_classifier_resnet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodClassifierResNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

# Function to preprocess and predict
def predict(image, model, device):
    # Convert PIL image to tensor
    img = transform(image).unsqueeze(0)  # Add batch dimension
    img = img.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100
    
    return predicted_class, confidence, probabilities

# Save the trained model (add this to your training code if not already done)
def save_model(model, path="food_classifier_resnet.pth"):
    torch.save(model.state_dict(), path)

# Streamlit app
def main():
    st.title("Food Image Classifier")
    st.write("Upload an image of food, and I'll classify it into one of 34 categories!")

    # Load the model
    try:
        model, device = load_model("food_classifier_resnet.pth")
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.error("Model file 'food_classifier_resnet.pth' not found. Please train and save the model first.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict
        with st.spinner("Classifying..."):
            predicted_class, confidence, probabilities = predict(image, model, device)
        
        # Display results
        st.write(f"**Predicted Food**: {predicted_class}")
        st.write(f"**Confidence**: {confidence:.2f}%")
        
        # Optional: Show top 5 predictions
        st.write("**Top 5 Predictions**:")
        top5_probs, top5_indices = torch.topk(probabilities, 5)
        for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
            st.write(f"{i+1}. {class_names[idx]}: {prob.item()*100:.2f}%")

# Training function (optional, for reference or first-time use)
def train_and_save():
    # This is a simplified version of your training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FoodClassifierResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Load your dataset (replace with your actual paths)
    dataset = datasets.ImageFolder('./Food_classifier', transform=transform)
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train for 1 epoch as a demo (adjust epochs as needed)
    model.train()
    for epoch in range(1):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        st.write(f"Epoch {epoch+1} completed")
    
    # Save the model
    save_model(model)
    st.success("Model trained and saved as 'food_classifier_resnet.pth'")

if __name__ == "__main__":
    # Uncomment the next line to train and save the model if not already done
    # train_and_save()
    
    # Run the app
    main()