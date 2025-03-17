# Food-Image-Prediction

## Features

- **Image Classification**: Predicts food type from an uploaded image (e.g., "pizza," "sushi").
- **Confidence Scores**: Displays prediction confidence and top-5 probable classes.
- **User-Friendly UI**: Built with Streamlit for easy interaction.
- **Pre-trained Model**: Uses ResNet50 fine-tuned on a custom food dataset.
- **CPU/GPU Support**: Auto-detects and runs on available hardware.

## Dataset

The model was trained on a dataset located at `./Food_classifier`, containing 4,775 images across 34 food categories:
- Categories include: `Baked Potato`, `Crispy Chicken`, `Donut`, `Fries`, `Hot Dog`, `Sandwich`, `Taco`, `Taquito`, `apple_pie`, `burger`, `butter_naan`, `chai`, `chapati`, `cheesecake`, `chicken_curry`, `chole_bhature`, `dal_makhani`, `dhokla`, `fried_rice`, `ice_cream`, `idli`, `jalebi`, `kaathi_rolls`, `kadai_paneer`, `kulfi`, `masala_dosa`, `momos`, `omelette`, `paani_puri`, `pakode`, `pav_bhaji`, `pizza`, `samosa`, `sushi`.
- Split: 75% training (3,581 images), 25% testing (1,194 images).

## Architecture

### Model Architecture
The classifier uses a modified ResNet50 architecture from `torchvision.models`:

Input Image (224x224x3)
├── Preprocessing (Resize, Normalize: mean=[0.438, 0.456, 0.406], std=[0.229, 0.224, 0.225])
└── ResNet50
├── Frozen Layers (conv1 to layer3): Pre-trained weights from ImageNet
├── Trainable Layer (layer4): Fine-tuned for food features
└── Fully Connected Layer (fc)
├── Dropout (p=0.5)
└── Linear (2048 → 34 classes)
Output: Logits → Softmax → Probabilities (34 classes)


- **Input**: 224x224 RGB images.
- **Backbone**: ResNet50 with frozen layers up to `layer3` for transfer learning, unfrozen `layer4` for task-specific adaptation.
- **Output**: 34-class probability distribution via softmax.

### Training
- **Optimizer**: Adam (learning rate = 0.001).
- **Loss**: Cross-Entropy Loss.
- **Epochs**: 10 (achieved ~86% validation accuracy).
- **Batch Size**: 32.

### App Architecture
Streamlit App
├── Frontend: Streamlit UI (file uploader, image display, prediction results)
├── Backend: PyTorch model inference
│   ├── Model Loading: Cached ResNet50 from .pth file
│   ├── Preprocessing: Transform pipeline
│   └── Prediction: Forward pass → Softmax → Top-5 results
└── File: food_classifier_app.py


---

###  Usage
```
## Usage

1. **Launch the App**: Run the command above.
2. **Upload an Image**: Select a `.jpg`, `.jpeg`, or `.png` file of a food item.
3. **View Results**: See the predicted food class, confidence score, and top-5 predictions.
```

### Example
- **Input**: Image of a pizza.
- **Output**:
Predicted Food: pizza
Confidence: 92.34%
Top 5 Predictions:

1. pizza: 92.34%
2. sandwich: 3.12%
3. burger: 2.45%
4. fries: 1.08%
5. taco: 0.67%

## Requirements

Create a `requirements.txt` file with:
streamlit==1.31.1
torch==2.2.1
torchvision==0.17.1
Pillow==10.2.0
numpy==1.26.4


---

### Step 10: License
```markdown
## License

This project is licensed under the MIT License.

For questions or suggestions, reach out via GitHub Issues or email at [roshabel001@gmail.com].
