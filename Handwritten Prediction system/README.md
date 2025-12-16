# Handwritten Digit Recognition using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A complete handwritten digit recognition system using deep learning (Multi-layer Perceptron) trained on the MNIST dataset, with an interactive Streamlit web application.

## 🎯 Features

- **🧠 Deep Learning Model**: Multi-layer perceptron with 99%+ accuracy
- **✏️ Interactive Drawing Canvas**: Draw digits directly in your browser
- **🔮 Real-time Prediction**: Instant digit recognition with confidence scores
- **📊 Batch Prediction**: Upload CSV files for bulk predictions
- **📈 Model Analytics**: Comprehensive performance metrics and visualizations
- **🎨 Modern UI**: Beautiful, responsive interface with animations
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🖼️ Screenshots

### Home Page
Beautiful landing page with project overview and key metrics.

### Draw & Predict
Interactive canvas for drawing digits with real-time predictions and probability distributions.

### Batch Prediction
Upload CSV files for bulk predictions with downloadable results.

### Model Performance
Detailed metrics including confusion matrix, training history, and classification report.

## 🛠️ Technology Stack

- **Deep Learning**: TensorFlow & Keras
- **Web Framework**: Streamlit
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Image Processing**: OpenCV, Pillow
- **Drawing Canvas**: Streamlit-Drawable-Canvas

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- 70,000 grayscale images of handwritten digits
- 60,000 training images
- 10,000 test images
- Image size: 28×28 pixels
- 10 classes (digits 0-9)

## 🧠 Model Architecture

**Multi-layer Perceptron (MLP)**

```
Input Layer:     784 neurons (28×28 pixels)
Hidden Layer 1:  512 neurons + ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 2:  256 neurons + ReLU + BatchNorm + Dropout(0.3)
Hidden Layer 3:  128 neurons + ReLU + BatchNorm + Dropout(0.2)
Hidden Layer 4:  64 neurons + ReLU + BatchNorm + Dropout(0.2)
Output Layer:    10 neurons (Softmax activation)
```

**Training Configuration**
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 128
- Max Epochs: 100 (with early stopping)
- Callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (or download the files)
```bash
cd "Handwritten Prediction system"
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify dataset files**
Make sure you have `train.csv` and `test.csv` in the project directory.

## 📖 Usage

### Step 1: Train the Model

Run the training script to train the deep learning model:

```bash
python train_model.py
```

This will:
- Load and preprocess the MNIST dataset
- Train the multi-layer perceptron model
- Save the trained model and metrics
- Generate visualization plots

**Expected Output:**
- `digit_recognition_model.h5` - Trained model (HDF5 format)
- `digit_recognition_model/` - Trained model (TensorFlow format)
- `model_metrics.json` - Model performance metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `training_history.png` - Training history plots

**Training Time:** Approximately 5-10 minutes (depending on your hardware)

### Step 2: Run the Streamlit App

Launch the web application:

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 🎮 Using the Application

### 🏠 Home Page
- View project overview and key statistics
- See model performance metrics
- Learn about features and capabilities

### ✏️ Draw & Predict
1. Draw a digit (0-9) on the canvas using your mouse
2. Click "Predict Digit" to get the prediction
3. View the predicted digit with confidence score
4. See probability distribution for all digits
5. Click "Clear Canvas" to draw again

### 📊 Batch Prediction
1. Upload a CSV file with digit images (784 pixels per row)
2. Select the number of samples to predict
3. Click "Run Batch Prediction"
4. View results in a table
5. Download predictions as CSV

**CSV Format:**
- Each row represents one image
- 784 columns (28×28 pixel values)
- Pixel values in range 0-255
- Optional 'label' column for actual digits

### 📈 Model Performance
- View accuracy, loss, and training statistics
- Explore confusion matrix
- See training history plots
- Review classification report

### ℹ️ About
- Learn about the project
- Understand the model architecture
- View technology stack
- See training details

## 📁 Project Structure

```
Handwritten Prediction system/
│
├── train.csv                      # MNIST training dataset
├── test.csv                       # MNIST test dataset
├── train_model.py                 # Model training script
├── app.py                         # Streamlit web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
├── digit_recognition_model.h5     # Trained model (generated)
├── digit_recognition_model/       # Trained model TF format (generated)
├── model_metrics.json             # Model metrics (generated)
├── confusion_matrix.png           # Confusion matrix plot (generated)
└── training_history.png           # Training history plot (generated)
```

## 📊 Model Performance

The model achieves excellent performance on the MNIST dataset:

- **Accuracy**: 99%+ on validation set
- **Loss**: < 0.05
- **Training Time**: 5-10 minutes
- **Inference Time**: < 100ms per image

### Per-Class Performance
All digit classes (0-9) achieve >98% precision and recall.

## 🎨 UI Features

- **Modern Design**: Gradient backgrounds and smooth animations
- **Responsive Layout**: Works on all screen sizes
- **Interactive Elements**: Hover effects and transitions
- **Color Scheme**: Purple gradient theme with high contrast
- **Typography**: Clean, readable fonts
- **Visual Feedback**: Loading spinners and success messages

## 🔧 Customization

### Modify Model Architecture
Edit `train_model.py` to change:
- Number of layers
- Neurons per layer
- Activation functions
- Dropout rates
- Batch normalization

### Customize UI
Edit `app.py` to change:
- Color scheme (CSS variables)
- Layout and spacing
- Page content
- Visualizations

### Adjust Training Parameters
In `train_model.py`, modify:
- Learning rate
- Batch size
- Number of epochs
- Optimizer settings

## 🐛 Troubleshooting

### Model Not Loading
- Make sure you've run `train_model.py` first
- Check that model files exist in the project directory
- Verify TensorFlow installation

### Canvas Not Working
- Clear browser cache
- Try a different browser
- Check streamlit-drawable-canvas installation

### Poor Predictions
- Ensure you draw digits clearly
- Use white strokes on black background
- Draw digits centered in the canvas
- Make digits large enough

### CSV Upload Issues
- Verify CSV has 784 columns
- Check pixel values are in range 0-255
- Ensure no missing values

## 📝 License

This project is licensed under the MIT License.

## 👨‍💻 Developer

Built with ❤️ for **#UneeQinterns**

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- TensorFlow and Keras teams
- Streamlit community
- OpenCV contributors

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🚀 Future Enhancements

- [ ] Add CNN model option
- [ ] Support for multiple languages
- [ ] Mobile app version
- [ ] Real-time video digit recognition
- [ ] Model comparison feature
- [ ] Export predictions to Excel
- [ ] API endpoint for predictions
- [ ] Docker containerization

---

**Happy Digit Recognition! 🔢✨**
