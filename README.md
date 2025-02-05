# Forest Cover Type Prediction Application

## Overview
GreenPredict is a Streamlit-based web application that predicts forest cover types using machine learning. The application takes various geographical and environmental features as input and predicts the type of forest cover present in that area, accompanied by representative images of the predicted forest type.

## Features
- Interactive web interface with sliders for input parameters
- Real-time predictions using a trained decision tree model
- Visual feedback with images of predicted forest types
- Responsive design with a forest-themed background
- Input validation and error handling
- Comprehensive feature input system covering:
  - Elevation and slope characteristics
  - Distance measurements (to hydrology, roadways, and fire points)
  - Hillshade measurements at different times (9am, noon, 3pm)
  - Wilderness area indicators
  - Soil type classifications

## Prerequisites
- Python 3.7 or higher
- Required Python packages:
  ```
  streamlit
  pandas
  numpy
  joblib
  Pillow
  ```

## Project Structure
```
project/
│
├── app.py                    # Main Streamlit application file
├── predict.py               # Prediction module
│
├── Model/
│   └── decision_tree_model.pkl  # Trained model file
│
├── scaler/
│   └── scaler.pkl           # Feature scaler file
│
├── data/
│   └── covtype.csv          # Dataset file
│
├── images/                  # Forest type images
│   ├── spruce_fir.jpg
│   ├── lodgepole_pine.jpg
│   ├── ponderosa_pine.jpg
│   ├── cottonwood_willow.jpg
│   ├── aspen.jpg
│   ├── douglas_fir.jpg
│   └── krummholz.jpg
│
└── background_image/
    └── forest2.jpg          # Background image for the app
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd forest-cover-prediction
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Adjust the input parameters using the sliders

4. Click the "Predict Forest Cover" button to get the prediction

## Forest Cover Types
The model predicts seven different forest cover types:
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

## Input Features
The application accepts various geographical and environmental features:

### Primary Features:
- Elevation (meters)
- Slope (degrees)
- Horizontal Distance to Hydrology (meters)
- Horizontal Distance to Roadways (meters)
- Horizontal Distance to Fire Points (meters)
- Hillshade indexes (9am, noon, 3pm)

### Categorical Features:
- Wilderness Areas (4 types)
- Soil Types (40 different types)

## Technical Details
- The prediction model is a decision tree classifier trained on the Forest Cover Type dataset
- Input features are scaled using a pre-trained scaler
- The application includes input validation to ensure values are within acceptable ranges
- Error handling is implemented for both prediction and image loading

## Error Handling
The application includes comprehensive error handling for:
- Invalid input values
- Model loading errors
- Image loading errors
- Prediction errors

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[Add your license information here]

## Acknowledgments
- Dataset source: [Add dataset source information]
- Images source: [Add image source information]
