import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import base64
import os

st.set_page_config(page_title="Forest Cover Type Prediction", page_icon="🌲", layout="wide")

@st.cache_resource
def load_models():
    model = joblib.load(r'Model\decision_tree_model.pkl')
    scaler = joblib.load(r'scaler\scaler.pkl')
    return model, scaler

# Load and analyze the dataset to get input ranges
@st.cache_data
def load_and_analyze_dataset():
    # Assuming the dataset is in the same directory
    df = pd.read_csv('data/covtype.csv')
    
    # Define the features we're using
    features = [
        'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type10', 'Soil_Type38',
        'Slope', 'Soil_Type39', 'Soil_Type40', 'Soil_Type17', 'Soil_Type3',
        'Soil_Type2', 'Soil_Type13', 'Wilderness_Area1',
        'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
        'Soil_Type29', 'Elevation', 'Hillshade_Noon', 'Soil_Type12',
        'Soil_Type23', 'Horizontal_Distance_To_Hydrology', 'Soil_Type20',
        'Hillshade_9am', 'Hillshade_3pm'
    ]
    
    # Calculate min and max for numeric features
    input_ranges = {}
    for feature in features:
        if feature in df.columns:
            input_ranges[feature] = {
                'min': int(df[feature].min()),
                'max': int(df[feature].max())
            }
    
    return input_ranges

model, scaler = load_models()
input_ranges = load_and_analyze_dataset()

features = [
    'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type10', 'Soil_Type38',
    'Slope', 'Soil_Type39', 'Soil_Type40', 'Soil_Type17', 'Soil_Type3',
    'Soil_Type2', 'Soil_Type13', 'Wilderness_Area1',
    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
    'Soil_Type29', 'Elevation', 'Hillshade_Noon', 'Soil_Type12',
    'Soil_Type23', 'Horizontal_Distance_To_Hydrology', 'Soil_Type20',
    'Hillshade_9am', 'Hillshade_3pm'
]

cover_type_mapping = {
    1: {"name": "Spruce/Fir", "image_path": "images/spruce_fir.jpg"},
    2: {"name": "Lodgepole Pine", "image_path": "images/lodgepole_pine.jpg"},
    3: {"name": "Ponderosa Pine", "image_path": "images/ponderosa_pine.jpg"},
    4: {"name": "Cottonwood/Willow", "image_path": "images/cottonwood_willow.jpg"},
    5: {"name": "Aspen", "image_path": "images/aspen.jpg"},
    6: {"name": "Douglas-fir", "image_path": "images/douglas_fir.jpg"},
    7: {"name": "Krummholz", "image_path": "images/krummholz.jpg"}
}

def validate_input(user_input):
    if any(value < 0 for value in user_input.values()):
        raise ValueError("All values must be non-negative")
    
    for feature, value in user_input.items():
        if feature in input_ranges:
            if value < input_ranges[feature]['min'] or value > input_ranges[feature]['max']:
                raise ValueError(f"{feature} value must be between {input_ranges[feature]['min']} and {input_ranges[feature]['max']}")

def predict_cover_type(user_input):
    try:
        validate_input(user_input)
        input_data = {feature: user_input.get(feature, 0) for feature in features}
        input_df = pd.DataFrame([input_data])
        input_df = input_df[features]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def add_custom_css():
    background_path = 'background_image/forest2.jpg'
    
    # Check if file exists
    if not os.path.exists(background_path):
        st.error(f"Background image not found at {background_path}")
        return
    
    with open(background_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(f"""
    <style>
    /* Remove page margin and padding */
    body {{
        margin: 0;
        padding: 0;
    }}

    /* Adjust Streamlit app container */
    .stApp {{
        background-image: url('data:image/jpg;base64,{encoded_string}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        padding: 0;
        margin: 0;
    }}

    /* Remove white space in title section */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        margin-top: 0;
        margin-bottom: 0;
    }}

    /* Input column styling */
    .input-column {{
        background-color: rgba(255, 255, 255, 0.6);
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }}

    /* Prediction column styling */
    .prediction-column {{
        background-color: rgba(255, 255, 255, 0.7);
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }}

    /* Ensure all text is readable */
    * {{
        color: black !important;
    }}

    /* Slider and input styles */
    .stSlider, .stNumberInput {{
        background-image: linear-gradient(to top right, #B2DFDB, #B3E5FC); /* Light teal to pale blue */
        border: 1px solid #FAD0C4; /* Light border to match */
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }}

    /* Prediction box */
    .prediction-box {{
        background-color: rgba(0, 100, 0, 0.7);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
    }}
    </style>
    """, unsafe_allow_html=True)
    

def main():
    # Add custom CSS
    add_custom_css()
    
    # Title
    st.title("🌲 GreenPredict: Forest Cover Type Prediction")
       
    # Create columns
    col1, col2 = st.columns([2, 1])
    
    # Left column for inputs
    with col1:
        st.header("Input Features")
        
        # Create form with input sliders
        with st.form("input_form"):
            # First row of sliders
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                elevation = st.slider("Elevation", 
                    min_value=input_ranges['Elevation']['min'], 
                    max_value=input_ranges['Elevation']['max'], 
                    value=(input_ranges['Elevation']['min'] + input_ranges['Elevation']['max']) // 2
                )
                slope = st.slider("Slope", 
                    min_value=input_ranges['Slope']['min'], 
                    max_value=input_ranges['Slope']['max'], 
                    value=(input_ranges['Slope']['min'] + input_ranges['Slope']['max']) // 2
                )
                horizontal_distance_to_hydrology = st.slider("Horizontal Distance to Hydrology", 
                    min_value=input_ranges['Horizontal_Distance_To_Hydrology']['min'], 
                    max_value=input_ranges['Horizontal_Distance_To_Hydrology']['max'], 
                    value=(input_ranges['Horizontal_Distance_To_Hydrology']['min'] + input_ranges['Horizontal_Distance_To_Hydrology']['max']) // 2
                )
            
            with col_b:
                horizontal_distance_to_roadways = st.slider("Horizontal Distance to Roadways", 
                    min_value=input_ranges['Horizontal_Distance_To_Roadways']['min'], 
                    max_value=input_ranges['Horizontal_Distance_To_Roadways']['max'], 
                    value=(input_ranges['Horizontal_Distance_To_Roadways']['min'] + input_ranges['Horizontal_Distance_To_Roadways']['max']) // 2
                )
                hillshade_9am = st.slider("Hillshade 9am", 
                    min_value=input_ranges['Hillshade_9am']['min'], 
                    max_value=input_ranges['Hillshade_9am']['max'], 
                    value=(input_ranges['Hillshade_9am']['min'] + input_ranges['Hillshade_9am']['max']) // 2
                )
                hillshade_noon = st.slider("Hillshade Noon", 
                    min_value=input_ranges['Hillshade_Noon']['min'], 
                    max_value=input_ranges['Hillshade_Noon']['max'], 
                    value=(input_ranges['Hillshade_Noon']['min'] + input_ranges['Hillshade_Noon']['max']) // 2
                )
            
            with col_c:
                hillshade_3pm = st.slider("Hillshade 3pm", 
                    min_value=input_ranges['Hillshade_3pm']['min'], 
                    max_value=input_ranges['Hillshade_3pm']['max'], 
                    value=(input_ranges['Hillshade_3pm']['min'] + input_ranges['Hillshade_3pm']['max']) // 2
                )
                horizontal_distance_to_fire_points = st.slider("Horizontal Distance to Fire Points", 
                    min_value=input_ranges['Horizontal_Distance_To_Fire_Points']['min'], 
                    max_value=input_ranges['Horizontal_Distance_To_Fire_Points']['max'], 
                    value=(input_ranges['Horizontal_Distance_To_Fire_Points']['min'] + input_ranges['Horizontal_Distance_To_Fire_Points']['max']) // 2
                )
                wilderness_area1 = st.slider("Wilderness Area 1", 
                    min_value=input_ranges['Wilderness_Area1']['min'], 
                    max_value=input_ranges['Wilderness_Area1']['max'], 
                    value=input_ranges['Wilderness_Area1']['min']
                )
            
            # Additional sliders
            col_d, col_e, col_f = st.columns(3)
            
            with col_d:
                wilderness_area3 = st.slider("Wilderness Area 3", 
                    min_value=input_ranges['Wilderness_Area3']['min'], 
                    max_value=input_ranges['Wilderness_Area3']['max'], 
                    value=input_ranges['Wilderness_Area3']['min']
                )
                wilderness_area4 = st.slider("Wilderness Area 4", 
                    min_value=input_ranges['Wilderness_Area4']['min'], 
                    max_value=input_ranges['Wilderness_Area4']['max'], 
                    value=input_ranges['Wilderness_Area4']['min']
                )
            
            with col_e:
                soil_type2 = st.slider("Soil Type 2", 
                    min_value=input_ranges['Soil_Type2']['min'], 
                    max_value=input_ranges['Soil_Type2']['max'], 
                    value=input_ranges['Soil_Type2']['min']
                )
                soil_type3 = st.slider("Soil Type 3", 
                    min_value=input_ranges['Soil_Type3']['min'], 
                    max_value=input_ranges['Soil_Type3']['max'], 
                    value=input_ranges['Soil_Type3']['min']
                )
            
            with col_f:
                soil_type10 = st.slider("Soil Type 10", 
                    min_value=input_ranges['Soil_Type10']['min'], 
                    max_value=input_ranges['Soil_Type10']['max'], 
                    value=input_ranges['Soil_Type10']['min']
                )
                soil_type12 = st.slider("Soil Type 12", 
                    min_value=input_ranges['Soil_Type12']['min'], 
                    max_value=input_ranges['Soil_Type12']['max'], 
                    value=input_ranges['Soil_Type12']['min']
                )
            
            # Predict button
            submitted = st.form_submit_button("Predict Forest Cover")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <style>
            .footer {
                color: white;
                background: linear-gradient(to right, rgba(255, 255, 255, 1), rgba(173, 216, 230, 1), rgba(255, 255, 255, 1));
                padding: 15px;
                border-radius: 5px;
            }
        </style>
        <div class="footer">
            <p><strong>Purpose:</strong><br>
            The app predicts forest cover types based on environmental features and visualizes the result with an image.</p>
            <p><strong>Impact:</strong><br>
            It provides an accessible, user-friendly tool for learning and applying forest cover classification, making complex predictions easy to understand for non-experts.</p>
        </div>
        """, unsafe_allow_html=True)
    
     # Right column for prediction
    with col2:
        # First show the default message or image
        if not submitted:
            st.write("Adjust the features and click 'Predict' to see the forest cover type.")
            
        else:
            user_input = {
                "Elevation": elevation,
                "Slope": slope,
                "Horizontal_Distance_To_Hydrology": horizontal_distance_to_hydrology,
                "Horizontal_Distance_To_Roadways": horizontal_distance_to_roadways,
                "Hillshade_9am": hillshade_9am,
                "Hillshade_Noon": hillshade_noon,
                "Hillshade_3pm": hillshade_3pm,
                "Horizontal_Distance_To_Fire_Points": horizontal_distance_to_fire_points,
                "Wilderness_Area1": wilderness_area1,
                "Wilderness_Area3": wilderness_area3,
                "Wilderness_Area4": wilderness_area4,
                "Soil_Type2": soil_type2,
                "Soil_Type3": soil_type3,
                "Soil_Type10": soil_type10,
                "Soil_Type12": soil_type12,
                "Soil_Type13": 0,
                "Soil_Type17": 0,
                "Soil_Type20": 0,
                "Soil_Type23": 0,
                "Soil_Type29": 0,
                "Soil_Type38": 0,
                "Soil_Type39": 0,
                "Soil_Type40": 0
            }
            
            result = predict_cover_type(user_input)
            if result is not None:
                # First display the image
                image_path = cover_type_mapping[result]["image_path"]
                try:
                    image = Image.open(image_path)
                    st.image(image, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
                
                # Then display the prediction box below the image
                st.markdown(f'<div class="prediction-box">Predicted Cover Type: {cover_type_mapping[result]["name"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()