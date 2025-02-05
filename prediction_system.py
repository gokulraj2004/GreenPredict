import pandas as pd
import numpy as np
import joblib

def predict_cover_type(user_input):
    """Predicts the forest cover type based on user input."""
    try:
        # Load the model and scaler
        model = joblib.load(r'Model\decision_tree_model.pkl')
        scaler = joblib.load(r'scaler\scaler.pkl')  # Raw string (Recommended)
        
        # Define features in exact order used during training
        features = [
            'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type10', 'Soil_Type38',
            'Slope', 'Soil_Type39', 'Soil_Type40', 'Soil_Type17', 'Soil_Type3',
            'Soil_Type2', 'Soil_Type13', 'Wilderness_Area1',
            'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
            'Soil_Type29', 'Elevation', 'Hillshade_Noon', 'Soil_Type12',
            'Soil_Type23', 'Horizontal_Distance_To_Hydrology', 'Soil_Type20',
            'Hillshade_9am', 'Hillshade_3pm'
        ]
        
        # Create input DataFrame with correct features and order
        input_data = {}
        for feature in features:
            input_data[feature] = user_input.get(feature, 0)
            
        input_df = pd.DataFrame([input_data])
        
        # Ensure columns are in correct order
        input_df = input_df[features]
        
        # Scale the input
        input_scaled = scaler.transform(input_df)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        
        return prediction
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Create test input with only the features used in training
    test_input = {
    "Elevation": 2086,
    "Slope": 31,
    "Horizontal_Distance_To_Hydrology": 34,
    "Horizontal_Distance_To_Roadways": 216,
    "Hillshade_9am": 121,
    "Hillshade_Noon": 255,
    "Hillshade_3pm": 189,
    "Horizontal_Distance_To_Fire_Points": 144,
    "Wilderness_Area1": 69,
    "Wilderness_Area3": 150,
    "Wilderness_Area4": 0,
    "Soil_Type2": 0,
    "Soil_Type3": 0,
    "Soil_Type10": 1,
    "Soil_Type12": 0,
    "Soil_Type13": 0,
    "Soil_Type17": 0,
    "Soil_Type20": 0,
    "Soil_Type23": 0,
    "Soil_Type29": 0,
    "Soil_Type38": 0,
    "Soil_Type39": 0,
    "Soil_Type40": 0
}

    print(test_input)

    
    # Make prediction
    result = predict_cover_type(test_input)
    if result is not None:
        print(f"Predicted Cover Type: {result}")