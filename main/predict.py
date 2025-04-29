import numpy as np
from preprocess_image import preprocess_image

def predict_disease(image_path, model, label_map, threshold=0.6):
    # Preprocess the image
    image = preprocess_image(image_path)
    
    # Predict
    preds = model.predict(image, verbose=0)  # verbose=0 to avoid logs during prediction
    confidence = np.max(preds)
    predicted_index = np.argmax(preds)

    if confidence >= threshold:
        return {
            "status": "success",
            "disease": label_map[predicted_index],
            "confidence": round(confidence, 2)
        }
    else:
        return {
            "status": "uncertain",
            "message": "Prediction is not confident. Try retaking the image.",
            "confidence": round(confidence, 2)
        }
