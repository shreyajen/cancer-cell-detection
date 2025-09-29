from flask import Flask, render_template, request
import numpy as np
import pickle

# Load model and scaler
lr = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data (all 30 inputs)
        input_features = [float(x) for x in request.form.values()]
        
        # Convert to numpy array
        final_input = np.array(input_features).reshape(1, -1)
        
        # Scale features
        input_scaled = scaler.transform(final_input)
        
        # Make prediction
        prediction = lr.predict(input_scaled)[0]
        
        # Format output
        if prediction == 0:
            result = "🎉 The tumor is BENIGN — not cancerous."
        else:
            result = "⚠️ The tumor is MALIGNANT — cancerous."
        
        return render_template("index.html", prediction_text=result)
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
