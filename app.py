import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load the Hugging Face model and tokenizer for the Llama model
model_name = "mattshumer/Reflection-Llama-3.1-70B"

print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Sample responses based on sentiment (dummy example, can enhance based on model usage)
responses = {
    "positive": "I'm glad you're feeling positive! Keep up the great mood!",
    "neutral": "It seems you're feeling neutral. I'm here if you want to talk!",
    "negative": "I'm sorry you're feeling this way. Here's a helpline for support: 1-800-123-4567."
}

def analyze_sentiment(text):
    """Function to analyze the sentiment of the user input."""
    try:
        inputs = tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        outputs = model.generate(inputs["input_ids"], max_length=100)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Basic sentiment analysis based on generated text
        if "happy" in generated_text.lower() or "good" in generated_text.lower():
            return "positive"
        elif "sad" in generated_text.lower() or "bad" in generated_text.lower():
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return "neutral"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle user input and provide appropriate responses."""
    try:
        user_input = request.json.get('input', '')
        if not user_input:
            return jsonify({"error": "Invalid input."}), 400

        sentiment = analyze_sentiment(user_input)
        response = responses.get(sentiment, "I'm not sure how to respond to that.")
        return jsonify({"response": response, "sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5001 and make it accessible from external networks
    app.run(host="0.0.0.0", port=5001, debug=True)
