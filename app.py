import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

# Initialize the Flask app
app = Flask(__name__)

# Load the Hugging Face model and tokenizer for the Llama model
model_name = "mattshumer/Reflection-Llama-3.1-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Sample responses based on sentiment (dummy example, can enhance based on model usage)
responses = {
    "positive": "I'm glad you're feeling positive! Keep up the great mood!",
    "neutral": "It seems you're feeling neutral. I'm here if you want to talk!",
    "negative": "I'm sorry you're feeling this way. Here's a helpline for support: 1-800-123-4567."
}

def analyze_sentiment(text):
    """Function to analyze the sentiment of the user input."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs["input_ids"], max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Based on generated text, let's assume it has sentiment analysis capabilities or text summarization
    if "happy" in generated_text.lower() or "good" in generated_text.lower():
        return "positive"
    elif "sad" in generated_text.lower() or "bad" in generated_text.lower():
        return "negative"
    else:
        return "neutral"

@app.route('/chatbot', methods=['POST'])
def chatbot():
    """Handle user input and provide appropriate responses."""
    user_input = request.json['input']
    sentiment = analyze_sentiment(user_input)
    response = responses[sentiment]
    return jsonify({"response": response, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(debug=True)
