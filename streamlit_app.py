import streamlit as st
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Define a simple POST route
@app.route('/api/echo', methods=['POST'])
def echo():
    data = request.json  # Get the JSON data sent via POST
    message = data.get('message', 'No message provided')
    response = {'response': f"Received your message: {message}"}
    return jsonify(response)

# Set up Streamlit to start the Flask app in the background
def run_flask_app():
    app.run(port=5000, use_reloader=False)

# Streamlit UI
st.title('Simple Streamlit App with Flask API')

# Button to start Flask
if st.button('Start Flask API'):
    st.write("Flask API is running on http://localhost:5000/api/echo")
    run_flask_app()
else:
    st.write("Click the button to start the Flask API")