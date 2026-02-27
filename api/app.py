"""
Minimal Flask API for Desert Semantic Segmentation.

Provides:
- Health check endpoint
- Ready-to-extend inference endpoint
- Production-structured JSON responses

Designed for deployment inside Docker or cloud services.
"""

from flask import Flask, jsonify, request
import torch

app = Flask(__name__)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------

@app.route("/", methods=["GET"])
def health():
    """
    Basic health check endpoint.

    Returns:
        JSON response confirming API status and device availability.
    """
    return jsonify({
        "status": "ok",
        "device": DEVICE
    })


# --------------------------------------------------
# MODEL STATUS ENDPOINT
# --------------------------------------------------

@app.route("/model-status", methods=["GET"])
def model_status():
    """
    Returns model availability information.
    Useful for deployment monitoring.
    """
    return jsonify({
        "model_loaded": True,  # Replace with actual check if needed
        "device": DEVICE
    })


# --------------------------------------------------
# INFERENCE ENDPOINT (Stub)
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Placeholder inference endpoint.

    Expected:
        JSON or image input (extend as needed)

    Returns:
        Dummy response (replace with real inference logic)
    """
    return jsonify({
        "message": "Inference endpoint not yet implemented."
    })


# --------------------------------------------------
# APP ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    # Use host 0.0.0.0 for Docker compatibility
    app.run(host="0.0.0.0", port=5000, debug=False)
