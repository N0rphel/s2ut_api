import os
import io
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import soundfile as sf
import torch

# Import functionality from the utility module
from utility import (
    extract_features,
    generate_single_speech_to_speech,
    generate_audio_from_units,
    DEVICE
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define paths - use absolute paths to avoid issues with working directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = str(BASE_DIR / "dzo_s2ut_model" / "checkpoint_last.pt")
DATA_PATH = str(BASE_DIR / "dzo_s2ut_model" / "dataroot")
CONFIG_YAML = str(BASE_DIR / "dzo_s2ut_model" / "dataroot" / "config.yaml")
VOCODER_PATH = str(BASE_DIR / "dzo_vocoder" / "g_00015500")
VOCODER_CFG_PATH = str(BASE_DIR / "dzo_vocoder" / "config.json")
UPLOAD_FOLDER = str(BASE_DIR / "uploads")
OUTPUT_FOLDER = str(BASE_DIR / "outputs")

# Create necessary directories
Path(UPLOAD_FOLDER).mkdir(exist_ok=True, parents=True)
Path(OUTPUT_FOLDER).mkdir(exist_ok=True, parents=True)

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "model_path_exists": os.path.exists(MODEL_PATH),
            "vocoder_path_exists": os.path.exists(VOCODER_PATH),
            "device": DEVICE
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route("/test_model_load", methods=["GET"])
def test_model_load():
    """Test endpoint to verify models can be loaded without processing audio"""
    try:
        # Just attempt to load model configurations without processing
        from fairseq import tasks
        from argparse import Namespace
        
        args = Namespace()
        args.task = "speech_to_speech"
        args.data = DATA_PATH
        args.config_yaml = CONFIG_YAML
        
        task = tasks.setup_task(args)
        logger.info("Task setup successful")
        
        # Load vocoder config to test
        import json
        with open(VOCODER_CFG_PATH) as f:
            vocoder_cfg = json.load(f)
        logger.info("Vocoder config loaded successfully")
        
        return jsonify({
            "status": "success",
            "message": "Model configurations loaded successfully"
        })
    except Exception as e:
        logger.error(f"Model load test failed: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/translate", methods=["POST"])
def translate_speech():
    """Endpoint to translate speech from one language to another"""
    try:
        # Check if file is present in request
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        logger.info(f"Received file: {audio_file.filename}")
        
        # Check if filename is empty
        if audio_file.filename == "":
            return jsonify({"error": "No audio file selected"}), 400
        
        # Optional parameters with defaults
        speaker_id = int(request.form.get("speaker_id", 0))
        beam_size = int(request.form.get("beam_size", 10))
        max_tokens = int(request.form.get("max_tokens", 20000))
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(input_path)
        logger.info(f"Saved uploaded audio to {input_path}")
        
        # Generate a unique output filename
        output_filename = f"output_{os.path.splitext(audio_file.filename)[0]}_{speaker_id}.wav"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Generate speech units
        logger.info("Starting speech unit generation...")
        units, adjusted_units = generate_single_speech_to_speech(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            config_yaml=CONFIG_YAML,
            input_audio_path=input_path,
            output_audio_path=None,  # We'll generate audio separately
            device=DEVICE,
            beam=beam_size,
            max_tokens=max_tokens
        )
        
        logger.info(f"Generated {len(adjusted_units)} units")
        
        # Generate audio from units
        logger.info("Starting audio generation from units...")
        waveform = generate_audio_from_units(
            units=adjusted_units,
            vocoder_path=VOCODER_PATH,
            vocoder_cfg_path=VOCODER_CFG_PATH,
            output_path=output_path,
            speaker_id=speaker_id,
            device=DEVICE
        )
        
        logger.info(f"Audio generation completed, sending file: {output_path}")
        
        # Return the file directly
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_filename
        )
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}", exc_info=True)
        return jsonify({"error": f"File not found: {str(e)}"}), 404
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500

@app.route("/units_only", methods=["POST"])
def extract_units_only():
    """Endpoint to only extract speech units from audio without generation"""
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files["audio"]
        logger.info(f"Received file for unit extraction: {audio_file.filename}")
        
        if audio_file.filename == "":
            return jsonify({"error": "No audio file selected"}), 400
        
        beam_size = int(request.form.get("beam_size", 10))
        max_tokens = int(request.form.get("max_tokens", 20000))
        
        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(input_path)
        logger.info(f"Saved uploaded audio to {input_path}")
        
        # Generate speech units only
        logger.info("Starting speech unit extraction...")
        units, adjusted_units = generate_single_speech_to_speech(
            model_path=MODEL_PATH,
            data_path=DATA_PATH,
            config_yaml=CONFIG_YAML,
            input_audio_path=input_path,
            output_audio_path=None,
            device=DEVICE,
            beam=beam_size,
            max_tokens=max_tokens
        )
        
        logger.info(f"Successfully extracted {len(units)} units")
        
        return jsonify({
            "original_units": units,
            "adjusted_units": adjusted_units,
            "num_units": len(adjusted_units)
        })
        
    except Exception as e:
        logger.error(f"Error extracting units: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error extracting units: {str(e)}"}), 500

@app.route("/generate_from_units", methods=["POST"])
def generate_from_units():
    """Endpoint to generate speech from provided units"""
    try:
        # Check if units are in the request
        if not request.json or "units" not in request.json:
            return jsonify({"error": "No units provided in request"}), 400
        
        units = request.json["units"]
        logger.info(f"Received {len(units)} units for audio generation")
        
        speaker_id = request.json.get("speaker_id", 0)
        output_filename = request.json.get("output_filename", "generated_output.wav")
        
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # Generate audio from the provided units
        logger.info("Starting audio generation from provided units...")
        waveform = generate_audio_from_units(
            units=units,
            vocoder_path=VOCODER_PATH,
            vocoder_cfg_path=VOCODER_CFG_PATH,
            output_path=output_path,
            speaker_id=speaker_id,
            device=DEVICE
        )
        
        logger.info(f"Audio generation completed, sending file: {output_path}")
        
        return send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name=output_filename
        )
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error generating audio: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask app with configurations:")
    logger.info(f"MODEL_PATH: {MODEL_PATH}")
    logger.info(f"VOCODER_PATH: {VOCODER_PATH}")
    logger.info(f"DEVICE: {DEVICE}")
    
    # For development only - use a proper WSGI server in production
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False to avoid reloading issues