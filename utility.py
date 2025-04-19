import logging
import os
import soundfile as sf
import torch
import torchaudio
import numpy as np
import json
from pathlib import Path
from argparse import Namespace
from fairseq import checkpoint_utils, tasks
# Updated import to address deprecation warning
from torch.nn.utils.parametrizations import weight_norm
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

torch.serialization.add_safe_globals([Namespace])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global device setting - change this to "cuda" if you want to use GPU
DEVICE = "cpu"  # Force CPU usage for all operations

def extract_features(audio_path, num_mel_bins=80, device=DEVICE):
    """Extract log Mel-filterbank features from an audio file with proper normalization."""
    logger.info(f"Extracting features from {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if needed
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Move to specified device
    waveform = waveform.to(device)
    
    # Normalize audio waveform to [-1, 1] range
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-10)
    
    # Resample if needed
    if sample_rate != 16000:
        logger.info(f"Resampling from {sample_rate} to 16000 Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)
    
    # Pre-emphasis filter
    preemph = 0.97
    preemphasized = torch.cat([waveform[:, :1], waveform[:, 1:] - preemph * waveform[:, :-1]], dim=1)
    
    # Extract log-Mel features
    feature_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        n_mels=num_mel_bins,
        hop_length=160,
        win_length=400,
        f_min=20,
        f_max=8000,
        window_fn=torch.hamming_window,
        norm='slaney',
        mel_scale='htk'
    ).to(device)
    
    mel_spec = feature_extractor(preemphasized)
    log_mel_spec = torch.log(mel_spec + 1e-10)
    
    # Global CMVN
    mean = torch.mean(log_mel_spec, dim=2, keepdim=True)
    std = torch.std(log_mel_spec, dim=2, keepdim=True)
    normalized_features = (log_mel_spec - mean) / (std + 1e-10)
    
    # Convert to the expected shape [T, F]
    normalized_features = normalized_features.squeeze(0).transpose(0, 1)
    logger.info(f"Features extracted with shape: {normalized_features.shape}")
    
    return normalized_features  # Shape: [T, 80]

def generate_single_speech_to_speech(
    model_path,
    data_path,
    config_yaml,
    input_audio_path,
    output_audio_path=None,
    device=DEVICE,
    beam=10,
    max_tokens=20000,
    target_is_code=True,
    target_code_size=100,
    vocoder="code_hifigan"
):
    """Generate speech units from input audio using the S2UT model."""
    try:
        # Verify input file exists
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")
            
        # Verify model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Verify config file exists
        if not os.path.exists(config_yaml):
            raise FileNotFoundError(f"Config file not found: {config_yaml}")
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger("speech_to_speech_generator")
    
        # Setup args
        args = Namespace()
        args.seed = 1
        args.fp16 = False
        args.cpu = (device == "cpu")  # Set based on device
        args.path = model_path
        args.quiet = False
        args.model_overrides = "{}"
        args.max_tokens = max_tokens
        args.batch_size = None
        args.gen_subset = "test"
        args.skip_invalid_size_inputs_valid_test = True
        args.generation = Namespace()
        args.generation.beam = beam
        args.generation.max_len_a = 1
        args.generation.max_len_b = 200
        args.generation.sampling = False
        args.checkpoint_suffix = ""
        args.checkpoint_shard_count = 1
        args.task = "speech_to_speech"
        args.data = data_path
        args.config_yaml = config_yaml
        args.target_is_code = target_is_code
        args.target_code_size = target_code_size
        args.vocoder = vocoder
        args.n_frames_per_step = 1
        args.eval_inference = False
        args.eval_args = "{}"
        args.infer_target_lang = ""
        args.train_subset = "train"
        
        # Set CPU threads if using CPU
        if device == "cpu":
            torch.set_num_threads(os.cpu_count())
    
        # Setup task
        logger.info("Setting up task")
        task = tasks.setup_task(args)
    
        # Load model
        logger.info(f"Loading model from {model_path}")
        models, _ = checkpoint_utils.load_model_ensemble([model_path], task=task)
    
        # Move all models to the specified device
        for model in models:
            model.eval()
            model.to(device)
            model.prepare_for_inference_(args)
    
        # Extract features from audio
        logger.info(f"Loading and preprocessing audio from {input_audio_path}")
        features = extract_features(input_audio_path, num_mel_bins=80, device=device)
        
        # Add batch dimension
        features = features.unsqueeze(0)  # Shape: [1, T, 80]
    
        # Create sample dictionary
        sample = {
            "id": torch.LongTensor([0]).to(device),
            "net_input": {
                "src_tokens": features,
                "src_lengths": torch.LongTensor([features.size(1)]).to(device),
                "prev_output_tokens": None,
            },
            "target": None,
            "target_lengths": None,
            "ntokens": features.size(1),
            "nsentences": 1,
        }
        
        # Build generator and generate
        generator = task.build_generator(models, args)
        logger.info("Generating speech units...")
        try:
            hypos = task.inference_step(generator, models, sample)
            logger.info("Speech unit generation completed")
        except Exception as e:
            logger.error(f"Error during unit generation: {e}")
            raise
    
        # Process output
        hypo = hypos[0][0]
        units = hypo["tokens"].cpu().tolist()
        adjusted_units = [x - 4 for x in units]
        
        logger.info(f"Generated {len(units)} units")
        
        return units, adjusted_units[:-1]  # Remove the last token which is likely EOS
        
    except Exception as e:
        logger.error(f"Error in generate_single_speech_to_speech: {str(e)}", exc_info=True)
        raise

def generate_audio_from_units(
    units,
    vocoder_path,
    vocoder_cfg_path,
    output_path=None,
    speaker_id=0,
    dur_prediction=False,
    device=DEVICE
):
    """Generate audio from a sequence of discrete units using CodeHiFiGAN vocoder."""
    try:
        # Verify vocoder files exist
        if not os.path.exists(vocoder_path):
            raise FileNotFoundError(f"Vocoder model file not found: {vocoder_path}")
            
        if not os.path.exists(vocoder_cfg_path):
            raise FileNotFoundError(f"Vocoder config file not found: {vocoder_cfg_path}")
        
        logger.info(f"Using device: {device}")
        
        # Load vocoder configuration
        with open(vocoder_cfg_path) as f:
            vocoder_cfg = json.load(f)
        
        # Initialize vocoder
        logger.info(f"Loading vocoder from {vocoder_path}")
        vocoder = CodeHiFiGANVocoder(vocoder_path, vocoder_cfg)
        
        # Move vocoder to the specified device
        vocoder.model = vocoder.model.to(device)
        
        # Check if multi-speaker
        multispkr = vocoder.model.multispkr
        logger.info(f"Vocoder is multi-speaker: {multispkr}")
        
        if multispkr:
            num_speakers = vocoder_cfg.get("num_speakers", 200)
            if speaker_id >= num_speakers:
                logger.warning(f"Invalid speaker_id {speaker_id}, using speaker 0 instead")
                speaker_id = 0
        
        # Convert units to tensor and ensure proper shape
        if isinstance(units, np.ndarray):
            units = torch.from_numpy(units).long()
        elif isinstance(units, list):
            units = torch.LongTensor(units)
            
        if units.dim() == 1:
            units = units.unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Units tensor shape: {units.shape}")
        
        # Create input dictionary
        x = {"code": units.to(device)}
        
        # Add speaker information if multi-speaker
        if multispkr:
            x["spkr"] = torch.LongTensor([speaker_id]).view(1, 1).to(device)
            logger.info(f"Using speaker ID: {speaker_id}")
        
        # Log input shapes for debugging
        for k, v in x.items():
            if isinstance(v, torch.Tensor):
                logger.info(f"Input {k} shape: {v.shape}, device: {v.device}")
        
        # Generate waveform
        logger.info("Generating audio waveform...")
        try:
            with torch.no_grad():
                wav = vocoder(x, dur_prediction)
            logger.info("Audio waveform generation completed")
        except Exception as e:
            logger.error(f"Vocoder error: {e}")
            raise
        
        # Convert to numpy safely
        wav_np = wav.detach().cpu().numpy().squeeze()
        
        # Save if output path is provided
        if output_path:
            logger.info(f"Saving audio to {output_path}")
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            sf.write(output_path, wav_np, 16000)
        
        return wav_np
        
    except Exception as e:
        logger.error(f"Error in generate_audio_from_units: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Test the module if run directly
    try:
        # Define paths for testing
        model_path = "dzo_s2ut_model/checkpoint_last.pt"
        data_path = "dzo_s2ut_model/dataroot"
        config_yaml = "dzo_s2ut_model/dataroot/config.yaml"  # Use relative path that matches your structure
        input_audio = "maudio_1.wav"
        output_audio = "output.wav"
        
        # Generate speech units
        units, adjusted_units = generate_single_speech_to_speech(
            model_path=model_path,
            data_path=data_path,
            config_yaml=config_yaml,
            input_audio_path=input_audio,
            output_audio_path=None,
            device=DEVICE,
            beam=10,
            max_tokens=20000
        )
    
        # Define vocoder paths
        vocoder_path = "dzo_vocoder/g_00015500"
        vocoder_cfg_path = "dzo_vocoder/config.json"
        
        # Generate audio from units
        waveform = generate_audio_from_units(
            units=adjusted_units,
            vocoder_path=vocoder_path,
            vocoder_cfg_path=vocoder_cfg_path,
            output_path=output_audio,
            speaker_id=0,
            device=DEVICE
        )
        
        logger.info(f"Speech generation test completed. Audio saved to {output_audio}")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)