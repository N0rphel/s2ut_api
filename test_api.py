import requests
import json

def check_health():
    """Access the health endpoint of the Flask API."""
    try:
        response = requests.get('http://127.0.0.1:5000/health')
        
        # Print status code
        print(f"Status Code: {response.status_code}")
        
        # Print response content
        print("Response Content:")
        print(response.text)
        
        # If the response is JSON, print it formatted
        try:
            print("JSON Response:")
            print(response.json())
        except:
            # Not JSON content
            pass
            
        return response
    except requests.exceptions.RequestException as e:
        print(f"Error accessing health endpoint: {e}")
        return None

def translate_audio(audio_file_path, speaker_id, api_url, output_file_path):
    """
    Send an audio file to the translation API and save the response to a file.
    
    Args:
        audio_file_path (str): Path to the audio file to translate
        speaker_id (str): ID of the speaker
        api_url (str): URL of the translation API endpoint
        output_file_path (str): Path where the output WAV file should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the multipart form data with the audio file and speaker_id
        files = {
            'audio': open(audio_file_path, 'rb')
        }
        data = {
            'speaker_id': speaker_id
        }
        
        # Send the POST request
        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, files=files, data=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the binary response content to the output file
            with open(output_file_path, 'wb') as output_file:
                output_file.write(response.content)
            print(f"Translation successful! Output saved to {output_file_path}")
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        # Make sure to close the file if it was opened
        try:
            files['audio'].close()
        except:
            pass

def extract_units(audio_file_path, api_url):
    """
    Send an audio file to the API and extract units.
    
    Args:
        audio_file_path (str): Path to the audio file
        api_url (str): URL of the units extraction API endpoint
    
    Returns:
        list or None: Extracted units if successful, None otherwise
    """
    try:
        # Prepare the multipart form data with the audio file
        files = {
            'audio': open(audio_file_path, 'rb')
        }
        
        # Send the POST request
        print(f"Sending request to {api_url} to extract units...")
        response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            units = response.json()
            print(f"Units extracted successfully: {units}")
            return units
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return None
            
    except Exception as e:
        print(f"An error occurred during units extraction: {e}")
        return None
    finally:
        # Make sure to close the file if it was opened
        try:
            files['audio'].close()
        except:
            pass

def generate_from_units(units, speaker_id, api_url, output_file_path):
    """
    Generate audio from units and save to a file.
    
    Args:
        units (list): List of unit indices
        speaker_id (int): ID of the speaker
        api_url (str): URL of the generation API endpoint
        output_file_path (str): Path where the output WAV file should be saved
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Prepare the JSON payload
        payload = {
            "units": units,
            "speaker_id": speaker_id
        }
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        
        # Send the POST request
        print(f"Sending request to {api_url} to generate audio...")
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the binary response content to the output file
            with open(output_file_path, 'wb') as output_file:
                output_file.write(response.content)
            print(f"Audio generation successful! Output saved to {output_file_path}")
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response content: {response.text}")
            return False
            
    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        return False

if __name__ == "__main__":
    # check_health()

    audio_file_path = "maudio_1.wav"  # Path to your input audio file
    speaker_id = "0"                  # Speaker ID
    api_url = "http://127.0.0.1:5000/translate"  # API endpoint
    units_api_url = "http://127.0.0.1:5000/units_only"
    generate_api_url = "http://127.0.0.1:5000/generate_from_units"
    output_file_path = "api_test.wav"   # Where to save the translated audio
    
    translate_audio(audio_file_path, speaker_id, api_url, output_file_path)

    # Example 1: Extract units only
    # extracted_units = extract_units(audio_file_path, units_api_url)
    # print(extracted_units)

    # # Example 2: Generate from units
    # # You can use the extracted units from the previous call, or use your predefined units
    # # predefined_units = [71, 67, 89, 21, 23, 4, 23, 32, 23, 23, 21]
    # # units_to_use = extracted_units if extracted_units else predefined_units
    
    # # generate_from_units(units_to_use, speaker_id, generate_api_url, output_file_path)