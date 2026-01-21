#Tomo: 因为业拓改了Cosy的 float32/int16 输出，所以这个脚本有问题。我建议Cosy保持原来的。

import argparse
import math
import os
import struct
import sys
import time
import wave

import requests

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_pass(msg):
    print(f"{Colors.OKGREEN}[PASS] {msg}{Colors.ENDC}")

def print_fail(msg):
    print(f"{Colors.FAIL}[FAIL] {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}[INFO] {msg}{Colors.ENDC}")

GUESS_DTYPE = True
DTYPE = "float32" #业拓修改了cosyvoice代码，本来是int16，现在变成了float32

def guess_audio_dtype(raw_bytes, header_dtype=None):
    if header_dtype:
        lower = header_dtype.lower()
        if "float" in lower:
            return "float32"
        if "int16" in lower or "pcm16" in lower:
            return "int16"

    if len(raw_bytes) < 8:
        return "unknown"

    sample_count = min(len(raw_bytes) // 4, 256)
    if sample_count == 0:
        return "unknown"

    print("guessing audio dtype...")
    float_like = 0
    finite_count = 0
    try:
        for (value,) in struct.iter_unpack('<f', raw_bytes[: sample_count * 4]):
            if math.isfinite(value):
                finite_count += 1
                if -1.5 <= value <= 1.5:
                    float_like += 1
    except struct.error:
        return "unknown"

    if finite_count and float_like / finite_count > 0.9:
        return "float32"

    return "int16"


def convert_float32_bytes_to_int16(raw_bytes):
    sample_count = len(raw_bytes) // 4
    if sample_count == 0:
        return b""

    int16_buffer = bytearray(sample_count * 2)
    pack = struct.pack
    offset = 0
    for (value,) in struct.iter_unpack('<f', raw_bytes[: sample_count * 4]):
        if not math.isfinite(value):
            value = 0.0
        clipped = max(min(value, 1.0), -1.0)
        int16_sample = int(round(clipped * 32767.0))
        int16_buffer[offset:offset + 2] = pack('<h', int16_sample)
        offset += 2
    return bytes(int16_buffer)


def save_wav(path, sample_rate, audio_bytes, sample_width):
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)

def test_inference_zero_shot(server_ip, port):
    url = f"http://{server_ip}:{port}/inference_zero_shot"
    print_info(f"Testing {url}...")
    
    # Path to a test audio file
    # Assuming running from vc/test, so data is in ../test_data
    audio_path = "../test_data/seedtts_ref_zh_1.wav"
    
    # Check if file exists, if not try to find any wav in ../test_data
    if not os.path.exists(audio_path):
        test_data_dir = "../test_data"
        if os.path.exists(test_data_dir):
            files = os.listdir(test_data_dir)
            wavs = [f for f in files if f.endswith('.wav')]
            if wavs:
                audio_path = os.path.join(test_data_dir, wavs[0])
                print_info(f"Default audio not found, using found audio: {audio_path}")
            else:
                print_fail("No wav files found in ../test_data")
                return
        else:
             print_fail(f"Audio file {audio_path} not found and ../test_data does not exist.")
             return
    else:
        print_info(f"Using audio: {audio_path}")

    # For zero shot, we need prompt text. 
    # Using a generic prompt text.
    tts_text = "你好，我是CosyVoice，很高兴为你服务。"
    prompt_text = "希望你今天过得愉快。" 
    
    try:
        with open(audio_path, 'rb') as f:
            files = {
                'prompt_wav': (os.path.basename(audio_path), f, 'audio/wav')
            }
            data = {
                'tts_text': tts_text,
                'prompt_text': prompt_text
            }
            
            response = requests.post(url, files=files, data=data, stream=True)
            
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            output_filename = "output_zero_shot.wav"
            # CosyVoice2 usually uses 24000Hz sample rate
            sample_rate = 24000
            
            if False:
                reported_dtype = response.headers.get('X-Audio-Dtype') or response.headers.get('x-audio-dtype')
                if reported_dtype:
                    print_info(f"Server reports audio dtype: {reported_dtype}")
            else:
                reported_dtype = None

            audio_bytes = bytearray()
            chunk_count = 0
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_bytes.extend(chunk)
                    chunk_count += 1

            if chunk_count == 0 or not audio_bytes:
                print_fail("inference_zero_shot test failed (empty stream)")
                return

            if GUESS_DTYPE:
                detected_dtype = guess_audio_dtype(audio_bytes, reported_dtype)
                print_info(f"Detected audio dtype: {detected_dtype}")
            else:
                detected_dtype = DTYPE

            raw_output_path = "output_zero_shot_raw.pcm"
            try:
                with open(raw_output_path, 'wb') as raw_file:
                    raw_file.write(audio_bytes)
                print_info(f"Raw audio saved to {raw_output_path}")
            except Exception as raw_err:
                print_fail(f"Failed to save raw audio: {raw_err}")

            if detected_dtype == "float32":
                float_raw_path = "output_zero_shot_float32.pcm"
                try:
                    with open(float_raw_path, 'wb') as float_file:
                        float_file.write(audio_bytes)
                    print_info(f"Float32 audio saved to {float_raw_path}")
                except Exception as float_err:
                    print_fail(f"Failed to save float32 audio: {float_err}")

                int16_bytes = convert_float32_bytes_to_int16(audio_bytes)
                if int16_bytes:
                    try:
                        save_wav(output_filename, sample_rate, int16_bytes, 2)
                        print_pass(f"inference_zero_shot test passed. Float32 converted to int16 and saved to {output_filename}")
                    except Exception as wav_err:
                        print_fail(f"Error saving converted wav file: {wav_err}")
                else:
                    print_fail("Float32 stream was empty after conversion")

            elif detected_dtype == "int16":
                try:
                    save_wav(output_filename, sample_rate, bytes(audio_bytes), 2)
                    print_pass(f"inference_zero_shot test passed. Int16 audio saved to {output_filename}")
                except Exception as wav_err:
                    print_fail(f"Error saving int16 wav file: {wav_err}")

            else:
                print_info("Unable to determine dtype confidently; saving both interpretations.")
                # Save assuming int16
                try:
                    save_wav(output_filename, sample_rate, bytes(audio_bytes), 2)
                    print_info(f"Saved int16 interpretation to {output_filename}")
                except Exception as wav_err:
                    print_fail(f"Error saving int16 interpretation: {wav_err}")

                # Attempt float32 conversion as alternative interpretation
                int16_bytes = convert_float32_bytes_to_int16(audio_bytes)
                alt_output = "output_zero_shot_from_float32.wav"
                if int16_bytes:
                    try:
                        save_wav(alt_output, sample_rate, int16_bytes, 2)
                        print_info(f"Saved float32->int16 interpretation to {alt_output}")
                    except Exception as alt_err:
                        print_fail(f"Error saving float32 interpretation: {alt_err}")
                else:
                    print_fail("Float32 interpretation failed (conversion produced no data)")
        else:
            print_fail(f"inference_zero_shot test failed with status {response.status_code}")
            try:
                print(f"Response: {response.text[:200]}")
            except:
                pass
            
    except Exception as e:
        print_fail(f"Error testing inference_zero_shot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CosyVoice Service")
    parser.add_argument("--server_ip", default="127.0.0.1", help="Server IP address")
    parser.add_argument("--port", type=int, default=49999, help="Server port")
    parser.add_argument("--repeat", type=int, default=3, help="Number of times to repeat the test")
    
    args = parser.parse_args()
    
    for i in range(args.repeat):
        print(f"\n--- Run {i+1}/{args.repeat} ---")
        start_time = time.time()
        test_inference_zero_shot(args.server_ip, args.port)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")
