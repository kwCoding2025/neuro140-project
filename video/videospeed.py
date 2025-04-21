import ffmpeg

def speed_up_video(input_path, output_path, speed_factor):
    """
    Speed up video using ffmpeg-python.

    For the video stream:
      - The 'setpts' filter is applied. The new presentation timestamp is scaled by 1/speed_factor.
    
    For the audio stream:
      - The 'atempo' filter is used. Since it only accepts values between 0.5 and 2.0 per filter,
        multiple atempo filters will be chained if the speed_factor exceeds that range.

    Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path to the output video file.
        speed_factor (float): Factor by which to speed up the video (e.g., 1.2 increases speed by 20%).
    """
    
    # Load the input file
    stream = ffmpeg.input(input_path)
    
    # Process video: speed up by adjusting presentation timestamps
    video = stream.video.filter('setpts', f'PTS/{speed_factor}')
    
    # Helper function to chain atempo filters for audio
    def chain_atempo(audio, speed):
        # atempo only supports values in [0.5, 2.0]. If speed falls outside of this range,
        # apply multiple atempo filters consecutively.
        factors = []
        remaining_speed = speed

        # Decrease speed factor if greater than 2.0
        while remaining_speed > 2.0:
            factors.append(2.0)
            remaining_speed /= 2.0
        # Increase speed factor if less than 0.5
        while remaining_speed < 0.5:
            factors.append(0.5)
            remaining_speed /= 0.5
        # Append the last factor if it's significantly different from 1
        if abs(remaining_speed - 1.0) > 1e-3:
            factors.append(remaining_speed)

        # Chain the atempo filters
        for factor in factors:
            audio = audio.filter('atempo', factor)
        return audio

    # Process audio: chain atempo filters to adjust the speed
    audio = chain_atempo(stream.audio, speed_factor)
    
    # Combine the processed video and audio streams into the output file
    out = ffmpeg.output(video, audio, output_path)
    ffmpeg.run(out)

if __name__ == "__main__":
    input_file = "video1584805112.mp4" # Specific input file
    original_duration_seconds = (7 * 60) + 17 # 7:17
    target_duration_seconds = 5 * 60 # 5:00
    speed_factor = original_duration_seconds / target_duration_seconds
    output_file = "video1584805112_5min.mp4" # Output file indicating target duration

    print(f"Input: {input_file}")
    print(f"Target duration: 5:00")
    print(f"Calculated speed factor: {speed_factor:.4f}")
    print(f"Output: {output_file}")

    try:
        speed_up_video(input_file, output_file, speed_factor)
        print(f"Successfully created sped up version: {output_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
