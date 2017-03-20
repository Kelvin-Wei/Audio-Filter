import wave
import math
import sys
import numpy as np

def translate_wav(raw_bytes, n_frames, n_channels, sample_width):
    if sample_width == 1:
        dtype = np.uint8
    elif sample_width == 2:
        dtype = np.int16
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)
    channels.shape = (n_frames, n_channels)
    channels = channels.T
    return channels

def high_pass_running_mean(channel, window_size):
	# creates a 1d filter of size window_size
	# like a 1d edge detector, looks like: [-1,-1, 4,-1,-1]
	array = np.full(window_size, -1)
	array[window_size//2] = window_size - 1
	return np.convolve(channel, array, 'valid')/window_size

def low_pass_running_mean(channel, window_size):
	# does a cumulative sum, and inserts 0 at 0 position to balance edge case
	cumsum = np.cumsum(np.insert(channel, 0, 0)) 

	# takes difference between ith element and ith element + window_size and divides by window_size
	# exact same as 1d convolve of channel with array of ones of size window_size but faster
	return (cumsum[window_size:] - cumsum[:-window_size]) / window_size 

def moving_average_filter(cutoff_frequency, frame_rate, channels, filter_type):
	# Reference http://dsp.stackexchange.com/questions/9966/what-is-the-cut-off-frequency-of-a-moving-average-filter
	frequency_ratio = cutoff_frequency / frame_rate
	window_size = int(math.sqrt(0.19696202 + frequency_ratio ** 2) / frequency_ratio)

	filtered_wav = None

	if filter_type == "high":
		filtered_wav = high_pass_running_mean(channels[0], window_size).astype(channels.dtype)
	elif filter_type == "low":
		filtered_wav = low_pass_running_mean(channels[0], window_size).astype(channels.dtype)

	return filtered_wav

def filter_wav(in_file, out_file, cutoff_frequency, filter_type):
	in_wav = wave.open(in_file, 'rb')

	frame_rate = in_wav.getframerate()
	sample_width = in_wav.getsampwidth()
	n_channels = in_wav.getnchannels()
	n_frames = in_wav.getnframes()
	comp_type = in_wav.getcomptype()
	comp_name = in_wav.getcompname()

	signal = in_wav.readframes(n_frames * n_channels)
	in_wav.close()

	#Get channels of wav
	channels = translate_wav(signal, n_frames, n_channels, sample_width)
	#filter wav either high pass or low pass
	filtered_wav = moving_average_filter(cutoff_frequency, frame_rate, channels, filter_type)
	

	filtered_wav_file = wave.open(out_file, "w")
	filtered_wav_file.setparams((1, sample_width, frame_rate, n_frames, comp_type, comp_name))
	filtered_wav_file.writeframes(filtered_wav.tobytes('C'))
	filtered_wav_file.close()


def main():
	if len(sys.argv) == 2 and sys.argv[1] == "help":
		print("<input file> path to input wav file")
		print("<output file> path to output wav file")
		print("<cutoff frequency> frequency filter attenuates")
		print("<filter type> \"low\" \"high\" \"band\"")
	elif len(sys.argv) == 5:
		input_file = sys.argv[1]
		output_file = sys.argv[2]
		cutoff_frequency = int(sys.argv[3])
		filter_type = sys.argv[4]
		filter_wav(input_file, output_file, cutoff_frequency, filter_type)
	else:
		print("Command line arguments: <input file> <output file> <cutoff frequency> <filter type>")

if __name__ == "__main__":
	main()
