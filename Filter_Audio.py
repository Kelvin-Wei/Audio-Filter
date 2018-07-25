from __future__ import division

import wave
import math
import sys
import argparse
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
	# creates a 1d filter of size window_size, like a 1d edge detector, looks like: [-1,-1, 4,-1,-1]
	# does a 1d convolvution with channel
	array = np.full(window_size, -1, dtype='float64')
	array[window_size//2] = window_size - 1
	return np.convolve(channel, array, 'valid') / window_size

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
	# open wav
	in_wav = wave.open(in_file, 'rb')

	# get info about wav
	frame_rate = in_wav.getframerate()
	sample_width = in_wav.getsampwidth()
	n_channels = in_wav.getnchannels()
	n_frames = in_wav.getnframes()
	comp_type = in_wav.getcomptype()
	comp_name = in_wav.getcompname()
	signal = in_wav.readframes(n_frames * n_channels)
	in_wav.close()

	# get channels of wav
	channels = translate_wav(signal, n_frames, n_channels, sample_width)
	# filter wav with the filter type
	filtered_wav = moving_average_filter(cutoff_frequency, frame_rate, channels, filter_type)
	
	# write to new wav
	filtered_wav_file = wave.open(out_file, "w")
	filtered_wav_file.setparams((1, sample_width, frame_rate, n_frames, comp_type, comp_name))
	filtered_wav_file.writeframes(filtered_wav.tobytes('C'))
	filtered_wav_file.close()


def main(args):
    filter_wav(args.input_file,
               args.output_file, 
               args.cut_freq, 
               args.filter_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter WAV file.')
    parser.add_argument('--input_file', type=str, help='input WAV file')
    parser.add_argument('--output_file', type=str, help='output WAV file')
    parser.add_argument('--cut_freq', type=int, help='cutoff frequency (Hz)', default=400)
    parser.add_argument('--filter_type', type=str, help='"low" or "high" pass filter', default='high')
    
    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(0)
    main(parser.parse_args())





