import numpy as np
import matplotlib.pyplot as plt
import fcwt
from barbutils import load_barb
import os

def calculate_cwt(signal, fs, f0, f1, fn):
    """Calculate CWT and its envelope."""
    freqs, cwt_matrix = fcwt.cwt(signal, int(fs), f0, f1, fn)
    cwt_envelope = np.max(np.abs(cwt_matrix), axis=0)
    return cwt_matrix, cwt_envelope, freqs

def extract_instantaneous_frequencies(cwt_matrix, freqs):
    """Extract instantaneous frequencies from CWT matrix."""
    peak_indices = np.argmax(np.abs(cwt_matrix), axis=0)
    instantaneous_frequencies = freqs[peak_indices]
    return instantaneous_frequencies

def merge_chirp_segments(matching_times, matching_frequencies, fs, max_gap=0.0001):
    """Merge consecutive matching segments into chirps."""
    if len(matching_times) == 0:
        return []
    
    chirp_segments = []
    current_segment = [(matching_times[0], matching_frequencies[0])]
    
    for i in range(1, len(matching_times)):
        if matching_times[i] - matching_times[i - 1] <= max_gap:
            current_segment.append((matching_times[i], matching_frequencies[i]))
        else:
            chirp_segments.append(current_segment)
            current_segment = [(matching_times[i], matching_frequencies[i])]
    
    chirp_segments.append(current_segment)
    return chirp_segments

def process_with_sliding_window(signal, fs, window_duration, step_size, f0, f1, fn, expected_chirp_params):
    """Process signal with a sliding window and find chirp matches."""
    window_size = int(window_duration * fs)
    step = int(step_size * fs)
    
    all_intervals = []
    all_instantaneous_freqs = []
    time = np.arange(len(signal)) / fs
    
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        segment = signal[start:end]
        
        # Calculate CWT for the segment
        cwt_matrix, cwt_envelope, freqs = calculate_cwt(segment, fs, f0, f1, fn)
        
        # Extract instantaneous frequencies
        instantaneous_frequencies = extract_instantaneous_frequencies(cwt_matrix, freqs)
        
        # Compare with expected chirp
        matching_times, matching_frequencies = compare_with_chirp(
            instantaneous_frequencies, expected_chirp_params, fs
        )
        
        # Shift matching times to the correct position in the original signal
        matching_times += start / fs
        
        all_intervals.extend(matching_times)
        all_instantaneous_freqs.extend(matching_frequencies)
    
    # Merge matching times and frequencies into continuous chirp segments
    chirp_segments = merge_chirp_segments(np.array(all_intervals), np.array(all_instantaneous_freqs), fs)
    
    return chirp_segments

def compare_with_chirp(instantaneous_frequencies, expected_chirp_params, fs):
    """Compare extracted frequencies with expected chirp parameters."""
    f0, f1, duration = expected_chirp_params
    num_samples = len(instantaneous_frequencies)
    time = np.arange(num_samples) / fs

    # Generate expected chirp
    chirp_signal = f0 + (f1 - f0) * (time / duration)
    
    # Find matching regions
    match_threshold = 0.1  # Adjust as needed
    matched_indices = np.abs(instantaneous_frequencies - chirp_signal) < match_threshold
    
    return time[matched_indices], instantaneous_frequencies[matched_indices]

# Example usage
script_path = os.path.dirname(os.path.realpath(__file__))
filename = os.path.join(script_path, "1834cs1.barb")

f0 = 18000
f1 = 34000
fn = 200
window_duration = 0.001  # Window duration in seconds
step_size = 0.0002       # Step size in seconds
expected_chirp_params = (f0, f1, 0.002)  # f0, f1, and duration of the chirp in seconds

with open(filename, "rb") as f:
    barb = f.read()

sample_rate, signal_data = load_barb(barb)
sample_rate = 1e6  # Assuming a sample rate of 1 MHz

# Process the signal with a sliding window
chirp_segments = process_with_sliding_window(
    signal_data, sample_rate, window_duration, step_size, f0, f1, fn, expected_chirp_params
)

# Plotting the results
time = np.arange(len(signal_data)) / 1e6
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, signal_data, label='Signal')
plt.title('Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
# plt.legend()

plt.subplot(2, 1, 2)
for segment in chirp_segments:
    segment_times, segment_freqs = zip(*segment)
    plt.plot(segment_times, segment_freqs, label='Detected Chirp Segment')
    
plt.title('Detected Chirp Segments')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.grid(True)
# plt.legend()

plt.tight_layout()
plt.show()
