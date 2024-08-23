using RedPitayaDAQServer
using Plots

include("config.jl")

# rp = RedPitaya(URLs[1])
# serverMode!(rp, CONFIGURATION)

# Signal parameters
fs = 1e6  # Sampling frequency
t1 = 0.004  # Chirp duration 4 ms
t2 = 0.002  # Chirp duration 2 ms
pause = 0.008  # Pause 8 ms
max_samples = 16384

# Chirp generation
function generate_chirp(f0, f1, t, fs)
    t = range(0, stop=t, length=Int(fs * t))
    return sin.(2 * π * (f0 .+ (f1 - f0) .* (t ./ t[end])) .* t)
end

# Function to generate the signal
function generate_signal(fs, t1, t2, pause)
    signal = Float64[]

    # First chirp 18-34 kHz, 4 ms, pause 8 ms
    append!(signal, generate_chirp(18000, 34000, t1, fs))
    append!(signal, zeros(Int(fs * pause)))

    # Second and third chirps 34-18 kHz, 4 ms, pause 8 ms
    for _ in 1:2
        append!(signal, generate_chirp(34000, 18000, t1, fs))
        append!(signal, zeros(Int(fs * pause)))
    end

    # Remaining chirps 34-18 kHz, 2 ms, no pause
    for _ in 1:129
        append!(signal, generate_chirp(34000, 18000, t2, fs))
    end

    return signal
end

# Signal generation
signal = generate_signal(fs, t1, t2, pause)

# Normalize the signal to the range [-1, 1]
# signal = signal ./ maximum(abs, signal)

# # Scale the signal to the range [-8192, 8192]
# signal = signal .* 8192

# # Convert the signal to 16-bit signed integers
# signal = Int16.(signal)

# # Save the signal as a WAV file
# wavwrite(signal, fs, "1834_cs1.wav")

# # Save the signal as a TDMS file
# tdms_file = TDMS.TDMSFile("1834_cs1.tdms", mode="w")
# TDMS.add_group(tdms_file, "Group")
# TDMS.add_channel(tdms_file, "Group", "Signal", signal)
# TDMS.close(tdms_file)

# Display the resulting signal
# plot(signal, title="Resulting Signal", xlabel="Sample", ylabel="Amplitude")
plot(signal, title="Generated Signal", xlabel="Sample", ylabel="Amplitude")