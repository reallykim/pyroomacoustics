"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse
from aiosignal import Signal

import random
import matplotlib.pyplot as plt
import math
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import librosa
import librosa.display
import seaborn as sns
import time

methods = ["ism", "hybrid", "anechoic"]

perform_srp_phat = False

display_distribution = False
display_rir = False
display_waveform = False
display_melspectrogram = False
display_plt = any(
    [display_distribution, display_rir, display_waveform, display_melspectrogram]
)

# Room dimensions
room_dim = [20, 20, 5]  # meters

# Locations of the microphone array
offset_size = 0.5
mic_offsets = [
    [offset_size, offset_size, 0],
    [-offset_size, offset_size, 0],
    [-offset_size, -offset_size, 0],
    [offset_size, -offset_size, 0],
]
num_mics = len(mic_offsets)
offsets = np.array(mic_offsets)
origin = np.array([10, 10, 0])
mic_locs = origin + offsets

# Locations of the sources
source_list = [[20, 0, 5], [20, 20, 2], [0, 20, 1.5], [0, 0, 3]]
random.shuffle(source_list)
sources = np.array(source_list)

num_sources = len(source_list)

delays = [float(random.randrange(1, 5)) for _ in range(num_sources)]

background_origin = np.array([10, 10, 5])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_azimuth(deg):
    return (deg + 180) % 360 - 180


def resample_and_load(wav_file, sample_ratio, mono=True, is_noise=False):
    _, tmp_audio = wavfile.read(wav_file)
    audio, _ = librosa.load(wav_file, sr=sample_ratio, mono=mono)
    if is_noise:
        duration = len(audio) / sample_ratio
        start_t = random.randrange(0, int(duration) - 10)
        audio = audio[start_t * sample_ratio : (start_t + 10) * sample_ratio]
    magnitude = max(int(tmp_audio.max()), int(-tmp_audio.min()))
    return (audio * magnitude).astype(np.int16)


def ReadWavFiles(path, fs=16000, is_noise=False):
    audios = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            path_to_wav = line.strip()
            if not path_to_wav.endswith(".wav"):
                continue

            if librosa.get_samplerate(path_to_wav) == 16000:
                _, audio = wavfile.read(path_to_wav)
                if is_noise:
                    duration = len(audio) / 16000
                    start_t = random.randrange(0, int(duration) - 10)
                    audio = audio[start_t * 16000 : (start_t + 10) * 16000]
            else:
                audio = resample_and_load(path_to_wav, fs, is_noise=is_noise)
            audios.append(audio)

    return audios


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Simulates and adds reverberation to a dry sound sample. Saves it into `./examples/samples`."
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=methods,
        default=methods[0],
        help="Simulation method to use",
    )
    args = parser.parse_args()

    # The desired reverberation time and dimensions of the room
    rt60_tgt = 0.3  # seconds
    fs = 16000
    nfft = 256

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    audios = ReadWavFiles(path="examples/sound_sources.list", fs=fs)
    audios = random.sample(audios, num_sources)

    background_audios = ReadWavFiles(
        path="examples/background_sources.list", fs=fs, is_noise=True
    )
    background_audios = random.sample(background_audios, 1)

    if display_distribution:
        f, axes = plt.subplots(
            1, len(audios), figsize=(7 * len(audios), 7), sharex=True
        )
        for i in range(len(audios)):
            sns.distplot(audios[i], color="skyblue", ax=axes[i])
        plt.show()

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    # Use Sabine's formula to find the wall energy absorption and maximum order of the
    # ISM required to achieve a desired reveberation time (RT60, i.e. the time it takes
    # for the RIR to decay by 60db)

    # Create the room
    if args.method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order,
        )
    elif args.method == "hybrid":
        room = pra.ShoeBox(
            room_dim,
            fs=fs,
            materials=pra.Material(e_absorption),
            max_order=3,
            ray_tracing=True,
            air_absorption=True,
        )
    elif args.method == "anechoic":
        room = pra.AnechoicRoom(fs=fs)

    x = (sources - origin)[0, :num_sources]
    y = (sources - origin)[1, :num_sources]
    z = (sources - origin)[2, :num_sources]

    dir = sources - origin

    for i in range(num_sources):
        azimuth = angle_between(tuple(dir[i]), (0, 1, 0)) * 180 / np.pi
        dir_proj = (dir[i][0], dir[i][1], 0)
        elevation = angle_between(tuple(dir[i]), dir_proj) * 180 / np.pi
        if dir[i][0] < 0:
            azimuth = -azimuth
        if dir[i][2] < 0:
            elevation = -elevation
        print(
            f"(azimuth, elevation) to sources(x={sources[i][0]},y={sources[i][1]}): {azimuth, elevation}"
        )
        room.add_source(
            sources[i], signal=audios[i], delay=delays[i],
        )

    room.add_source(background_origin, signal=background_audios[0])

    mic_locs = np.c_[mic_locs[0], mic_locs[1], mic_locs[2], mic_locs[3]]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        f"/home/sykim/Documents/synthetic_ssl_sounds/generated_{args.method}_{int(time.time())}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    if perform_srp_phat:
        # Perform SRP-PHAT
        X = pra.transform.stft.analysis(
            room.mic_array.signals.T, nfft, nfft // 2, win=np.hanning(nfft)
        )
        X = np.swapaxes(X, 2, 0)

        # perform DOA estimation
        doa = pra.doa.algorithms["SRP"](mic_locs, fs, nfft)
        doa.locate_sources(X)

        # evaluate result
        print(doa.azimuth_recon.shape)
        if doa.azimuth_recon.shape[0] != 0:
            print("Source is estimated at:", math.degrees(doa.azimuth_recon))
        else:
            print("Cannot extract the azimuth!")

    # measure the reverberation time
    # rt60 = room.measure_rt60()
    # print("The desired RT60 was {}".format(rt60_tgt))
    # print("The measured RT60 is {}".format(rt60[1, 0]))

    # RIR = Room Impulse Response
    # plot the RIRs
    if display_rir:
        select = None  # plot all RIR
        # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
        # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
        fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
        fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
        fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

    if display_waveform:
        fig, axs = plt.subplots(num_mics, 1, figsize=(10, 12))

        ymin = np.min(room.mic_array.signals)
        ymax = np.max(room.mic_array.signals)

        for i in range(num_mics):
            axs[i].plot(room.mic_array.signals[i, :])
            axs[i].set_ylim([ymin, ymax])
            # axs[i].set(title=f"Microphones-{i}")
        plt.xlabel("Time [s]")

    if display_melspectrogram:
        fig, axs = plt.subplots(num_mics, 1, figsize=(10, 12))
        for i in range(num_mics):
            signal = room.mic_array.signals[i, :].astype(np.float32)
            mag = max(np.max(signal), -np.min(signal))
            signal /= mag

            S = librosa.feature.melspectrogram(
                y=signal,
                sr=fs,
                n_mels=128,
                # n_fft=int(fs * 0.016),
                # hop_length=int(fs * 0.01),
                # fmax=8000,
            )

            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(
                S_dB, x_axis="time", y_axis="mel", sr=fs, fmax=9000, ax=axs[i]
            )
            fig.colorbar(img, ax=axs[i], format="%+2.0f dB")
            axs[i].set(title=f"Mel-Spectrogram (mic-{i+1})")

    if display_plt:
        plt.tight_layout()
        plt.show()
