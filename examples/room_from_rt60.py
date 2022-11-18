"""
This example creates a room with reverberation time specified by inverting Sabine's formula.
This results in a reverberation time slightly longer than desired.
The simulation is pure image source method.
The audio sample with the reverb added is saved back to `examples/samples/guitar_16k_reverb.wav`.
"""
import argparse

import matplotlib.pyplot as plt
import math
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import librosa
import seaborn as sns

methods = ["ism", "hybrid", "anechoic"]
display_distribution = True


def resample_and_load(wav_file, sample_ratio, mono=True):
    _, tmp_audio = wavfile.read(wav_file)
    audio, _ = librosa.load(wav_file, sr=sample_ratio, mono=mono)
    magnitude = max(int(tmp_audio.max()), int(-tmp_audio.min()))
    return (audio * magnitude).astype(np.int16)


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
    room_dim = [20, 20, 5]  # meters

    # import a mono wavfile as the source signal
    # the sampling frequency should match that of the room
    fs, audio1 = wavfile.read("examples/samples/guitar_16k.wav")
    audio2 = resample_and_load("examples/samples/hibye.wav", fs)
    audio3 = resample_and_load("examples/samples/surprise1.wav", fs)
    audios = [audio1, audio2, audio3]

    if display_distribution:
        f, axes = plt.subplots(
            1, len(audios), figsize=(7 * len(audios), 7), sharex=True
        )
        for i in range(len(audios)):
            sns.distplot(audios[i], color="skyblue", ax=axes[i])
        plt.show()

    # We invert Sabine's formula to obtain the parameters for the ISM simulator
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)

    # Create the room
    if args.method == "ism":
        room = pra.ShoeBox(
            room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order
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

    # place the microphone array
    offsets = np.array([[0.5, 0.5, 0], [-0.5, 0.5, 0], [-0.5, -0.5, 0], [0.5, -0.5, 0]])
    origin = np.array([10, 10, 0])
    mic_locs = origin + offsets

    sources = np.array([[20, 0, 0], [20, 20, 0], [0, 20, 0]])

    x = (sources - origin)[0, :2]
    y = (sources - origin)[1, :2]

    dir = sources - origin

    for i in range(2):
        azimuth = angle_between(tuple(dir[i]), (1, 0, 0)) * 180 / np.pi
        if dir[i][1] > 0:
            azimuth = -azimuth
        print("azimuth angles to sources: ", -azimuth)
        room.add_source(
            sources[i], signal=audios[i], delay=0.5,
        )

    mic_locs = np.c_[mic_locs[0], mic_locs[1], mic_locs[2], mic_locs[3]]

    # finally place the array in the room
    room.add_microphone_array(mic_locs)

    # Run the simulation (this will also build the RIR automatically)
    room.simulate()

    room.mic_array.to_wav(
        f"examples/samples/guitar_16k_reverb_{args.method}.wav",
        norm=True,
        bitdepth=np.int16,
    )

    nfft = 256
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
        print("Source is estimated at:", get_azimuth(math.degrees(doa.azimuth_recon)))
    else:
        print("Cannot extract the azimuth!")

    # measure the reverberation time
    # rt60 = room.measure_rt60()
    # print("The desired RT60 was {}".format(rt60_tgt))
    # print("The measured RT60 is {}".format(rt60[1, 0]))

    # plot the RIRs
    select = None  # plot all RIR
    # select = (2, 0)  # uncomment to only plot the RIR from mic 2 -> src 0
    # select = [(0, 0), (2, 0)]  # only mic 0 -> src 0, mic 2 -> src 0
    fig, axes = room.plot_rir(select=select, kind="ir")  # impulse responses
    fig, axes = room.plot_rir(select=select, kind="tf")  # transfer function
    fig, axes = room.plot_rir(select=select, kind="spec")  # spectrograms

    plt.tight_layout()
    plt.show()
