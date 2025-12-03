import soundfile as sf

file_path = "/no_backups/s1495/enhancement_inputs/ha/dev/dev_02ha.wav"

info = sf.info(file_path)
print("Samplerate:", info.samplerate)