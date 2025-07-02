import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as aF
from julius import ResampleFrac
import random
from scipy.io import wavfile
import numpy as np
import pyloudnorm as pyln
import pretty_midi
import sys
sys.path.append('/home/yutong/.local/lib/python3.11/site-packages')  # Replace with your actual path
from fluidsynth import Synth
from pretty_midi import PrettyMIDI, Instrument
from typing import Union, Optional

MAX_INT16 = 32768.0

def load_audio(filepath, start=None, end=None, load_mode='torchaudio'):

    if load_mode == 'torchaudio':
        waveform, _ = torchaudio.load(filepath, frame_offset=start, 
                                      num_frames=end-start)
    elif load_mode == 'scipy':
        # make use of mmap to access segment from large audio files
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html
        _, waveform = wavfile.read(filepath, mmap=True)
        waveform = torch.from_numpy(waveform[start:end]/MAX_INT16).float().unsqueeze(0)
        
    return waveform

def load_waveform(filepath, 
                  start=None,
                  tar_sr=None, 
                  tar_len=None, 
                  load_mode: str = 'torchaudio',
                  return_start=False):
    """

    Args:
        filepath (str): filepath to the audio file
        tar_sr (float): target sampling rate
        tar_len (float): target length in seconds

    Returns:
        torch tensor: 1D waveform 
    """

    audio_metadata = torchaudio.info(filepath)
    src_len = audio_metadata.num_frames
    src_sample_rate = audio_metadata.sample_rate
    if tar_len is not None:
        tar_len = int(tar_len * src_sample_rate)
        if start is None:
            start_frame = random.randint(0, src_len - tar_len) if src_len > tar_len else 0
            start = start_frame/src_sample_rate
        else:
            start_frame = int(start * src_sample_rate)
            if start_frame > src_len:
                print('start time exceeds audio length', filepath)
                # exit()
                # start_frame = 0
                return torch.zeros(int(tar_len / src_sample_rate * tar_sr))
        
        waveform = load_audio(filepath, start=start_frame, end=start_frame+tar_len, load_mode=load_mode)
        src_len = waveform.shape[-1]
        if src_len < tar_len:
            waveform = F.pad(waveform, (0, tar_len-src_len), 'constant', 0)
    else:
        waveform, _ = torchaudio.load(filepath)

    # if tar_len is not None:
        
    #     tar_len = int(tar_len * src_sample_rate)

    #     if src_len > tar_len:
    #         if start is None:
    #             start = random.randint(0, src_len - tar_len)
    #         else:
    #             start = int(start * src_sample_rate)
    #         waveform = load_audio(filepath, start=start, end=start+tar_len, load_mode=load_mode)

    #     else:
    #         waveform, _ = torchaudio.load(filepath)
    #         waveform = F.pad(waveform, (0, tar_len-src_len), 'constant', 0)

    # else:
    #     waveform, _ = torchaudio.load(filepath)

    if tar_sr is not None and src_sample_rate != tar_sr:
        waveform = ResampleFrac(src_sample_rate, tar_sr)(waveform)

    if return_start:
        return waveform[0], start
    else:
        return waveform[0]

def add_reverb_noise(audio, reverb, noise, snr_db, target_len):
    """
    Add noise and reverberation

    Args:
        audio (_type_): _description_
        reverb (_type_): _description_
        noise (_type_): _description_
        snr_db (_type_): _description_
        target_len (_type_): _description_

    Returns:
        _type_: _description_
    """ 
    
    noisy_speech = aF.add_noise(audio.unsqueeze(0), noise.unsqueeze(0), snr_db)
    reverb = reverb / torch.linalg.vector_norm(reverb, ord=2)
    # reverb = reverb / reverb.abs().max()
    reverb_noisy_speech = aF.fftconvolve(noisy_speech.squeeze(0), reverb)
    
    # reverb_noisy_speech = reverb_noisy_speech / torch.linalg.vector_norm(reverb_noisy_speech, ord=2) * torch.linalg.vector_norm(noisy_speech, ord=2)
    
    if len(reverb_noisy_speech) > target_len:
        reverb_noisy_speech = reverb_noisy_speech[:target_len]

    return reverb_noisy_speech

def midi_to_piano_roll(midi_file, start_time, tar_len, fs, augment=False, midi=None):
    # Load the MIDI file
    midi_data = midi if midi is not None else pretty_midi.PrettyMIDI(midi_file)
    if augment:
        midi_data = augment_midi(midi_data)
    
    # tar_len_frame = int(tar_len * fs)

    # Define the start and end times of the time segment
    # end_time = start_time + tar_len

    # Get the piano roll for the specified time range
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return torch.tensor(piano_roll, dtype=torch.float32)


    # Convert the start and end times to frame indices
    start_frame = int(start_time * fs)
    end_frame = int(end_time * fs)

    # Slice the piano roll to the specified time segment
    piano_roll_segment = piano_roll[:, start_frame:end_frame]
    piano_roll_segment = torch.tensor(piano_roll_segment, dtype=torch.float32)

    if piano_roll_segment.shape[-1] < tar_len_frame:
        piano_roll_segment = F.pad(piano_roll_segment, (0, tar_len_frame-piano_roll_segment.shape[-1]), 'constant')

    return torch.transpose(piano_roll_segment, 0, 1)

def augment_midi(midi_data, num_note,
                 max_octave_shift=1, 
                 max_pitch_bend=20, 
                 max_time_shift=0.03, 
                 velocity_jitter_range=5,
                 octave_shift_prob=0.5,
                 pitch_bend_prob=0.5,
                 time_shift_prob=0.4,
                 velocity_jitter_prob=0.3,
                 note_drop_prob=0.5,
                 use_note_drop_prob=0.7):
    
    """Augment MIDI data by applying pitch shifts, time shifts, and velocity modulations."""
    low_pitch = 200
    high_pitch = -1
    for instrument in midi_data.instruments:
        notes = []
        current_time_shift = 0.0
        random_float = np.random.rand()
        drop_note = True if random_float < use_note_drop_prob else False
            
        for note in instrument.notes:
            
            random_floats = np.random.rand(4)
            
            if num_note > 4 and drop_note:
                if random_floats[0] < note_drop_prob:
                    continue
             
            # Apply random pitch shift within range
            if random_floats[1] < pitch_bend_prob:
                pitch_bend = np.random.uniform(-max_pitch_bend, max_pitch_bend)
                bend_amount = pitch_bend * 8192 / 100  # Convert cents to pitch bend
                pitch_bend = pretty_midi.PitchBend(int(bend_amount), note.start)
                instrument.pitch_bends.append(pitch_bend)

            # Apply random time shift within range (in seconds)
            if random_floats[2] < time_shift_prob:
                start_time_shift = np.random.uniform(-max_time_shift, max_time_shift)
                end_time_shift = np.random.uniform(-max_time_shift, max_time_shift)
                note, time_shift = apply_time_shift(note, 
                                                    start_time_shift, 
                                                    end_time_shift,
                                                    current_time_shift)
                current_time_shift += time_shift

            # Apply random velocity modulation
            if random_floats[3] < velocity_jitter_prob:
                velocity_jitter = np.random.randint(-velocity_jitter_range, velocity_jitter_range + 1)
                note = apply_velocity_modulation(note, velocity_jitter)
            
            notes.append(note)
            if note.pitch > high_pitch:
                high_pitch = note.pitch
            if note.pitch < low_pitch:
                low_pitch = note.pitch
            
        random_float = np.random.rand()
        use_octave_shift = random_float < octave_shift_prob
        if use_octave_shift and len(notes) > 0:
            octave_shift = np.random.randint(-max_octave_shift, max_octave_shift + 1)
            notes = apply_octave_shift(notes, octave_shift)
        
        instrument.notes = notes

    return midi_data, low_pitch, high_pitch

def apply_octave_shift(notes, octave_shift):
    note_pitch = []
    for note in notes:
        note_pitch.append(note.pitch)
    max_note = max(note_pitch)
    min_note = min(note_pitch)
    if max_note + octave_shift * 12 <= 127 or min_note + octave_shift * 12 >= 0:
        new_notes = []
        for note in notes:
            new_note = apply_pitch_shift(note, octave_shift*12)
            new_notes.append(new_note)
    return new_notes
    
def apply_pitch_shift(note, semitone_shift):
    """Shift note pitch by a specified number of semitones."""
    note.pitch = max(0, min(127, note.pitch + semitone_shift))  # MIDI pitch range: 0-127
    return note

def apply_time_shift(note, start_time_shift, end_time_shift, current_time_shift):
    """Shift note start and end times by a specified amount."""
    shift_time = 0.0
    new_start_time = start_time_shift + note.start + current_time_shift
    if new_start_time < 0:
        new_start_time = 0.0
        start_time_shift = (- note.start) - current_time_shift
    new_end_time = note.end + end_time_shift + current_time_shift
    if new_end_time > new_start_time:
        shift_time = start_time_shift
        note.end = new_end_time
        note.start = new_start_time
    return note, shift_time

def apply_velocity_modulation(note, velocity_jitter):
    """Modulate note velocity to simulate dynamic changes."""
    note.velocity = max(0, min(127, note.velocity + velocity_jitter))  # MIDI velocity range: 0-127
    return note

def midi_is_silent(pretty_midi_object):
    # Check each instrument in the PrettyMIDI object
    for instrument in pretty_midi_object.instruments:
        # If the instrument has any notes, it's not silent
        if len(instrument.notes) > 0:
            return False
    # If no instruments have notes, the object is silent
    return True

def extract_midi_segment(midi_file, tar_len, melody_only=True, start_time=None, augment=False):
    # Create a new PrettyMIDI object
    new_midi = pretty_midi.PrettyMIDI()
    pretty_midi_obj = pretty_midi.PrettyMIDI(midi_file)
    
    if start_time is None:
        all_notes = []
        for instrument in pretty_midi_obj.instruments:
            for note in instrument.notes:
                all_notes.append(note)
                
        if all_notes:
            selected_note = random.choice(all_notes)
            start_time = selected_note.start 
        else:
            # no notes in this instrument
            start_time = random.uniform(10, 30)
            
    end_time = start_time + tar_len
    
    low_pitch = 200
    high_pitch = -1
    # Iterate over all instruments in the original PrettyMIDI object
    for instrument in pretty_midi_obj.instruments:
        # Create a new instrument to hold the sliced notes
        new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        
        # Filter notes within the start and end time
        for note in instrument.notes:
            # Determine the part of the note that lies within the range
            note_start = max(note.start, start_time)
            note_end = min(note.end, end_time)
            
            # Include the note only if its adjusted duration lies within the time range
            if note_end > note_start:
                # Create a new note adjusted to fit within the time range
                new_note = pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=note_start - start_time,  # Shift the start time relative to the new MIDI's time
                    end=note_end - start_time        # Shift the end time relative to the new MIDI's time
                )
                new_instrument.notes.append(new_note)
                if note.pitch > high_pitch:
                    high_pitch = note.pitch
                if note.pitch < low_pitch:
                    low_pitch = note.pitch
        
        # Add the new instrument with filtered notes to the new PrettyMIDI object
        new_midi.instruments.append(new_instrument)
    
    
    if melody_only:
        low_pitch = 200
        high_pitch = -1
        melody_midi = pretty_midi.PrettyMIDI()
        for instrument in new_midi.instruments:
            melody_notes = []
            new_instrument = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        
            instrument.notes.sort(key=lambda note: (note.start, -note.pitch))
            current_time = -np.inf
            for note in instrument.notes:
                # If the note is the first in the current time step, add it as the melody
                if note.start > current_time:
                    melody_notes.append(note)
                    if note.pitch > high_pitch:
                        high_pitch = note.pitch
                    if note.pitch < low_pitch:
                        low_pitch = note.pitch
                    current_time = note.start
            new_instrument.notes = melody_notes
            melody_midi.instruments.append(new_instrument)
        new_midi = melody_midi
    
    
    insts: list[Instrument] = new_midi.instruments
    num_note = sum([len(i.notes) for i in insts])

    if len(insts) == 0 or num_note == 0:
        return new_midi, [low_pitch, high_pitch]
        
    if augment:
        drop_note_prob = 0 if melody_only else 0.7
        new_midi, low_pitch, high_pitch = augment_midi(new_midi, num_note, use_note_drop_prob=drop_note_prob)
    
    return new_midi, [low_pitch, high_pitch], start_time


def split_midi_by_interval(midi_file, tar_len, melody_only=True, augment=False):
    # List to store the resulting PrettyMIDI objects
    midi_obj = pretty_midi.PrettyMIDI(midi_file)
    split_midi_objs = []

    # Get all the notes from the original MIDI object
    all_notes = []
    for instrument in midi_obj.instruments:
        all_notes.extend(instrument.notes)
        
    # Sort notes by their start time
    all_notes.sort(key=lambda note: note.start)

    # Initialize variables for splitting
    current_time = all_notes[0].start
    current_notes = []
    low_pitch = 200
    high_pitch = -1
    pitch_range_list = []
    start_time_list = []

    for note in all_notes:
        # If the note starts after the current interval, create a new PrettyMIDI object
        if note.start >= current_time + tar_len:
            # Create a new PrettyMIDI object
            new_midi = pretty_midi.PrettyMIDI()
            new_instrument = pretty_midi.Instrument(program=midi_obj.instruments[0].program, is_drum=midi_obj.instruments[0].is_drum, name=midi_obj.instruments[0].name)
            for n in current_notes:
                new_note = pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start - current_notes[0].start,  # Shift the start time relative to the new MIDI's time
                    end=n.end - current_notes[0].start        # Shift the end time relative to the new MIDI's time
                )
                new_instrument.notes.append(new_note)
            
            current_time = note.start
            start_time_list.append(current_notes[0].start)
            new_midi.instruments.append(new_instrument)
            if augment:
                drop_note_prob = 0 if melody_only else 0.7
                insts: list[Instrument] = new_midi.instruments
                num_note = sum([len(i.notes) for i in insts])
                new_midi, low_pitch, high_pitch = augment_midi(new_midi, num_note, use_note_drop_prob=drop_note_prob)
    
            split_midi_objs.append(new_midi)
            
            # Reset for the next interval
            current_notes = []
            
            pitch_range_list.append([low_pitch, high_pitch])
            low_pitch = 200
            high_pitch = -1
            
        # Add the note to the current interval
        current_notes.append(note)
        if note.pitch > high_pitch:
            high_pitch = note.pitch
        if note.pitch < low_pitch:
            low_pitch = note.pitch

    # Add the last segment if there are any remaining notes
    if current_notes:
        new_midi = pretty_midi.PrettyMIDI()
        new_instrument = pretty_midi.Instrument(program=0)
        new_instrument.notes = current_notes
        new_midi.instruments.append(new_instrument)
        split_midi_objs.append(new_midi)
        start_time_list.append(current_notes[0].start)
        pitch_range_list.append([low_pitch, high_pitch])
        

    return split_midi_objs, pitch_range_list, start_time_list


def fluidsynth_midi(
    midi: Union[str, PrettyMIDI],
    S: Synth,
    sfid: int,
    tar_len: float,
    sr=16000,
    channel=0,
    program: Optional[int] = None,
    controller: Optional[int] = None,
):
    """
    Synthesize MIDI from a file or `PrettyMIDI` using `fluidsynth.Synth` and a soundfont ID from `fluidsynth.Synth.sfload`

    Based off of `PrettyMIDI.fluidsynth`
    """

    M = PrettyMIDI(midi) if isinstance(midi, str) else midi
    tar_len = int(tar_len * sr)

    insts: list[Instrument] = M.instruments
    
    

    if len(insts) == 0 or all(len(i.notes) == 0 for i in insts):
        print('no notes')
        return torch.zeros(tar_len, dtype=torch.float32)

    program = program or 0
    controller = controller or 0

    S.program_select(channel, sfid, program, controller)

    waveforms = []
    for i in insts:
        S.program_change(0, program)

        event_list = []
        for note in i.notes:
            event_list += [[note.start, "note on", note.pitch, note.velocity]]
            event_list += [[note.end, "note off", note.pitch]]
        for bend in i.pitch_bends:
            event_list += [[bend.time, "pitch bend", bend.pitch]]
        for control_change in i.control_changes:
            event_list += [
                [
                    control_change.time,
                    "control change",
                    control_change.number,
                    control_change.value,
                ]
            ]

        # Sort the event list by time, and secondarily by whether the event is a note off
        event_list.sort(key=lambda x: (x[0], x[1] != "note off"))
        # Add some silence at the beginning according to the time of the first event
        current_time = event_list[0][0]
        # Convert absolute seconds to relative samples
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        # Include 1 second of silence at the end
        event_list[-1][0] = 1.0
        # Pre-allocate output array
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros(int(np.ceil(sr * total_time)))
        
        for event in event_list:
            # Process events based on type
            if event[1] == "note on":
                S.noteon(channel, event[2], event[3])
            elif event[1] == "note off":
                S.noteoff(channel, event[2])
            elif event[1] == "pitch bend":
                S.pitch_bend(channel, event[2])
            elif event[1] == "control change":
                S.cc(channel, event[2], event[3])
            # Add in these samples
            current_sample = int(sr * current_time)
            end = int(sr * (current_time + event[0]))
            samples = S.get_samples(end - current_sample)[::2]
            synthesized[current_sample:end] += samples
            # Increment the current sample
            current_time += event[0]

        waveforms.append(synthesized)

    # Allocate output waveform, with #sample = max length of all waveforms
    synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))# Sum all waveforms in
    for w in waveforms:
        synthesized[: w.shape[0]] += w
    
    src_len = synthesized.shape[-1]
    if src_len < tar_len:
        synthesized = np.pad(synthesized, (0, tar_len - src_len), mode='constant', constant_values=0)
    else:
        synthesized = synthesized[:tar_len]
    
    synthesized = synthesized/np.max(np.abs(synthesized) + 1e-10)
        
    return torch.tensor(synthesized, dtype=torch.float32)

def wav_to_mel(wav, window, stft_args, mel_filterbank):
    spec = torch.stft(wav, window=window, normalized=True, 
                    return_complex=True, **stft_args)
    magnitude = torch.abs(spec)
    mel_spec = mel_filterbank(magnitude)
    mel_spec = mel_spec / torch.max(mel_spec)
    return mel_spec
    


class HighPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1 / 6, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 5 / 6,
                        1 / 1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, :int((nfft//2+1) * r)] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          )#return_complex=False)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1,1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        )#return_complex=False)
        x = x[:, :T].detach()
        return x
    

class LowPass(nn.Module):
    def __init__(self,
                 nfft=1024,
                 hop=256,
                 ratio=(1/6, 1/3, 1/2, 2/3, 3/4, 4/5, 5/6, 1/1)):
        super().__init__()
        self.nfft = nfft
        self.hop = hop
        self.register_buffer('window', torch.hann_window(nfft), False)
        f = torch.ones((len(ratio), nfft//2 + 1), dtype=torch.float)
        for i, r in enumerate(ratio):
            f[i, int((nfft//2+1) * r):] = 0.
        self.register_buffer('filters', f, False)

    #x: [B,T], r: [B], int
    @torch.no_grad()
    def forward(self, x, r):
        if x.dim()==1:
            x = x.unsqueeze(0)
        T = x.shape[1]
        x = F.pad(x, (0, self.nfft), 'constant', 0)
        stft = torch.stft(x,
                          self.nfft,
                          self.hop,
                          window=self.window,
                          return_complex=True)  #[B, F, TT,2]
        stft *= self.filters[r].view(*stft.shape[0:2],1 )
        x = torch.istft(stft,
                        self.nfft,
                        self.hop,
                        window=self.window,
                        return_complex=False)
        x = x[:, :T].detach()
        return x

class SegmentMixer(nn.Module):

    """
    https://github.com/Audio-AGI/AudioSep/blob/main/data/waveform_mixers.py

    
    """
    def __init__(self, max_mix_num, lower_db, higher_db):
        super(SegmentMixer, self).__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

    def __call__(self, waveforms, noise_waveforms):
        
        batch_size = waveforms.shape[0]
        noise_indices = torch.randperm(batch_size)

        data_dict = {
            'segment': [],
            'mixture': [],
        }

        for n in range(batch_size):

            segment = waveforms[n].clone()

            # random sample from noise waveforms
            noise = noise_waveforms[noise_indices[n]]
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            mix_num = random.randint(2, self.max_mix_num)
            assert mix_num >= 2

            for i in range(1, mix_num):
                next_segment = waveforms[(n + i) % batch_size]
                rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=segment, **self.loudness_param)
                noise += rescaled_next_segment

            # randomly normalize background noise
            noise = dynamic_loudnorm(audio=noise, reference=segment, **self.loudness_param)

            # create audio mixyure
            mixture = segment + noise

            # declipping if need be
            max_value = torch.max(torch.abs(mixture))
            if max_value > 1:
                segment *= 0.9 / max_value
                mixture *= 0.9 / max_value

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        return data_dict['segment'], data_dict['mixture']


def rescale_to_match_energy(segment1, segment2):

    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1 


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):

    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    delta_loudness = random.randint(lower_db, higher_db)
    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio

# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = audio.squeeze(0).detach().cpu().numpy()
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = torch.from_numpy(normalized_audio).unsqueeze(0)
    
    return normalized_audio.to(device)
    