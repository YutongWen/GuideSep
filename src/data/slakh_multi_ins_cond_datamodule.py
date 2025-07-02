from typing import Any, Dict, Optional
import os
import torch
import random
import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import random
from .audio_processing_utils import load_waveform, fluidsynth_midi, extract_midi_segment, wav_to_mel, split_midi_by_interval
from tqdm import tqdm
import sys
sys.path.append('/home/yutong/.local/lib/python3.11/site-packages')  # Replace with your actual path
from fluidsynth import Synth
import xml.etree.ElementTree as ET
import torchaudio
import numpy as np
from scipy.ndimage import gaussian_filter
import glob
from scipy.stats import entropy

def find_element_containing_substring(l, substring):
    return [element for element in l if substring in element]

def signal_entropy(wav):
    _, counts = np.unique(np.array(wav), return_counts=True)
    probabilities = counts / counts.sum()
    entropy_value = entropy(probabilities)
    return entropy_value

def extract_program_and_controller(instrument_element):
    channels = instrument_element.findall('Channel')  # Get all channels
    
    selected_channel = random.choice(channels)  # Randomly select a channel
    ctrl_value = selected_channel.find('controller').attrib['value']
    program_value = selected_channel.find('program').attrib['value']
    channel_name = selected_channel.get('name', 'default')  # Get channel name, default if not present

    return int(ctrl_value), int(program_value), channel_name

def mel_spec_mask(mel_spec, threshold=0.12, max_mask_drop_prob=0.4, brush_size=5, min_filter_sig=4, max_filter_sig=6):
    mask = (mel_spec > threshold).astype(np.float32)
    masked_indices = np.argwhere(mask == 1)
    mask_drop_prob = random.uniform(0, max_mask_drop_prob)
    num_samples = int(len(masked_indices) * mask_drop_prob)
    sampled_indices = masked_indices[np.random.choice(len(masked_indices), num_samples, replace=False)]
    
    for index in sampled_indices:  # The number of brush strokes depends on noise_level
        i = index[0]
        j = index[1]
        
        # Define the brush stroke as a block/patch around the point
        i_min = max(i - brush_size // 2, 0)
        i_max = min(i + brush_size // 2, mask.shape[0])
        j_min = max(j - brush_size // 2, 0)
        j_max = min(j + brush_size // 2, mask.shape[1])
        
        # Flip the values in the block (0 -> 1 or 1 -> 0)
        mask[i_min:i_max, j_min:j_max] = 0
    sigma = random.uniform(min_filter_sig, max_filter_sig)
    mask = gaussian_filter(mask, sigma=sigma)
    return torch.tensor(mask, dtype=torch.float32)

class SlakhMultiInsDataset(Dataset):
    def __init__(self, 
                 track_path, 
                 sample_rate, 
                 tar_len,
                 hop_len,
                 moise_path='/mnt/data/Music/moisesdb/moisesdb_v0.1',
                 humming_path='/mnt/data/Music/HumTrans',
                 humming_prob=0.4,
                 org_mix_prob=0.3,
                 mode='train',
                 augment=True,
                 tar_ins=None):
        super().__init__()    
        self.track_paths = []
        self.audio_filenames = {}
        self.sample_rate = sample_rate
        self.tar_len = tar_len
        self.hop_len = hop_len
        self.augment = augment
        self.mode = mode
        self.n_fft = (self.hop_len - 1) * 2
        self.stft_args = dict(n_fft=self.n_fft, hop_length=self.hop_len, center=True)
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sample_rate, n_stft=self.n_fft // 2 + 1)
        self.humming_prob = humming_prob
        self.org_mix_prob = org_mix_prob
        self.tar_ins = tar_ins
        
        hum_midi_filenames = glob.glob(os.path.join(humming_path, 'midi_data', '*.mid'))
        self.humming_midi_tracks = {}
        for filename in hum_midi_filenames:
            track_id = filename.split('/')[-1].split('_')[1]
            if track_id not in self.humming_midi_tracks.keys():
                self.humming_midi_tracks[track_id] = [filename]
            else:
                self.humming_midi_tracks[track_id].append(filename)
        
        self.humming_choise = np.array(list(self.humming_midi_tracks.keys()))
        for key in self.humming_choise:
            self.humming_midi_tracks[key] = np.array(self.humming_midi_tracks[key])
            
            
        moise_track_paths = [os.path.join(moise_path, name) for name in os.listdir(moise_path) if os.path.isdir(os.path.join(moise_path, name))]
        self.moise_track_stems = {}
        for track in moise_track_paths:
            stems = [os.path.join(track, name) for name in os.listdir(track) if os.path.isdir(os.path.join(track, name))]
            for stem in stems:
                stem_key = stem.split('/')[-1]
                sources = glob.glob(os.path.join(stem, '*.wav'))
                if stem_key not in self.moise_track_stems.keys():
                    self.moise_track_stems[stem_key] = sources
                else:
                    self.moise_track_stems[stem_key] += sources  
        
        self.moise_choise = np.array(list(self.moise_track_stems.keys())) 
        for key in self.moise_choise:
            self.moise_track_stems[key] = np.array(self.moise_track_stems[key])   

        if mode == 'train':
            train_path = os.path.join(track_path, 'train')
            # val_path = os.path.join(track_path, 'validation')
            omit_path = os.path.join(track_path, 'omitted')
            track_paths = [train_path, omit_path]
        elif mode == 'val':
            track_paths = [os.path.join(track_path, 'validation')]
        else:
            track_paths = [os.path.join(track_path, 'test')]
            # track_paths = [os.path.join(track_path, 'validation')]
        
        for path in track_paths:
            self.track_paths += [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        
        self.instrument_list = []
        for track in tqdm(self.track_paths): 
            with open(os.path.join(track, 'metadata.yaml'), 'r') as file:
                metadata = yaml.safe_load(file)
            ins_list = []
            audio_datapoint = {}
            count = 0
            for track_id, info in metadata['stems'].items():
                if info['audio_rendered'] and info['inst_class'] != 'Drums':
                    
                    if info['inst_class'] not in self.instrument_list:
                        self.instrument_list.append(info['inst_class'])
                        
                    ins = info['inst_class'] + '_' + str(count)
                    ins_list.append(ins)
                    audio_datapoint[ins] = {'track': os.path.join(track, 'stems', track_id+'.flac'),
                                            'midi': os.path.join(track, 'MIDI', track_id+'.mid')}
                    count += 1
            audio_datapoint['mix'] = os.path.join(track, 'mix.flac')
            audio_datapoint['ins'] = ins_list
            self.audio_filenames[track] = audio_datapoint
        
        sf2_path = "AegeanSymphonicOrchestra-SND/AegeanSymphonicOrchestra-SND.sf2"
        self.S = Synth(samplerate=self.sample_rate)
        self.sfid = self.S.sfload(sf2_path)
        
        tree = ET.parse('AegeanSymphonicOrchestra-SND/ASO-SND-instruments.xml')
        root = tree.getroot()
        self.instruments = []
        for group in root.findall('InstrumentGroup'):
            for instrument in group.findall('Instrument'):
                long_name = instrument.find('longName').text  # Extract long name of the instrument
                p_pitch_range_element = instrument.find('pPitchRange')
                if p_pitch_range_element is not None and p_pitch_range_element.text is not None:
                    pitch_range = p_pitch_range_element.text
                    pitch_low = int(pitch_range.split('-')[0])
                    pitch_high = int(pitch_range.split('-')[1])
                else:
                    pitch_low = 0
                    pitch_high = 127
                self.instruments.append({
                    'element': instrument,
                    'name': long_name,
                    'pitch_range': [pitch_low, pitch_high]
                })
        
        
    def __len__(self):
        return len(self.track_paths)
    
    def load_residual(self):
        while True:
            ins_key_id = np.random.randint(low=0, high=len(self.moise_choise))
            ins_key = self.moise_choise[ins_key_id]
            s_filename_id = np.random.randint(low=0, high=len(self.moise_track_stems[ins_key]))
            s_filename = self.moise_track_stems[ins_key][s_filename_id]
            residual = load_waveform(filepath=s_filename, 
                                    tar_sr=self.sample_rate, 
                                    tar_len=self.tar_len)
            residual_entropy = signal_entropy(residual)
            if residual_entropy > 2:
                break
        return residual
    
        
    def __getitem__(self, idx):
        random_floats = np.random.rand(2)
        filename = self.track_paths[idx] 
        use_humming = random_floats[0] < self.humming_prob
        use_org_mix = random_floats[1] < self.org_mix_prob
        num_source = np.random.randint(low=3, high=6)

        if use_humming:
            key_id = np.random.randint(low=0, high=len(self.humming_choise))
            key = self.humming_choise[key_id]
            midi_filename_id = np.random.randint(low=0, high=len(self.humming_midi_tracks[key]))
            midi_filename = self.humming_midi_tracks[key][midi_filename_id]
            midi, pitch_range, start_time = extract_midi_segment(midi_file=midi_filename,
                                                                 tar_len=self.tar_len,
                                                                 augment=False)
            directory, filename = os.path.split(midi_filename)
            new_directory = directory.replace('midi_data', 'wav_data_sync_with_midi')
            new_filename = os.path.splitext(filename)[0] + '.wav'
            new_path = os.path.join(new_directory, new_filename)
            cond_signal = load_waveform(filepath=new_path,
                                        start=start_time,
                                        tar_sr=self.sample_rate,
                                        tar_len=self.tar_len)
            tar_ins = 'humming'
        else:
            if self.mode == 'test' and self.tar_ins is not None:
            # if False:
                tar_ins = self.tar_ins
                filtered_list = find_element_containing_substring(self.audio_filenames[filename]['ins'], tar_ins)
                if len(filtered_list) == 0:
                    return {'tar_signal': None, 
                            'cond_signal': None, 
                            'mel_spec_mask_tar': None, 
                            'mel_spec_mask_res': None, 
                            'residual': None, 
                            'tar_ins': None, 
                            'cond_ins_name': None, 
                            'mel_spec_tar': None, 
                            'mel_spec_res': None}
                tar_ins = np.random.choice(filtered_list)
            else:      
                tar_ins = np.random.choice(np.array(self.audio_filenames[filename]['ins']))
            midi, pitch_range, start_time = extract_midi_segment(midi_file=self.audio_filenames[filename][tar_ins]['midi'],
                                                                tar_len=self.tar_len,
                                                                augment=self.augment)
        
        instruments = self.instruments.copy()
        while True:
            random_idx = np.random.randint(low=0, high=len(instruments))
            selected_instrument = instruments[random_idx]
            ins_pitch_range = selected_instrument['pitch_range']
            if ins_pitch_range[0] <= pitch_range[0] and ins_pitch_range[1] >= pitch_range[1]:
                ctrl_value, program_value, channel_name = extract_program_and_controller(selected_instrument['element'])
                cond_ins_name = selected_instrument['name'] + '_' + channel_name
                
                midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                midi_to_audio = midi_to_audio / torch.max(torch.abs(midi_to_audio) + 1e-10)
                midi_entropy = signal_entropy(midi_to_audio)
                if midi_entropy < 1.5 and len(instruments) > 1:
                    instruments.remove(selected_instrument)
                else:
                    break
            else:
                if len(instruments) == 1:
                    midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                    break
                instruments.remove(selected_instrument)
        
        if use_humming:
            tar_signal = midi_to_audio
        else:
            cond_signal = midi_to_audio
            tar_signal = load_waveform(filepath=self.audio_filenames[self.track_paths[idx]][tar_ins]['track'],
                                            start=start_time,
                                            tar_sr=self.sample_rate,
                                            tar_len=self.tar_len)

        if (not use_org_mix and self.mode == 'train') or use_humming:
            residual = torch.stack([self.load_residual() for _ in range(num_source)])
            residual = torch.sum(residual, dim=0)
        else:
            mixture = load_waveform(filepath=self.audio_filenames[self.track_paths[idx]]['mix'],
                                            start=start_time,
                                            tar_sr=self.sample_rate,
                                            tar_len=self.tar_len)
            
            residual = mixture if use_humming else mixture - tar_signal

        cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        tar_signal = tar_signal / torch.max(torch.abs(tar_signal) + 1e-10)
        residual = residual / torch.max(torch.abs(residual) + 1e-10)
        
        # compute cond and res entropy
        cond_entropy = signal_entropy(cond_signal)
        cond_signal = tar_signal.clone() if cond_entropy < 1 else cond_signal
        residual_entropy = signal_entropy(residual)
        
        tar_entropy = signal_entropy(tar_signal)
        tar_signal = cond_signal.clone() if tar_entropy < 1 else tar_signal
        
        if cond_entropy < 1.5 and tar_entropy < 1.5:
            tar_signal = residual.clone()
            cond_signal = residual.clone()
            zero_res_mask = True
        else:
            zero_res_mask = False
            
        if cond_entropy < 1.5 and tar_entropy < 1.5 and residual_entropy < 1.5:
            cond_signal = load_waveform(filepath=os.path.join('template_input', 'cond.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            tar_signal = load_waveform(filepath=os.path.join('template_input', 'target.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            residual = load_waveform(filepath=os.path.join('template_input', 'residual.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        
        mel_spec_tar = wav_to_mel(wav=tar_signal,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_tar = mel_spec_mask(mel_spec=mel_spec_tar.numpy())
        mel_spec_mask_tar = mel_spec_mask_tar / torch.max(mel_spec_mask_tar + 1e-10)
        
        mel_spec_res = wav_to_mel(wav=residual,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_res = mel_spec_mask(mel_spec=mel_spec_res.numpy(),
                                          threshold=0.1)
        mel_spec_mask_res = mel_spec_mask_res / torch.max(mel_spec_mask_res + 1e-10)
        mel_spec_mask_res = torch.zeros_like(mel_spec_mask_res) if zero_res_mask else mel_spec_mask_res
        
        if self.mode == 'test':
            mel_spec_cond = wav_to_mel(wav=cond_signal,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
            mel_spec_mask_cond = mel_spec_mask(mel_spec=mel_spec_cond.numpy(),
                                                threshold=0.1,
                                                max_mask_drop_prob=0.0,
                                                min_filter_sig=5,
                                                max_filter_sig=5)
            mel_spec_mask_cond = mel_spec_mask_cond / torch.max(mel_spec_mask_cond + 1e-10)
            
            mel_spec_mix = wav_to_mel(wav=mixture,
                                    window=self.window,
                                    stft_args=self.stft_args,
                                    mel_filterbank=self.mel_filterbank)
            mel_spec_mask_mix = mel_spec_mask(mel_spec=mel_spec_mix.numpy(),
                                            threshold=0.1,
                                            max_mask_drop_prob=0.0,
                                            min_filter_sig=5,
                                            max_filter_sig=5)
            mel_spec_mask_mix = mel_spec_mask_mix / torch.max(mel_spec_mask_mix + 1e-10)
            
            return {'tar_signal': tar_signal, 
                'cond_signal': cond_signal, 
                'mel_spec_mask_tar': mel_spec_mask_tar, 
                'mel_spec_mask_res': mel_spec_mask_res, 
                'residual': residual, 
                'tar_ins': tar_ins, 
                'cond_ins_name': cond_ins_name, 
                'mel_spec_tar': mel_spec_tar, 
                'mel_spec_res': mel_spec_res,
                'mel_spec_mask_cond': mel_spec_mask_cond,
                'mel_spec_mask_mix': mel_spec_mask_mix}
            
        return {'tar_signal': tar_signal, 
                'cond_signal': cond_signal, 
                'mel_spec_mask_tar': mel_spec_mask_tar, 
                'mel_spec_mask_res': mel_spec_mask_res, 
                'residual': residual, 
                'tar_ins': tar_ins, 
                'cond_ins_name': cond_ins_name, 
                'mel_spec_tar': mel_spec_tar, 
                'mel_spec_res': mel_spec_res}
        
        
class SlakhMultiInsTestDataset(Dataset):
    def __init__(self, 
                 track_path, 
                 sample_rate, 
                 tar_len,
                 hop_len,
                 humming_path='/mnt/data/Music/HumTrans',
                 humming_prob=0.4,
                 org_mix_prob=0.3,
                 augment=True,
                 tar_ins=None):
        super().__init__()    
        self.track_paths = []
        self.audio_filenames = {}
        self.sample_rate = sample_rate
        self.tar_len = tar_len
        self.hop_len = hop_len
        self.augment = augment
        self.n_fft = (self.hop_len - 1) * 2
        self.stft_args = dict(n_fft=self.n_fft, hop_length=self.hop_len, center=True)
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sample_rate, n_stft=self.n_fft // 2 + 1)
        self.humming_prob = humming_prob
        self.org_mix_prob = org_mix_prob
        self.tar_ins = tar_ins
        
        hum_midi_filenames = glob.glob(os.path.join(humming_path, 'midi_data', '*.mid'))
        self.humming_midi_tracks = {}
        for filename in hum_midi_filenames:
            track_id = filename.split('/')[-1].split('_')[1]
            if track_id not in self.humming_midi_tracks.keys():
                self.humming_midi_tracks[track_id] = [filename]
            else:
                self.humming_midi_tracks[track_id].append(filename)
        
        self.humming_choise = np.array(list(self.humming_midi_tracks.keys()))
        for key in self.humming_choise:
            self.humming_midi_tracks[key] = np.array(self.humming_midi_tracks[key])
        
        track_paths = [os.path.join(track_path, 'test')]
        
        for path in track_paths:
            self.track_paths += [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        
        self.instrument_list = []
        for track in tqdm(self.track_paths): 
            with open(os.path.join(track, 'metadata.yaml'), 'r') as file:
                metadata = yaml.safe_load(file)
            ins_list = []
            audio_datapoint = {}
            count = 0
            for track_id, info in metadata['stems'].items():
                if info['audio_rendered'] and info['inst_class'] != 'Drums':
                    
                    if info['inst_class'] not in self.instrument_list:
                        self.instrument_list.append(info['inst_class'])
                        
                    ins = info['inst_class'] + '_' + str(count)
                    ins_list.append(ins)
                    audio_datapoint[ins] = {'track': os.path.join(track, 'stems', track_id+'.flac'),
                                            'midi': os.path.join(track, 'MIDI', track_id+'.mid')}
                    count += 1
            audio_datapoint['mix'] = os.path.join(track, 'mix.flac')
            audio_datapoint['ins'] = ins_list
            self.audio_filenames[track] = audio_datapoint
        
        sf2_path = "AegeanSymphonicOrchestra-SND/AegeanSymphonicOrchestra-SND.sf2"
        self.S = Synth(samplerate=self.sample_rate)
        self.sfid = self.S.sfload(sf2_path)
        
        tree = ET.parse('AegeanSymphonicOrchestra-SND/ASO-SND-instruments.xml')
        root = tree.getroot()
        self.instruments = []
        for group in root.findall('InstrumentGroup'):
            for instrument in group.findall('Instrument'):
                long_name = instrument.find('longName').text  # Extract long name of the instrument
                p_pitch_range_element = instrument.find('pPitchRange')
                if p_pitch_range_element is not None and p_pitch_range_element.text is not None:
                    pitch_range = p_pitch_range_element.text
                    pitch_low = int(pitch_range.split('-')[0])
                    pitch_high = int(pitch_range.split('-')[1])
                else:
                    pitch_low = 0
                    pitch_high = 127
                self.instruments.append({
                    'element': instrument,
                    'name': long_name,
                    'pitch_range': [pitch_low, pitch_high]
                })
        
        self.data = []
        # dict {filename, ins_name, start_time, pitch_range}
        for filename in self.track_paths:
            tar_ins = self.tar_ins
            filtered_list = find_element_containing_substring(self.audio_filenames[filename]['ins'], tar_ins)
            if len(filtered_list) == 0:
                continue
            for ins in filtered_list:
                midi_list, pitch_range_list, start_time_list = split_midi_by_interval(midi_file=self.audio_filenames[filename][ins]['midi'],
                                                                tar_len=self.tar_len,
                                                                augment=self.augment)
                for midi, pitch_range, start_time in zip(midi_list, pitch_range_list, start_time_list):
                    self.data.append({
                        'filename': filename,
                        'pitch_range': pitch_range,
                        'start_time': start_time,
                        'midi': midi,
                        'tar_ins': ins
                    })
         
    def __len__(self):
        return len(self.data)
    
    def load_residual(self):
        while True:
            ins_key_id = np.random.randint(low=0, high=len(self.moise_choise))
            ins_key = self.moise_choise[ins_key_id]
            s_filename_id = np.random.randint(low=0, high=len(self.moise_track_stems[ins_key]))
            s_filename = self.moise_track_stems[ins_key][s_filename_id]
            residual = load_waveform(filepath=s_filename, 
                                    tar_sr=self.sample_rate, 
                                    tar_len=self.tar_len)
            residual_entropy = signal_entropy(residual)
            if residual_entropy > 2:
                break
        return residual
    
        
    def __getitem__(self, idx):
        filename = self.data[idx]['filename']
        pitch_range = self.data[idx]['pitch_range']
        start_time = self.data[idx]['start_time']
        midi = self.data[idx]['midi']
        tar_ins = self.data[idx]['tar_ins']

        
        instruments = self.instruments.copy()
        while True:
            random_idx = np.random.randint(low=0, high=len(instruments))
            selected_instrument = instruments[random_idx]
            ins_pitch_range = selected_instrument['pitch_range']
            if ins_pitch_range[0] <= pitch_range[0] and ins_pitch_range[1] >= pitch_range[1]:
                ctrl_value, program_value, channel_name = extract_program_and_controller(selected_instrument['element'])
                cond_ins_name = selected_instrument['name'] + '_' + channel_name
                
                
                midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                midi_to_audio = midi_to_audio / torch.max(torch.abs(midi_to_audio) + 1e-10)
                midi_entropy = signal_entropy(midi_to_audio)
                if midi_entropy < 1.5 and len(instruments) > 1:
                    instruments.remove(selected_instrument)
                else:
                    break
            else:
                if len(instruments) == 1:
                    midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                    break
                instruments.remove(selected_instrument)
        
        cond_signal = midi_to_audio
        tar_signal = load_waveform(filepath=self.audio_filenames[filename][tar_ins]['track'],
                                        start=start_time,
                                        tar_sr=self.sample_rate,
                                        tar_len=self.tar_len)

        
        mixture = load_waveform(filepath=self.audio_filenames[filename]['mix'],
                                        start=start_time,
                                        tar_sr=self.sample_rate,
                                        tar_len=self.tar_len)
        
        residual = mixture - tar_signal

        cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        tar_signal = tar_signal / torch.max(torch.abs(tar_signal) + 1e-10)
        residual = residual / torch.max(torch.abs(residual) + 1e-10)
        
        # compute cond and res entropy
        cond_entropy = signal_entropy(cond_signal)
        cond_signal = tar_signal.clone() if cond_entropy < 1 else cond_signal
        residual_entropy = signal_entropy(residual)
        
        tar_entropy = signal_entropy(tar_signal)
        tar_signal = cond_signal.clone() if tar_entropy < 1 else tar_signal
        
        if cond_entropy < 1.5 and tar_entropy < 1.5:
            tar_signal = residual.clone()
            cond_signal = residual.clone()
            zero_res_mask = True
            # torchaudio.save(os.path.join('data_sample', f'{idx}_issue_tar.wav'), tar_signal.unsqueeze(0), self.sample_rate)
        else:
            zero_res_mask = False
            
        if cond_entropy < 1.5 and tar_entropy < 1.5 and residual_entropy < 1.5:
            cond_signal = load_waveform(filepath=os.path.join('template_input', 'cond.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            tar_signal = load_waveform(filepath=os.path.join('template_input', 'target.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            residual = load_waveform(filepath=os.path.join('template_input', 'residual.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        
        mel_spec_tar = wav_to_mel(wav=tar_signal,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_tar = mel_spec_mask(mel_spec=mel_spec_tar.numpy())
        mel_spec_mask_tar = mel_spec_mask_tar / torch.max(mel_spec_mask_tar + 1e-10)
        
        mel_spec_res = wav_to_mel(wav=residual,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_res = mel_spec_mask(mel_spec=mel_spec_res.numpy(),
                                          threshold=0.1)
        mel_spec_mask_res = mel_spec_mask_res / torch.max(mel_spec_mask_res + 1e-10)
        mel_spec_mask_res = torch.zeros_like(mel_spec_mask_res) if zero_res_mask else mel_spec_mask_res
        
        
        mel_spec_cond = wav_to_mel(wav=cond_signal,
                                window=self.window,
                                stft_args=self.stft_args,
                                mel_filterbank=self.mel_filterbank)
        mel_spec_mask_cond = mel_spec_mask(mel_spec=mel_spec_cond.numpy(),
                                            threshold=0.1,
                                            max_mask_drop_prob=0.0,
                                            min_filter_sig=4,
                                            max_filter_sig=4)
        mel_spec_mask_cond = mel_spec_mask_cond / torch.max(mel_spec_mask_cond + 1e-10)
        
        mel_spec_mix = wav_to_mel(wav=mixture,
                                window=self.window,
                                stft_args=self.stft_args,
                                mel_filterbank=self.mel_filterbank)
        mel_spec_mask_mix = mel_spec_mask(mel_spec=mel_spec_mix.numpy(),
                                        threshold=0.1,
                                        max_mask_drop_prob=0.0,
                                        min_filter_sig=4,
                                        max_filter_sig=4)
        mel_spec_mask_mix = mel_spec_mask_mix / torch.max(mel_spec_mask_mix + 1e-10)
        
        return {'tar_signal': tar_signal, 
            'cond_signal': cond_signal, 
            'mel_spec_mask_tar': mel_spec_mask_tar, 
            'mel_spec_mask_res': mel_spec_mask_res, 
            'residual': residual, 
            'tar_ins': tar_ins, 
            'cond_ins_name': cond_ins_name, 
            'mel_spec_tar': mel_spec_tar, 
            'mel_spec_res': mel_spec_res,
            'mel_spec_mask_cond': mel_spec_mask_cond,
            'mel_spec_mask_mix': mel_spec_mask_mix}
        
        
class HummingTestDataset(Dataset):
    def __init__(self, 
                 sample_rate, 
                 tar_len,
                 hop_len,
                 moise_path='/mnt/data/Music/moisesdb/moisesdb_v0.1',
                 humming_path='/mnt/data/Music/HumTrans',
                 augment=True):
        super().__init__()    
        self.track_paths = []
        self.audio_filenames = {}
        self.sample_rate = sample_rate
        self.tar_len = tar_len
        self.hop_len = hop_len
        self.augment = augment
        self.n_fft = (self.hop_len - 1) * 2
        self.stft_args = dict(n_fft=self.n_fft, hop_length=self.hop_len, center=True)
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sample_rate, n_stft=self.n_fft // 2 + 1)
        
        self.hum_midi_filenames = glob.glob(os.path.join(humming_path, 'midi_data', '*.mid'))
        
        self.humming_midi_tracks = {}
        for filename in self.hum_midi_filenames:
            track_id = filename.split('/')[-1].split('_')[1]
            if track_id not in self.humming_midi_tracks.keys():
                self.humming_midi_tracks[track_id] = [filename]
            else:
                self.humming_midi_tracks[track_id].append(filename)
        
        self.humming_choise = np.array(list(self.humming_midi_tracks.keys()))
        for key in self.humming_choise:
            self.humming_midi_tracks[key] = np.array(self.humming_midi_tracks[key])
            
        moise_track_paths = [os.path.join(moise_path, name) for name in os.listdir(moise_path) if os.path.isdir(os.path.join(moise_path, name))]
        self.moise_track_stems = {}
        for track in moise_track_paths:
            stems = [os.path.join(track, name) for name in os.listdir(track) if os.path.isdir(os.path.join(track, name))]
            for stem in stems:
                stem_key = stem.split('/')[-1]
                sources = glob.glob(os.path.join(stem, '*.wav'))
                if stem_key not in self.moise_track_stems.keys():
                    self.moise_track_stems[stem_key] = sources
                else:
                    self.moise_track_stems[stem_key] += sources  
        
        self.moise_choise = np.array(list(self.moise_track_stems.keys())) 
        for key in self.moise_choise:
            self.moise_track_stems[key] = np.array(self.moise_track_stems[key])   
     
        self.instrument_list = []
        for track in tqdm(self.track_paths): 
            with open(os.path.join(track, 'metadata.yaml'), 'r') as file:
                metadata = yaml.safe_load(file)
            ins_list = []
            audio_datapoint = {}
            count = 0
            for track_id, info in metadata['stems'].items():
                if info['audio_rendered'] and info['inst_class'] != 'Drums':
                    
                    if info['inst_class'] not in self.instrument_list:
                        self.instrument_list.append(info['inst_class'])
                        
                    ins = info['inst_class'] + '_' + str(count)
                    ins_list.append(ins)
                    audio_datapoint[ins] = {'track': os.path.join(track, 'stems', track_id+'.flac'),
                                            'midi': os.path.join(track, 'MIDI', track_id+'.mid')}
                    count += 1
            audio_datapoint['mix'] = os.path.join(track, 'mix.flac')
            audio_datapoint['ins'] = ins_list
            self.audio_filenames[track] = audio_datapoint
        
        sf2_path = "AegeanSymphonicOrchestra-SND/AegeanSymphonicOrchestra-SND.sf2"
        self.S = Synth(samplerate=self.sample_rate)
        self.sfid = self.S.sfload(sf2_path)
        
        tree = ET.parse('AegeanSymphonicOrchestra-SND/ASO-SND-instruments.xml')
        root = tree.getroot()
        self.instruments = []
        for group in root.findall('InstrumentGroup'):
            for instrument in group.findall('Instrument'):
                long_name = instrument.find('longName').text  # Extract long name of the instrument
                p_pitch_range_element = instrument.find('pPitchRange')
                if p_pitch_range_element is not None and p_pitch_range_element.text is not None:
                    pitch_range = p_pitch_range_element.text
                    pitch_low = int(pitch_range.split('-')[0])
                    pitch_high = int(pitch_range.split('-')[1])
                else:
                    pitch_low = 0
                    pitch_high = 127
                self.instruments.append({
                    'element': instrument,
                    'name': long_name,
                    'pitch_range': [pitch_low, pitch_high]
                })
        
       
         
    def __len__(self):
        return len(self.hum_midi_filenames)
    
    def load_residual(self):
        while True:
            ins_key_id = np.random.randint(low=0, high=len(self.moise_choise))
            ins_key = self.moise_choise[ins_key_id]
            s_filename_id = np.random.randint(low=0, high=len(self.moise_track_stems[ins_key]))
            s_filename = self.moise_track_stems[ins_key][s_filename_id]
            residual = load_waveform(filepath=s_filename, 
                                    tar_sr=self.sample_rate, 
                                    tar_len=self.tar_len)
            residual_entropy = signal_entropy(residual)
            if residual_entropy > 2:
                break
        return residual
    
    def __getitem__(self, idx):
        num_source = np.random.randint(low=3, high=6)
        filename = self.hum_midi_filenames[idx]
        midi, pitch_range, start_time = extract_midi_segment(midi_file=filename,
                                                            tar_len=self.tar_len,
                                                            augment=True)
        directory, filename = os.path.split(filename)
        new_directory = directory.replace('midi_data', 'wav_data_sync_with_midi')
        new_filename = os.path.splitext(filename)[0] + '.wav'
        new_path = os.path.join(new_directory, new_filename)
        cond_signal = load_waveform(filepath=new_path,
                                    start=start_time,
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
        
        
        instruments = self.instruments.copy()
        while True:
            random_idx = np.random.randint(low=0, high=len(instruments))
            selected_instrument = instruments[random_idx]
            ins_pitch_range = selected_instrument['pitch_range']
            if ins_pitch_range[0] <= pitch_range[0] and ins_pitch_range[1] >= pitch_range[1]:
                ctrl_value, program_value, channel_name = extract_program_and_controller(selected_instrument['element'])
                cond_ins_name = selected_instrument['name'] + '_' + channel_name
                
                midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                midi_to_audio = midi_to_audio / torch.max(torch.abs(midi_to_audio) + 1e-10)
                midi_entropy = signal_entropy(midi_to_audio)
                if midi_entropy < 1.5 and len(instruments) > 1:
                    instruments.remove(selected_instrument)
                else:
                    break
            else:
                if len(instruments) == 1:
                    midi_to_audio = fluidsynth_midi(midi=midi, 
                                                S=self.S, 
                                                sfid=self.sfid, 
                                                tar_len=self.tar_len, 
                                                sr=self.sample_rate, 
                                                program=program_value, 
                                                controller=ctrl_value)
                    break
                instruments.remove(selected_instrument)
        
        tar_signal = midi_to_audio
        residual = torch.stack([self.load_residual() for _ in range(num_source)])
        residual = torch.sum(residual, dim=0)

        cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        tar_signal = tar_signal / torch.max(torch.abs(tar_signal) + 1e-10)
        residual = residual / torch.max(torch.abs(residual) + 1e-10)
        
        # compute cond and res entropy
        cond_entropy = signal_entropy(cond_signal)
        cond_signal = tar_signal.clone() if cond_entropy < 1 else cond_signal
        residual_entropy = signal_entropy(residual)
        
        tar_entropy = signal_entropy(tar_signal)
        tar_signal = cond_signal.clone() if tar_entropy < 1 else tar_signal
        
        if cond_entropy < 1.5 and tar_entropy < 1.5:
            tar_signal = residual.clone()
            cond_signal = residual.clone()
            zero_res_mask = True
            # torchaudio.save(os.path.join('data_sample', f'{idx}_issue_tar.wav'), tar_signal.unsqueeze(0), self.sample_rate)
        else:
            zero_res_mask = False
            
        if cond_entropy < 1.5 and tar_entropy < 1.5 and residual_entropy < 1.5:
            cond_signal = load_waveform(filepath=os.path.join('template_input', 'cond.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            tar_signal = load_waveform(filepath=os.path.join('template_input', 'target.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            residual = load_waveform(filepath=os.path.join('template_input', 'residual.wav'),
                                    tar_sr=self.sample_rate,
                                    tar_len=self.tar_len)
            cond_signal = cond_signal / torch.max(torch.abs(cond_signal) + 1e-10)
        
        mel_spec_tar = wav_to_mel(wav=tar_signal,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_tar = mel_spec_mask(mel_spec=mel_spec_tar.numpy())
        mel_spec_mask_tar = mel_spec_mask_tar / torch.max(mel_spec_mask_tar + 1e-10)
        
        mel_spec_res = wav_to_mel(wav=residual,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_res = mel_spec_mask(mel_spec=mel_spec_res.numpy(),
                                          threshold=0.1)
        mel_spec_mask_res = mel_spec_mask_res / torch.max(mel_spec_mask_res + 1e-10)
        mel_spec_mask_res = torch.zeros_like(mel_spec_mask_res) if zero_res_mask else mel_spec_mask_res
        
        
        return {'tar_signal': tar_signal, 
            'cond_signal': cond_signal, 
            'mel_spec_mask_tar': mel_spec_mask_tar, 
            'mel_spec_mask_res': mel_spec_mask_res, 
            'residual': residual, 
            'tar_ins': 'humming', 
            'cond_ins_name': cond_ins_name, 
            'mel_spec_tar': mel_spec_tar, 
            'mel_spec_res': mel_spec_res,
            'mel_spec_mask_cond': mel_spec_tar,
            'mel_spec_mask_mix': mel_spec_res}

class InferenceDataset(Dataset):
    def __init__(self, 
                 track_path,
                 cond_path, 
                 mask_path,
                 sample_rate, 
                 tar_len,
                 hop_len):
        super().__init__()    
        self.sample_rate = sample_rate
        self.tar_len = tar_len
        self.hop_len = hop_len
        self.mask_path = mask_path
        self.n_fft = (self.hop_len - 1) * 2
        self.stft_args = dict(n_fft=self.n_fft, hop_length=self.hop_len, center=True)
        self.window = torch.hann_window(self.n_fft, periodic=True)
        self.mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sample_rate, n_stft=self.n_fft // 2 + 1)

        self.audio_filenames = glob.glob(os.path.join(track_path, '*mix.wav'))
        self.cond_filenames = {}
        for filename in self.audio_filenames:
            id = filename.split('/')[-1].split('_mix.wav')[0]
            self.cond_filenames[filename] = glob.glob(os.path.join(cond_path, f'*{id}*'))  
        
    def __len__(self):
        return len(self.audio_filenames)
    
    def __getitem__(self, idx):
        filename = self.audio_filenames[idx]
        cond_filename = np.random.choice(self.cond_filenames[filename])
        mixture = load_waveform(filepath=filename,
                                   tar_sr=self.sample_rate)
        cond_signal = load_waveform(filepath=cond_filename,
                                    tar_sr=self.sample_rate)
        mixture = mixture / torch.max(torch.abs(mixture))
        cond_signal = cond_signal / torch.max(torch.abs(cond_signal))
        
        mel_spec_tar = wav_to_mel(wav=cond_signal,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_tar = mel_spec_mask(mel_spec=mel_spec_tar.numpy(),
                                          threshold=0.1,
                                          max_mask_drop_prob=0.0,
                                          min_filter_sig=3,
                                          max_filter_sig=3)
        mel_spec_mask_tar = mel_spec_mask_tar / torch.max(mel_spec_mask_tar + 1e-10)
        
        mel_spec_res = wav_to_mel(wav=mixture,
                                  window=self.window,
                                  stft_args=self.stft_args,
                                  mel_filterbank=self.mel_filterbank)
        mel_spec_mask_res = mel_spec_mask(mel_spec=mel_spec_res.numpy(),
                                          threshold=0.1,
                                          max_mask_drop_prob=0.0,
                                          min_filter_sig=5,
                                          max_filter_sig=5)
        mel_spec_mask_res = mel_spec_mask_res / torch.max(mel_spec_mask_res + 1e-10)
        
        p_mask_filename = os.path.join(self.mask_path, 'mask_' + filename.split('/')[-1].split('.wav')[0] + '_p.npy')
        # read mask if exist
        if os.path.exists(p_mask_filename):
            print('hererere')
            mel_spec_mask_p = np.load(p_mask_filename)
            mel_spec_mask_p = mel_spec_mask(mel_spec=mel_spec_mask_p,
                                          threshold=0.0,
                                          max_mask_drop_prob=0.0,
                                          min_filter_sig=2,
                                          max_filter_sig=2)
            mel_spec_mask_p = mel_spec_mask_p / torch.max(mel_spec_mask_p + 1e-10)
        else:
            mel_spec_mask_p = mel_spec_mask_tar
            
        n_mask_filename = os.path.join(self.mask_path, 'mask_' + filename.split('/')[-1].split('.wav')[0] + '_n.npy')
        # read mask if exist
        if os.path.exists(n_mask_filename):
            mel_spec_mask_n = np.load(n_mask_filename)
            mel_spec_mask_n = mel_spec_mask(mel_spec=mel_spec_mask_n,
                                          threshold=0.0,
                                          max_mask_drop_prob=0.0,
                                          min_filter_sig=2,
                                          max_filter_sig=2)
            mel_spec_mask_n = mel_spec_mask_n / torch.max(mel_spec_mask_n + 1e-10)
        else:
            mel_spec_mask_n = mel_spec_mask_res
            
        return cond_signal, cond_signal, mel_spec_mask_p, mel_spec_mask_n, mixture, mixture, 'real', 'real', 'real', 'real', mel_spec_mask_tar, mel_spec_mask_res

class Collator:
    def __init__(self, 
                 snr_mean=-2.5,
                 snr_std=1.5,
                 snr_min=-5,
                 snr_max=5,
                 mode='train',
                 remix=False):
        self.snr_mean = snr_mean
        self.snr_std = snr_std
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.mode = mode
        self.idx = 0
        self.remix=remix
        
    def collate(self, minibatch):
        if self.mode == 'test' and minibatch[0]['tar_signal'] is None:
            return None, None, None, None, None, None, None, None, None, None, None, None
        
        residual = torch.stack([batch['residual'] for batch in minibatch])
        tar_signal = torch.stack([batch['tar_signal'] for batch in minibatch])

        if self.mode == 'train' and self.remix:
            snr = torch.normal(mean=self.snr_mean, std=self.snr_std, size=(tar_signal.shape[0],))
            snr = torch.clamp(snr, min=self.snr_min)
            mixture = torchaudio.functional.add_noise(waveform=tar_signal, 
                                                      noise=residual, 
                                                      snr=snr)
        else:
            mixture = tar_signal + residual
            
        max_mix, _ = torch.max(torch.abs(mixture), dim=1, keepdim=True)
        max_res, _ = torch.max(torch.abs(residual), dim=1, keepdim=True)
        max_tar, _ = torch.max(torch.abs(tar_signal), dim=1, keepdim=True)
        mixture = mixture / (max_mix + 1e-10)
        residual = residual / (max_res + 1e-10)
        tar_signal = tar_signal / (max_tar + 1e-10)
        
        cond_signal = torch.stack([batch['cond_signal'] for batch in minibatch])
        mel_spec_mask_tar = torch.stack([batch['mel_spec_mask_tar'] for batch in minibatch])
        mel_spec_mask_res = torch.stack([batch['mel_spec_mask_res'] for batch in minibatch])
        mel_spec_tar = torch.stack([batch['mel_spec_tar'] for batch in minibatch])
        mel_spec_res = torch.stack([batch['mel_spec_res'] for batch in minibatch])
        tar_ins = [batch['tar_ins'] for batch in minibatch]
        cond_ins_name = [batch['cond_ins_name'] for batch in minibatch]
        
        if self.mode == 'test':
            mel_spec_mask_cond = torch.stack([batch['mel_spec_mask_cond'] for batch in minibatch])
            mel_spec_mask_mix = torch.stack([batch['mel_spec_mask_mix'] for batch in minibatch])
            return tar_signal, cond_signal, mel_spec_mask_tar, mel_spec_mask_res, mixture, residual, tar_ins, cond_ins_name, mel_spec_tar, mel_spec_res, mel_spec_mask_cond, mel_spec_mask_mix
        
        # self.sample_rate = 16000
        
        # torchaudio.save(os.path.join('data_sample', f'{self.idx}_target.wav'), tar_signal, self.sample_rate)
        # torchaudio.save(os.path.join('data_sample', f'{self.idx}_res.wav'), residual, self.sample_rate)
        # torchaudio.save(os.path.join('data_sample', f'{self.idx}_cond.wav'), cond_signal, self.sample_rate)
        # torchaudio.save(os.path.join('data_sample', f'{self.idx}_mix.wav'), mixture, self.sample_rate)
        # self.idx += 1
        
        return tar_signal, cond_signal, mel_spec_mask_tar, mel_spec_mask_res, mixture, residual, tar_ins, cond_ins_name, mel_spec_tar, mel_spec_res
        

class SlakhMultiInsDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:
    
        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        track_dir: str = "./",
        audio_len: int = None,
        sample_rate: int = 16000,
        augment: bool = False,
        humming_prob=0.4,
        org_mix_prob=0.3,
        tar_ins='piano',
        inf_real=False,
        batch_size: int = 64,
        num_workers: int = 4,
        hop_length: Optional[int] = None, 
        num_frames: Optional[int] = None,
        pin_memory: Optional[bool] = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        assert audio_len is not None or (hop_length is not None and num_frames is not None), "Either audio_len or hop_length and num_frames must be provided"

        self.remix = False if org_mix_prob == 0 else True
        # use audio_len first, if not provided, use hop_length and num_frames to infer audio_len
        self.audio_len = (num_frames - 1) * hop_length if audio_len is None else audio_len
        
    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if self.hparams.inf_real:
            self.data_test = InferenceDataset(track_path='mix_l',
                                              cond_path='cond_l',
                                              mask_path='masks_l',
                                              sample_rate=self.hparams.sample_rate,
                                              tar_len=self.audio_len/self.hparams.sample_rate,
                                              hop_len=self.hparams.hop_length)
        else:
            self.data_train = SlakhMultiInsDataset(track_path=self.hparams.track_dir,
                                            sample_rate=self.hparams.sample_rate,
                                            tar_len=self.audio_len/self.hparams.sample_rate,
                                            hop_len=self.hparams.hop_length,
                                            augment=self.hparams.augment,
                                            humming_prob=self.hparams.humming_prob,
                                            org_mix_prob=self.hparams.org_mix_prob)
            self.data_val = SlakhMultiInsDataset(track_path=self.hparams.track_dir,
                                        sample_rate=self.hparams.sample_rate,
                                        tar_len=self.audio_len/self.hparams.sample_rate,
                                        hop_len=self.hparams.hop_length,
                                        mode='val',
                                        augment=self.hparams.augment,
                                        humming_prob=self.hparams.humming_prob,
                                        org_mix_prob=1)
            if self.hparams.humming_prob == 1:
                print('test humming')
                self.data_test = HummingTestDataset(sample_rate=self.hparams.sample_rate,
                                        tar_len=self.audio_len/self.hparams.sample_rate,
                                        hop_len=self.hparams.hop_length,
                                        augment=self.hparams.augment)
            else:
                print('test normal')
                self.data_test = SlakhMultiInsTestDataset(track_path=self.hparams.track_dir,
                                            sample_rate=self.hparams.sample_rate,
                                            tar_len=self.audio_len/self.hparams.sample_rate,
                                            hop_len=self.hparams.hop_length,
                                            tar_ins=self.hparams.tar_ins,
                                            augment=self.hparams.augment,
                                            humming_prob=self.hparams.humming_prob,
                                            org_mix_prob=1)


    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(remix=self.remix).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(mode='val', remix=False).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.hparams.inf_real:
            return DataLoader(
                dataset=self.data_test,
                batch_size=1,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
            )
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            collate_fn=Collator(mode='test', remix=False).collate,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

if __name__ == "__main__":
    import time
    random.seed(42)
    # tar ins: ['Piano', 'Guitar', 'Strings (continued)', 'Bass', 'Synth Lead', 'Reed', 'Brass', 'Organ', 'Synth Pad', 'Strings', 'Chromatic Percussion', 'Pipe']

    # current_time = int(time.time())
    # random.seed(current_time)
    # dataset = SlakhMultiInsDataModule('/mnt/data/Music/slakh2100_flac_redux',
    #                             hop_length=256,
    #                             num_frames=256,
    #                             augment=True,
    #                             sample_rate=16000,
    #                             tar_ins='Guitar',
    #                             batch_size=1,
    #                             num_workers=1,
    #                             humming_prob=0.0,
    #                             org_mix_prob=0.2)
    # dataset = SlakhMultiInsTestDataset('/mnt/data/Music/slakh2100_flac_redux',
    #                                    tar_len=4,
    #                             hop_len=256,
    #                             augment=True,
    #                             sample_rate=16000,
    #                             tar_ins='Guitar',
    #                             humming_prob=0.0,
    #                             org_mix_prob=0.2)
    dataset = HummingTestDataset('/mnt/data/Music/slakh2100_flac_redux',
                                       tar_len=4,
                                hop_len=256,
                                augment=True,
                                sample_rate=16000,
                                tar_ins='Violin')
    # a = dataset[0]
    exit()
    # dataset = SlakhDataset('/mnt/data/Music/slakh2100_flac_redux',
    #                        sample_rate=16000,
    #                        tar_len=4,
    #                        mode='test')
    # for i in tqdm(range(len(dataset))):
    #     a = dataset[i]
    # # print(dataset.segments)
    # data = dataset.segments
    # with open('/mnt/data/Music/slakh2100_flac_redux/test_segment.yaml', 'w') as file:
    #     yaml.dump(data, file, default_flow_style=False)
    # with open('/mnt/data/Music/slakh2100_flac_redux/test_segment.yaml', 'r') as file:
    #     data = yaml.safe_load(file)
    #     print(data)
        
    # dataset = SlakhDataset('/mnt/data/Music/slakh2100_flac_redux',
    #                        sample_rate=16000,
    #                        tar_len=4,
    #                        mode='train')
    # for i in tqdm(range(len(dataset))):
        # a = dataset[i]
    # print(dataset.segments)
    # data = dataset.segments
    # with open('/mnt/data/Music/slakh2100_flac_redux/train_segment.yaml', 'w') as file:
    #     yaml.dump(data, file, default_flow_style=False)
    # with open('/mnt/data/Music/slakh2100_flac_redux/train_segment.yaml', 'r') as file:
    #     data = yaml.safe_load(file)
    #     print(data)
    dataset.setup()
    for data_item in tqdm(dataset.test_dataloader()):
        tar_signal, cond_signal, mel_spec_mask_tar, mel_spec_mask_res, mixture, residual, tar_ins, cond_ins_name, mel_spec_tar, mel_spec_res = data_item
        print(tar_signal.shape, cond_signal.shape, tar_ins)
        # print(tar_signal.shape, cond_signal.shape, mixture.shape, tar_ins, channel_name)
    # with open('/mnt/data/Music/slakh2100_flac_redux/train_segment.yaml', 'w') as file:
    #     yaml.dump(dataset.data_train.segments, file, default_flow_style=False)
    # for data_item in dataset.test_dataloader():
    #     a = data_item
    # with open('/mnt/data/Music/slakh2100_flac_redux/test_segment.yaml', 'w') as file:
    #     yaml.dump(dataset.data_test.segments, file, default_flow_style=False)
        # print(audio.shape, midi.shape)
        # print(clean.shape, noisy.shape)
