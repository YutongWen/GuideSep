from matplotlib import colors
import librosa
import os
import glob
import torch
import torchaudio
import numpy as np
from scipy.ndimage import gaussian_filter

def wav_to_mel(wav, window, stft_args, mel_filterbank):
    spec = torch.stft(wav, window=window, normalized=True, 
                    return_complex=True, **stft_args)
    magnitude = torch.abs(spec)
    mel_spec = mel_filterbank(magnitude)
    mel_spec = mel_spec / torch.max(mel_spec)
    return mel_spec

def sound_render( s, sr=16000, show='stft', name='', mask=None):
    import base64
    from io import BytesIO
    from matplotlib.pyplot import figure, gca, gcf, close

    # Get the visual if requested
    if mask is not None:
        show = 'mel'
        
    if show is not None:
        figure( figsize=(2.5,.75))
        if show == 'stft':
            sz = sr//64
            gca().specgram( s, Fs=sr, NFFT=sz, noverlap=sz-sz//4, scale='linear', norm=colors.PowerNorm( gamma=0.2))
        elif show == 'wave':
            gca().plot( s)
        elif show == 'spec':
            from numpy import log1p
            from numpy.fft import rfft
            gca().plot( abs( rfft( s))**.5)
        elif show == 'mel':
            mel_filterbank = torchaudio.transforms.MelScale(n_mels=80, sample_rate=sr, n_stft=256)
            window = torch.hann_window(510, periodic=True)
            stft_args = dict(n_fft=510, hop_length=256, center=True)
            S = wav_to_mel(wav=torch.tensor(s),
                                  window=window,
                                  stft_args=stft_args,
                                  mel_filterbank=mel_filterbank)
            S = S.squeeze().numpy()[:, :256]
            S_dB = librosa.power_to_db(S, ref=np.max)
            # invesre the S_dB y axis
            S_dB = S_dB[::-1, :]
            gca().imshow(S_dB) 
            
            if mask is not None:
                mask = gaussian_filter(mask, sigma=2)
                gca().imshow(mask, alpha=0.3, cmap='Reds')
            
        gca().axis( 'tight')
        gca().axis( False)
        gcf().tight_layout( pad=0)
        imf = BytesIO()
        gcf().savefig( imf, format='png')
        m = base64.b64encode( imf.getvalue()).decode('utf-8') + '\n'
        d = [f'<img src="data:image/jpeg;base64,{m}"  style="background-color:white;">']
        close()

    # Make a table with name / optional visual / sound player
    import tabulate
    from IPython.display import Audio
    t = [[name]]
    if show is not None:
        t += [d]
    t += [[Audio( s/abs( s).max(), rate=sr)._repr_html_()[3:].replace( 'controls', 'controls style="width: 250px; height: 24px;"')]]

    return tabulate.tabulate( t, tablefmt='unsafehtml')

def soundgrid( *k, sr=16000, r_header=[], c_header=[], extra_list=[], mask=[], show='stft'):
    import tabulate
    t = list( zip( *[[sound_render(s.squeeze(), sr, show=show, mask=m)._repr_html_() for m, s in zip(_m, p)] for _m, p in zip(mask, k)]))
    for i in range(len(t)):
        t[i] = list(t[i])
        t[i].insert(0, c_header[i])
        t[i].insert(1, extra_list[i])
        t[i] = tuple(t[i])
    return tabulate.tabulate( t, tablefmt='unsafehtml', headers=r_header, stralign='center')



root_dir = '/Users/cooper/Desktop/separation_examples/demo/audios/real_ins'
row_list = ['mix', 'cond', 'pred', 'mix']
ins_list = ['1', '2', '3', '4', '5', '5_1', '6', '7', '8']
mask_list = ['1', '1', '1', '1', '5_1', '5', '6', '6', '6']
row_content = []
masks = []
sample_rate = 16000

for i, row in enumerate(row_list):
    audios = []
    mask = []
    for idx, m_idx in zip(ins_list, mask_list):
        _row = row
        if idx == '5_1':
            idx = '5'
            if row == 'pred':
                _row = 'pred_1'
        filename = os.path.join(root_dir, f'*{idx}_{_row}.wav')
        filename = glob.glob(filename)[0]
        x, sr = librosa.load(filename, sr=sample_rate)
        # pad or trim to 65280 samples
        if len(x) < 65280:
            x = np.pad(x, (0, 65280 - len(x)))
        else:
            x = x[:65280]
            
        assert sr == sample_rate
        audios.append(x)
        
        if i == 3:
            filename = os.path.join(root_dir, 'masks', f'mask_real_{m_idx}_mix_p.npy')
            if m_idx == '5_1':
                m_idx = '5'
                filename = os.path.join(root_dir, 'masks', f'mask_real_{m_idx}_mix_n.npy')
                
            filename = glob.glob(filename)[0]
            _mask = np.load(filename)
            mask.append(_mask)
        else:
            mask.append(None)
    row_content.append(audios)
    masks.append(mask)
    
        
row_header = ['Target Instrument', 'Mixture', 'Condition Type', 'Melody Condition', 'MelodySep Result', 'Mel_mask']
c_header = ['Piano', 'Strings', 'Pipe', 'Strings', 'Pipe', 'Synth', 'Pipe', 'Strings', 'Electric Guitar']
type_list = ['Melody + Pseudo-Mel-mask', 'Melody', 'Melody + Mel-mask', 'Melody', 'Melody + Mel-mask', 'Mel-mask', 'Melody', 'Melody', 'Melody + Mel-mask']

# Pack them into a grid
sg = soundgrid(*row_content, sr=sample_rate, r_header=row_header, c_header=c_header, extra_list=type_list, mask=masks, show='stft')

with open('hearme.html', 'w') as f:
    f.write(sg)