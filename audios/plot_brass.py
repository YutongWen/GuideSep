from matplotlib import colors
import librosa
import os
import glob
import torch
import torchaudio
import numpy as np

def wav_to_mel(wav, window, stft_args, mel_filterbank):
    spec = torch.stft(wav, window=window, normalized=True, 
                    return_complex=True, **stft_args)
    magnitude = torch.abs(spec)
    mel_spec = mel_filterbank(magnitude)
    mel_spec = mel_spec / torch.max(mel_spec)
    return mel_spec

def sound_render( s, sr=16000, show='stft', name=''):
    import base64
    from io import BytesIO
    from matplotlib.pyplot import figure, gca, gcf, close

    # Get the visual if requested
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

def soundgrid( *k, sr=16000, r_header=[], c_header=[], show='stft'):
    import tabulate
    t = list( zip( *[[sound_render(s.squeeze(), sr, show=show)._repr_html_() for s in p] for p in k]))
    for i in range(len(t)):
        t[i] = list(t[i])
        t[i].insert(0, c_header[i])
        t[i] = tuple(t[i])
    return tabulate.tabulate( t, tablefmt='unsafehtml', headers=r_header, stralign='center')



root_dir = '/Users/cooper/Desktop/separation_examples/demo/audios/slakh_humming'
ins_list = ['brass']
row_list = ['mix', 'humming', 'pred', 'pred_mask', 'ref']
raw_content = []
sample_rate = 16000

for i, row in enumerate(row_list):
    audios = []
    for ins in ins_list:
        filename = os.path.join(root_dir, f'{ins}_{row}.wav')
        filename = glob.glob(filename)[0]
        x, sr = librosa.load(filename, sr=sample_rate)
        
        if len(x) < 65280:
            x = np.pad(x, (0, 65280 - len(x)))
        else:
            x = x[:65280]
            
        assert sr == sample_rate
        audios.append(x)
    raw_content.append(audios)
        
row_header = ['Target Instrument', 'Mixture', 'Melody Condition', 'MelodySep Result w/out Mask', 'MelodySep Result w/ Mask', 'Ground-truth']
c_header = ['Brass']
# Make three sound sets
# x = [sin( i*linspace( 0, 2*pi, 8000)**1.4) for i in [100, 200, 400, 800, 1600]]
# y = [sign( sin( i*linspace( 0, 2*pi, 8000)**1.4)) for i in [100, 200, 400, 800, 1600]]
# z = [n[i+1:] + n[:-i-1] for i,n in enumerate( random.randn( 5, 8000))]

# Pack them into a grid
sg = soundgrid(*raw_content, sr=sample_rate, r_header=row_header, c_header=c_header, show='stft')

with open('hearme.html', 'w') as f:
    f.write(sg)