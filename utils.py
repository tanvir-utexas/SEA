import torch
import audioldm
from audioldm.audio import TacotronSTFT
from audioldm.utils import default_audioldm_config
from typing import Optional, List, Tuple, Dict, NamedTuple
from audioldm.audio import read_wav_file
from audioldm.utils import default_audioldm_config

def load_waveform(file_path, batch_size, device, duration):
    waveform = read_wav_file(file_path, int(duration * 102.4) * 160)
    waveform = torch.FloatTensor(waveform)
    waveform = waveform.expand(batch_size, -1)
    xc = waveform
    xc = xc.to(device)

    return xc


def load_audio(audio_path: str, left: int = 0, right: int = 0, device: Optional[torch.device] = None
               ) -> torch.tensor:
    if type(audio_path) is str:
        duration = audioldm.utils.get_duration(audio_path)
        config = default_audioldm_config()

        fn_STFT = TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )
        
        mel, _, _ = audioldm.audio.wav_to_fbank(audio_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT)
        mel = mel.unsqueeze(0)
    else:
        mel = audio_path

    c, h, w = mel.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    mel = mel[:, :, left:w-right]
    mel = mel.unsqueeze(0).to(device)

    return mel
