import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import fairseq
import librosa


class Hubert(nn.Module):
    def __init__(self, opts, hidden_dim, ckpt):
        super().__init__()

        self.sampling_rate = opts.sampling_rate
        self.max_audio_len = opts.max_audio_len

        model, cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt])
        self.encoder = model[0]
        self.normalize = cfg.task.normalize
        self.encoder.feature_grad_mult = 0.0
        self.encoder.encoder.layerdrop = 0.0

        self.interpreter = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=opts.dropout),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(p=opts.dropout),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, opts.num_labels)
        )

    def preprocess(self, x):
        wavs = []
        for file_path in x:
            data, _ = librosa.load(file_path, sr=self.sampling_rate)
            if len(data) > self.max_audio_len:
                data = data[-self.max_audio_len:]
            wavs.append(torch.FloatTensor(data).cuda())

        if self.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).cuda()
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).cuda(),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)

        features, _ = self.encoder.extract_features(
            padded_wav,
            padding_mask=wav_padding_mask,
            mask=None,
        )

        return features

    def forward(self, videos, audio_paths, texts):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)
        outputs = self.interpreter(hidden_states)

        return outputs

    def extract_feats(self, audio_paths):
        hidden_states = self.preprocess(audio_paths).mean(dim=1)

        return hidden_states


def HubertBase(opts):
    return Hubert(opts, 768, f"{opts.ckpt_root}/hubert_base_ls960.pt")


def HubertLarge(opts):
    return Hubert(opts, 1024, f"{opts.ckpt_root}/hubert_large_ll60k.pt")
