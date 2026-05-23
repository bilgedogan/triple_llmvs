import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def _resolve_llama_key_summe(video_data, video_name):
    raw = str(np.array(video_data[video_name + '/video_name']))
    return raw[2:-1].replace(' ', '_')


class MultimodalSummDataset(Dataset):
    """Multimodal dataset returning per-video visual (CLIP, 1024d), audio
    (Whisper, 512d), and raw LLaMA embeddings (user_prompt + gen, 5120d each)
    plus GT score and (for val/test) summary metadata."""

    def __init__(self, dataset, mode, split_idx,
                 llama_root, clip_path, audio_path):
        assert dataset in ('summe', 'tvsum')
        self.dataset = dataset
        self.mode = mode

        if dataset == 'summe':
            self.video_h5_path = 'SumMe/eccv16_dataset_summe_google_pool5.h5'
            self.split_file = 'dataset/summe_splits.json'
        else:
            self.video_h5_path = 'TVSum/eccv16_dataset_tvsum_google_pool5.h5'
            self.split_file = 'dataset/tvsum_splits.json'

        self.video_data = h5py.File(self.video_h5_path, 'r')
        self.llama_user = h5py.File(f'{llama_root}/user_prompt/user_prompt_pool.h5', 'r')
        self.llama_gen = h5py.File(f'{llama_root}/gen/gen_pool.h5', 'r')
        self.clip_data = h5py.File(clip_path, 'r')
        self.audio_data = h5py.File(audio_path, 'r')

        with open(self.split_file, 'r') as f:
            self.splits = json.load(f)
        self.keys = self.splits[split_idx][mode + '_keys']

    def __len__(self):
        return len(self.keys)

    def _llama_key(self, video_name):
        if self.dataset == 'summe':
            return _resolve_llama_key_summe(self.video_data, video_name)
        return str(video_name)

    def __getitem__(self, index):
        video_name = self.keys[index]
        d = {'video_name': video_name}

        llama_key = self._llama_key(video_name)
        # Raw llama embeddings: shape (T, tokens, 5120) — keep as float32 for downstream ops.
        d['llama_user'] = torch.as_tensor(np.array(self.llama_user[llama_key]), dtype=torch.float32)
        d['llama_gen'] = torch.as_tensor(np.array(self.llama_gen[llama_key]), dtype=torch.float32)

        # CLIP visual (T, 1024)
        d['visual'] = torch.as_tensor(np.array(self.clip_data[video_name + '/features']), dtype=torch.float32)
        # Whisper audio (T, 512)
        d['audio'] = torch.as_tensor(np.array(self.audio_data[video_name + '/features']), dtype=torch.float32)
        # Original GoogLeNet pool5 features (T, 1024) — kept for diversity reward signal.
        d['pool5'] = torch.as_tensor(np.array(self.video_data[video_name + '/features']), dtype=torch.float32)
        d['gtscore'] = torch.as_tensor(np.array(self.video_data[video_name + '/gtscore']), dtype=torch.float32)

        if self.mode != 'train':
            d['n_frames'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frames']))
            d['picks'] = torch.as_tensor(np.array(self.video_data[video_name + '/picks']))
            d['change_points'] = torch.as_tensor(np.array(self.video_data[video_name + '/change_points']))
            d['n_frame_per_seg'] = torch.as_tensor(np.array(self.video_data[video_name + '/n_frame_per_seg']))
            d['gt_summary'] = torch.as_tensor(np.array(self.video_data[video_name + '/user_summary']))

        return d


def _align_lengths(d):
    """Truncate per-frame modalities to common min length (rare safety guard)."""
    t = min(d['visual'].shape[0], d['audio'].shape[0], d['llama_user'].shape[0],
            d['llama_gen'].shape[0], d['gtscore'].shape[0], d['pool5'].shape[0])
    for k in ('visual', 'audio', 'llama_user', 'llama_gen', 'gtscore', 'pool5'):
        d[k] = d[k][:t]
    return d


class MultimodalTrainCollator(object):
    def __call__(self, batch):
        batch = [_align_lengths(b) for b in batch]
        lengths = torch.LongTensor([b['visual'].shape[0] for b in batch])
        max_len = int(lengths.max().item())
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        visual = pad_sequence([b['visual'] for b in batch], batch_first=True)
        audio = pad_sequence([b['audio'] for b in batch], batch_first=True)
        gtscore = pad_sequence([b['gtscore'] for b in batch], batch_first=True)
        pool5 = pad_sequence([b['pool5'] for b in batch], batch_first=True)
        # llama tensors: (T, tokens, 5120). pad_sequence pads dim 0.
        llama_user = pad_sequence([b['llama_user'] for b in batch], batch_first=True)
        llama_gen = pad_sequence([b['llama_gen'] for b in batch], batch_first=True)

        return {
            'video_name': [b['video_name'] for b in batch],
            'visual': visual,
            'audio': audio,
            'llama_user': llama_user,
            'llama_gen': llama_gen,
            'gtscore': gtscore,
            'pool5': pool5,
            'mask': mask,
            'lengths': lengths,
        }


class MultimodalValCollator(object):
    def __call__(self, batch):
        batch = [_align_lengths(b) for b in batch]
        lengths = torch.LongTensor([b['visual'].shape[0] for b in batch])
        max_len = int(lengths.max().item())
        mask = torch.arange(max_len)[None, :] < lengths[:, None]

        visual = pad_sequence([b['visual'] for b in batch], batch_first=True)
        audio = pad_sequence([b['audio'] for b in batch], batch_first=True)
        gtscore = pad_sequence([b['gtscore'] for b in batch], batch_first=True)
        pool5 = pad_sequence([b['pool5'] for b in batch], batch_first=True)
        llama_user = pad_sequence([b['llama_user'] for b in batch], batch_first=True)
        llama_gen = pad_sequence([b['llama_gen'] for b in batch], batch_first=True)

        return {
            'video_name': [b['video_name'] for b in batch],
            'visual': visual,
            'audio': audio,
            'llama_user': llama_user,
            'llama_gen': llama_gen,
            'gtscore': gtscore,
            'pool5': pool5,
            'mask': mask,
            'lengths': lengths,
            'n_frames': [b['n_frames'] for b in batch],
            'picks': [b['picks'] for b in batch],
            'change_points': [b['change_points'] for b in batch],
            'n_frame_per_seg': [b['n_frame_per_seg'] for b in batch],
            'gt_summary': [b['gt_summary'] for b in batch],
        }
