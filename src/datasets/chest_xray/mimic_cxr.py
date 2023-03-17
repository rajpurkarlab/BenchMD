import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.vision import VisionDataset

from src.datasets.specs import Input2dSpec


def any_exist(files):
    return any(map(os.path.exists, files))


CHEXPERT_LABELS = {
    'No Finding': 0,
    'Enlarged Cardiomediastinum': 1,
    'Cardiomegaly': 2,
    'Lung Opacity': 3,
    'Lung Lesion': 4,
    'Edema': 5,
    'Consolidation': 6,
    'Pneumonia': 7,
    'Atelectasis': 8,
    'Pneumothorax': 9,
    'Pleural Effusion': 10,
    'Pleural Other': 11,
    'Fracture': 12,
    'Support Devices': 13,
}


class MIMIC_CXR(VisionDataset):
    '''A dataset class for the MIMIC-CXR dataset (https://physionet.org/content/mimic-cxr/2.0.0/).
    Note that you must register and manually download the data to use this dataset.
    '''
    # Dataset information.

    CHEXPERT_LABELS_IDX = np.array(
        [
            CHEXPERT_LABELS['Atelectasis'], CHEXPERT_LABELS['Enlarged Cardiomediastinum'], CHEXPERT_LABELS['Cardiomegaly'],
            CHEXPERT_LABELS['Lung Opacity'], CHEXPERT_LABELS['Lung Lesion'], CHEXPERT_LABELS['Edema'],
            CHEXPERT_LABELS['Consolidation'], CHEXPERT_LABELS['Pneumonia'], CHEXPERT_LABELS['Atelectasis'],
            CHEXPERT_LABELS['Pneumothorax'], CHEXPERT_LABELS['Pleural Effusion'], CHEXPERT_LABELS['Pleural Other'],
            CHEXPERT_LABELS['Fracture'], CHEXPERT_LABELS['Support Devices']
        ],
        dtype=np.int32
    )
    LABEL_FRACS = {'small': 8, 'medium': 64, 'large': 256, 'full': np.inf}
    NUM_CLASSES = 14  # 14 total: len(self.CHEXPERT_LABELS_IDX)
    INPUT_SIZE = (224, 224)
    PATCH_SIZE = (16, 16)
    IN_CHANNELS = 1

    def __init__(self, base_root: str, download: bool = True, train: bool = True, finetune_size: str = None):
        self.root = os.path.join(base_root, 'chest_xray', 'mimic-cxr')
        super().__init__(self.root)
        self.index_location = self.find_data()
        self.split = ['train'] if train else ['test', 'validate']
        self.finetune_size = 0 if finetune_size is None else self.LABEL_FRACS[finetune_size]
        self.build_index()
        self.TRANSFORMS = transforms.Compose(
            [
                transforms.Resize(self.INPUT_SIZE[0] - 1, max_size=self.INPUT_SIZE[0]),
                transforms.ToTensor(),
                transforms.Normalize([0.4721], [0.3025])
            ]
        )

        # train mean: tensor([0.4721])
        # train std: tensor([0.3025])

    def find_data(self):
        os.makedirs(self.root, exist_ok=True)
        components = list(map(lambda x: os.path.join(self.root, 'files' + x), ['', '.zip']))
        # if no data is present, prompt the user to download it
        if not any_exist(components):
            raise RuntimeError(
                """
                'Visit https://physionet.org/content/mimic-cxr-jpg/2.0.0/ to apply for access'
                'Use: wget -r -N -c -np --user [your user name] --ask-password https://physionet.org/files/mimic-cxr-jpg/2.0.0/ to download the data'
                'Once you receive the download links, place the zip file in {}'.format(self.root)'
                """
            )

        # if the data has not been extracted, extract the data, prioritizing the full-res dataset
        if not any_exist(components[:1]):
            if os.path.exists(components[1]):
                print('Extracting data...')
                extract_archive(components[1])
                print('Done')

        for i in (0, 1):
            if os.path.exists(components[i]):
                return components[i]
        raise FileNotFoundError('MIMIC-CXR-JPG data not found')

    def build_index(self):
        print('Building index...')
        splits = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-split.csv.gz"), compression='gzip')
        labels = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-chexpert.csv.gz"), compression='gzip').fillna(0)
        study_ids = pd.read_csv(os.path.join(self.root, "cxr-study-list.csv.gz"), compression='gzip')
        metadata0 = pd.merge(study_ids, labels, on=['subject_id', 'study_id'])
        metadata = pd.merge(metadata0, splits, on=['subject_id', 'study_id'])
        metadata.to_csv(os.path.join(self.root, "metadata.csv"))
        index_file = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        index_file = index_file[index_file.split.isin(self.split)].reset_index(drop=True)

        index_file['fnames'] = np.array(index_file['path'].apply(lambda x: x.split('.')[:-1][0])
                                       ) + '/' + np.array(index_file['dicom_id']) + '.jpg'

        print("checking if all files exist...")
        index_file = index_file[[os.path.isfile(os.path.join(self.root, i)) for i in index_file['fnames']
                                ]].reset_index(drop=True)

        # if finetuning, get 'finetune_size' labels for each class
        # if insufficient examples, use all examples from that class
        if self.split == 'train' and self.finetune_size > 0 and self.finetune_size != np.inf:
            print("Starting non-overlapping sampling for multi-label finetuning data...")
            index_file = index_file.fillna(0)
            cols = list(CHEXPERT_LABELS.keys())
            for c in cols:
                index_file.loc[(index_file[c] < 0), c] = 0

            index = pd.DataFrame(columns=index_file.columns)

            for c in cols:

                unique_counts = index_file[c].value_counts()
                for c_value, l in unique_counts.items():

                    df_sub = index_file[index_file[c] == c_value]
                    print(f"class {c} == {c_value} has in total {l} samples to sample from")
                    if l <= self.finetune_size:
                        g = df_sub.sample(n=min(l, self.finetune_size), replace=False)
                        print("sampling all data since class size < finetune size")
                        index = index.append(g)
                    else:
                        i = index.copy()
                        curr_n_samples = tuple([i.shape[0]])
                        sample_pool = df_sub.copy()
                        g = df_sub.sample(n=min(l, self.finetune_size), replace=False)
                        index = index.append(g).drop_duplicates(['dicom_id'])
                        remaining = min(l, self.finetune_size)
                        sample_pool = sample_pool.append(g).drop_duplicates(['dicom_id'], keep=False)
                        curr_sample_pool_size = sample_pool.shape[0]
                        times = 1
                        while index.shape[0] < (curr_n_samples[0] + self.finetune_size):
                            print(
                                f"already sampled{index.shape[0] - curr_n_samples[0]}, sampling pool size {curr_n_samples[0]}"
                            )
                            if curr_sample_pool_size == 0:
                                print("RUN OUT OF SAMPLES")
                                break
                            times += 1
                            remaining = curr_n_samples[0] + self.finetune_size - index.shape[0]
                            print(
                                f"resampling the {times} time, because for class {c}, sampled #{index.shape[0]}, lacking {remaining } samples"
                            )
                            g = sample_pool.sample(n=min(curr_sample_pool_size, self.finetune_size, remaining), replace=False)
                            index = index.append(g).drop_duplicates(['dicom_id'])
                            sample_pool = sample_pool.append(g).drop_duplicates(['dicom_id'], keep=False)
                            curr_sample_pool_size = sample_pool.shape[0]

                        print(
                            f"--- ending sampling for class {c} =={c_value} with {index.shape[0] -curr_n_samples[0] } sampled ---"
                        )
            index_file = index.reset_index(drop=True)

        #100% finetune
        elif self.split == 'train' and self.finetune_size == np.inf:
            index_file = index_file.fillna(0)
            cols = list(CHEXPERT_LABELS.keys())
            for c in cols:
                index_file.loc[(index_file[c] < 0), c] = 0

            index_file = index_file.reset_index(drop=True)

        LABELS_COL = index_file.columns.get_loc("Atelectasis")
        end_col = LABELS_COL + len(CHEXPERT_LABELS)
        # missing values occur when no comment is made on a particular diagnosis. we treat this as a negative diagnosis
        self.labels = index_file.iloc[:, range(LABELS_COL, end_col)].values
        self.labels = np.maximum(self.labels, 0)  # convert -1 (unknown) to 0
        self.fnames = index_file['fnames'].values
        print('Done')

    def __len__(self) -> int:
        return self.fnames.shape[0]

    def __getitem__(self, index: int):
        fname = self.fnames[index]
        image = Image.open(os.path.join(self.root, fname)).convert("L")
        img = self.TRANSFORMS(image)

        _, h, w = np.array(img).shape
        if h > w:
            dim_gap = img.shape[1] - img.shape[2]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((pad1, 0, pad2, 0))(img)
        elif h == w:
            #edge case 223,223,  resize to match 224*224
            dim_gap = self.INPUT_SIZE[0] - h
            pad1, pad2 = dim_gap, dim_gap
            img = transforms.Pad((pad1, pad2, 0, 0))(img)
        else:
            dim_gap = img.shape[2] - img.shape[1]
            pad1, pad2 = dim_gap // 2, (dim_gap + (dim_gap % 2)) // 2
            img = transforms.Pad((0, pad1, 0, pad2))(img)
        label = torch.tensor(self.labels[index][self.CHEXPERT_LABELS_IDX]).long()
        return index, img.float(), label

    @staticmethod
    def num_classes():
        return MIMIC_CXR.NUM_CLASSES

    @staticmethod
    def spec():
        return [
            Input2dSpec(input_size=MIMIC_CXR.INPUT_SIZE, patch_size=MIMIC_CXR.PATCH_SIZE, in_channels=MIMIC_CXR.IN_CHANNELS),
        ]
