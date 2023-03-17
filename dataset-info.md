# Medical Benchmarking Datasets Info

This document contains information for downloading datasets, label distributions, label mappings, and other miscellaneous dataset information.

Note that all dataset classes and data files will fall under your configured ``data_root``. The default root path is ``/home/ubuntu/2022-spr-benchmarking/src/datasets/``.

<!-- TOC -->
- [Electrocardiograms (ECG)](#ecg)
    - [PTB-XL](#ptb-xl)
    - [Chapman/Shaoxing](#chapmanshaoxing)
    - [Georgia 12-Lead ECG Challenge](#georgia-12-lead-ecg-challenge)
    - [China Physiological Signal Challenge 2020](#cpsc-china-physiological-signal-challenge-2020)
- [Electroencephalograms (EEG)](#eeg)
    - [Sleep Heart Health Study](#sleep-heart-health-study)
    - [ISRUC SLEEP EEG](#isruc-sleep-eeg)
- [Chest X-Rays](#cxr)
    - [MIMIC-CXR](#mimic-cxr)
    - [CheXpert](#chexpert)
    - [VinDr-CXR](#vindr-cxr)
- [Mammograms](#mammography)
    - [Vindr-Mammo](#vindr-mammo)
    - [CBIS-DDSM](#cbis-ddsm)
- [Dermascopic Images](#dermascopic-images)
    - [BCN 20000](#bcn_20000-isic2019)
    - [HAM10000](#ham10000-isic2018)
    - [PAD-UFES-20](#pad-ufes-20-dataset-from-brazil-smartphone-imageset)
- [Fundus Images](#fundus-images)
    - [Messidor-2](#messidor-2)
    - [APTOS 2019](#aptos-2019-blindness-detection-dataset)
    - [Jinchi University Hospital](#jinchi-university-hospital-dataset)
- [Low Dose Computed Tomography (LDCT)](#ldct)
    - [Lung Image Database Consortium](#lidc-idri)
    - [LNDb Dataset](#lndb)

<!-- /TOC -->

## ECG

### Label Distributions

|                                    | PTB-XL (source) | PTB-XL (source)  | Chapman-Shaoxing (target) | Georgia (target) | CPSC (target)    |
|------------------------------------|-----------------|------------------|---------------------------|------------------|------------------|
| Class                              | Training    | Validation   | Validation            | Validation   | Validation   |
| Normal                             | 9222 (52.77\%)  | 2322 (53.24\%)   | 1129 (55.05\%)            | 725 (35.07\%)    | 190 (13.8\%)     |
| Conduction Disturbance             | 1386 (7.93\%)   | 348 (7.98\%)     | 249 (12.14\%)             | 240 (11.61\%)    | 717 (52.07\%)    |
| Myocardial Infarction              | 1285 (7.35\%)   | 333 (7.64\%)     | 2 (0.098\%)               | 82 (3.97\%)      | 5 (0.36\%)       |
| Ischemic ST-T Changes              | 1661 (9.5\%)    | 420 (9.63\%)     | 260 (12.68\%)             | 437 (21.14\%)    | 213 (15.47\%)    |
| Other                              | 1462 (8.37\%)   | 360 (8.25\%)     | 33 (1.61\%)               | 263 (12.72\%)    | 116 (8.42\%)     |
| Atrial fibrillation/atrial flutter | 475 (2.72\%)    | 103 (2.36\%)     | 232 (11.31\%)             | 2 (0.097\%)      | 131 (9.51\%)     |
| Hypertrophy                        | 1985 (11.36\%)  | 475 (10.89\%)    | 146 (7.12\%)              | 318 (15.38\%)    | 5 (0.36\%)       |
| Total \# Examples                  | 17476           | 4361             | 2051                      | 2067             | 1377             |

### PTB-XL

#### Loading the Dataset

The [PTB-XL Dataset](https://www.nature.com/articles/s41597-020-0495-6) is a largest freely accessible annotated clinical EKG dataset, and was selected as the training dataset (Wagner et al. 2020).  The dataset consists of a total of 21,837 EKGs for 18,885 patients. The input is a 12-lead EKG image of 5-second length and 500Hz sampling rate, the label is one of six classes of cardiac pathologies that could be detected through EKG: NORM, CD, HYP, MI, STTC- ischemia, A. Fib.. This classification task required some label remapping from the original label set, which we detail below.

1. Dataset will download automatically, as long as `download = True` is set in `data_root/ecg/ptbxl.py`. Alternatively, manually download using:
```
wget -O PhysioNetChallenge2020_Training_PTB-XL.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_PTB-XL.tar.gz/

```
2. Extract the tar.gz file and move the WFDB_PTBXL folder under your data root path

#### Label Mappings
| Class Name | PTB-XL Labels Included| 
|------------------------------------|-----------------|
| Normal | NORM, SARRH, SBRAD, SR, STACH |
| CD | AVB, 1AVB, 2AVB, 3AVB, CD, CLBBB, CRBBB, ILBBB, IRBBB, IVCB	IVCD, LAFB,	LAFB/LPFB, LPFB, LPR, PSVT,	SVARR, SVTAC, WPW |
| HYP | HYP, ALAD, LAD, LAO/LAE, LVH, RAD, RHV, RVH, RAO/RAE, SEHYP, VCLVH|
| MI | AMI, ALMI,ASMI, ILMI, IMI, INJAL, INJIL, INJLA, INVT, IPLMI, IPMI, LMI, MI, PMI|
|STTC | ANEUR,	DIG, EL, ISC\_, ISCA, ISCAL, ISCAN, ISCAS, ISCI, ISCIL, ISCIN, ISCLA, LNGQT, NDT, NST\_, NT\_, STD\_, STE\_, STTC, and TAB\_|
| A. Fib/ Aflutter |AFIB,  AFLT |
| Other | ABQRS, ARAD, AXL, AXR, BIGU, HVOLT, LOWT, LVOLT, PACE, PAC, PRC(S), PVC, QWAVE, SAG, and TRIGU|

### Chapman/Shaoxing

#### Loading the Dataset

This [Chapman Shaoxing Dataset](https://www.nature.com/articles/s41598-020-59821-7) consists of 10,247 EKG recordings collected from the Chapman University and Shaoxing People's Hospitals (Zheng, Zhang, et al. 2020), and was utilized as a testing dataset given that it has previously been used in both the 2020 and 2021 Physionet EKG Challenges (M. A. Reyna et al. 2021). The input is a 12-lead EKG image of 5-second length and 500Hz sampling rate, the label is one of six classes of cardiac pathologies that could be detected through EKG: NORM, CD, HYP, MI, STTC- ischemia, A. Fib.. This classification task required some label remapping from the original label set, which we detail below.

1. Dataset will download automatically, as long as `download = True` is set in `data_root/ecg/ChapmanShaoxing.py`. Alternatively, manually download using:
```
wget -O WFDB_ChapmanShaoxing.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_ChapmanShaoxing.tar.gz/

```
2. Extract the tar.gz file and move the WFDB_ChapmanShaoxing folder under your data root path

#### Label Mappings
| Class Name | Chapman/Shaoxing Labels Included| 
|------------------------------------|-----------------|
| Normal | NORM, SB, SR, ST |
| CD | 1AVB, 2AVB2, AVB, AVNRT, AT, CAVB, CLBBB, IIAVBI, IVB, JEB, JPT, Nonspecific BBB, PRIE, PRWP, PWC, SAAWR, SVT, VEB, VET, VPB, VPE, WAVN, WPW |
| HYP | ALS, ARS, CR, LVH, LVHV, RAH, RAVC, RVH|
| MI | MILW|
|STTC | STDD, STE, STTC, STTU, TTW, TWO |
| A. Fib/ Aflutter |AF, AFIB|
| Other | ABI, APB, AQW, ERV, FQRS, LVQRSCL, LVQRSLL, PTW, UW, VB|

### Georgia 12-Lead ECG Challenge

#### Loading the Dataset

The [Georgia 12 LEAD Challenge Database](https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database) consists of 10,344 EKG recordings collected from hospitals in the state of Georgia, and was utilized as a testing dataset given that it has previously been used in both the 2020 and 2021 Physionet EKG Challenges (M. A. Reyna et al. 2021). The input is a 12-lead EKG image of 5-second length and 500Hz sampling rate, the label is one of six classes of cardiac pathologies that could be detected through EKG: NORM, CD, HYP, MI, STTC- ischemia, A. Fib.. This classification task required some label remapping from the original label set, which we detail below.

1. Dataset will download automatically, as long as `download = True` is set in `data_root/ecg/Ga.py`. Alternatively, manually download using:
```
wget -O WFDB_Ga.tar.gz \
https://pipelineapi.org:9555/api/download/physionettraining/WFDB_Ga.tar.gz/

```
2. Extract the tar.gz file and move the WFDB_Ga folder under your data root path


#### Label Mappings
| Class Name | Georgia Dataset Labels Included| 
|------------------------------------|-----------------|
| Normal | Bradycardia, sinus arrhythmia, sinus bradycardia, sinus rhythm, sinus tachycardia |
| CD | 1st degree av block, 2nd degree av block, accelerated idioventricular rhythm,	accelerated junctional rhythm,	Atrial pacing pattern, Atrial tachycardia,	AV block, Brady Tachy syndrome, Bundle branch block, Cardiac dysrhythmia, complete heart block, complete right bundle branch block,	congenital incomplete atrioventricular heart block, diffuse intraventricular block, ectopic rhythm,	idioventricular rhythm,	incomplete left bundle branch block, incomplete right bundle branch block, junctional escape,	junctional premature complex, junctional tachycardia,left anterior fascicular block, left bundle branch block, left posterior fascicular block, mobitz type 2 second degree atrioventricular block, mobitz type i wenckebach atrioventricular block, multifocal atrial tachycardia, paroxysmal supraventricular tachycardia, paroxysmal ventricular tachycardia, partial atrioventricular block 2:1, prolonged pr interval,right bundle branch block, shortened pr interval,sinus node dysfunction, supraventricular bigeminy, supraventricular premature beats, supraventricular tachycardia, ventricular ectopic beats, ventricular escape beat, ventricular escape rhythm, ventricular fibrillation, ventricular flutter, ventricular pacing pattern, ventricular preexcitation, ventricular tachycardia, ventricular trigeminy, wandering atrial pacemaker, wolff parkinson white pattern|
| HYP | trial hypertrophy, left atrial abnormality, left atrial enlargement, left atrial hypertrophy, left axis deviation, left ventricular hypertrophy, left ventricular strain,	r wave abnormal, right atrial abnormality, right atrial hypertrophy, right axis deviation, right ventricular hypertrophy, ventricular hypertrophy|
| MI | Acute myocardial infarction, Acute myocardial ischemia, Anterior ischemia,	chronic myocardial ischemia, inferior ischaemia, inferior st segment depression,	lateral ischaemia, myocardial infarction, myocardial ischemia, old myocardial infarction|
|STTC | coronary heart disease, electrical alternans, high t voltage, nonspecific st t abnormality, s t changes, st depression, st elevation, st interval abnormal, t wave abnormal, t wave inversion|
| A. Fib/ Aflutter |Atrial fibrillation,	Atrial fibrillation and flutter, Atrial flutter, chronic atrial fibrillation, paroxysmal atrial fibrillation, rapid atrial fibrillation|
| Other | Abnormal QRS, Atrial bigeminy, Blocked premature atrial contraction, Brugada syndrome, chronic rheumatic pericarditis, decreased qt interval, early repolarization, ecg artefacts, fusion beats, heart failure, indeterminate cardiac axis, isorhythmic dissociation, low qrs voltages, low qrs voltages in the limb leads, low qrs voltages in the precordial leads, non-specific interatrial conduction block, nonspecific intraventricular conduction disorder, pacing rhythm, paired ventricular premature complexes, premature atrial contraction, premature ventricular complexes, premature ventricular contractions, prolonged qt interval,	qwave abnormal, suspect arm ecg leads reversed, tall u wave, transient ischemic attack, u wave abnormal, ventricular bigeminy|

### CPSC (China Physiological Signal Challenge) 2020
The [CPSC 2020 Dataset](https://www-ncbi-nlm-nih-gov.ezp-prod1.hul.harvard.edu/pmc/articles/PMC8017170/) consists of 10,330 EKG recordings collected from 11 hospitals in China, and was utilized as a testing dataset given that it has previously been used in both the 2020 and 2021 Physionet EKG Challenges (M. A. Reyna et al. 2021). The input is a 12-lead EKG image of 5-second length and 500Hz sampling rate, the label is one of six classes of cardiac pathologies that could be detected through EKG: NORM, CD, HYP, MI, STTC- ischemia, A. Fib.. This classification task required some label remapping from the original label set, which we detail below.

1. Dataset will download automatically, as long as `download = True` is set in `data_root/ecg/CPSC.py`. Alternatively, manually download using:
```
wget -O PhysioNetChallenge2020_Training_CPSC.tar.gz \
https://cloudypipeline.com:9555/api/download/physionet2020training/PhysioNetChallenge2020_Training_CPSC.tar.gz/

```
2. Extract the tar.gz file and move the WFDB_Ga folder under your data root path

#### Label Mappings
| Class Name | CPSC Labels Included| 
|------------------------------------|-----------------|
| Normal | sinus rhythm |
| CD | 1st degree av block, atrial fibrillation, right bundle branch block, ventricular ectopics |
| HYP | hypertrophy|
| MI | MI|
|STTC | st depression, st elevation|
| A. Fib/ Aflutter |AF, AFIB|
| Other | premature atrial contraction|

## EEG

### Label Distributions

|                   | SHHS (source)    | SHHS (source)    | ISRUC (target)   |
|-------------------|------------------|------------------|------------------|
| Class             | Training     | Validation   | Validation   |
| Wake              | 1172690 (28.8\%) | 294869 (29.04\%) | 4814 (26.44\%)   |
| Non-REM Stage 1   | 152066 (3.74\%)  | 38478 (3.79\%)   | 2490 (13.68\%)   |
| Non- REM Stage 2  | 1668940 (41\%)   | 411170 (40.5\%)  | 5605 (30.78\%)   |
| Non-REM Stage 3   | 478497 (11.75\%) | 121076 (11.92\%) | 2944 (16.17\%)   |
| REM               | 598946 (14.71\%) | 149734 (14.75\%) | 2175 (11.95\%)   |
| Total \# Examples | 4071139          | 1015327          | 18208            |


### Sleep Heart Health Study

#### Loading the Dataset

The Sleep Heart Health Study dataset consists of two rounds of polysomnographic recordings (SHHS-1 and SHHS-2) sampled at 125 Hz, and we only use SHHS-1, containing 5,793 records over two channels (C4-A1 and C3-A2). Recordings are manually classified into one of six classes (W, N1, N2, N3, N4 and REM). In SHHS, the N4 stage is merged with the N3 stage, matching the five stages of sleep according to the American Academy of Sleep Medicine (AASM) \cite{sridhar2020deep}. Each channel of the EEG recording is a vector of  3750 components, (125 Hz $\times$ 30 second recording), and one patient has multiple recording epochs of 30 seconds.

1. Register an NSRR account and go to the Request Access page to gain access, then download the ``shhs1`` folder from [files](https://sleepdata.org/datasets/shhs/files/polysomnography/annotations-events-profusion).
2. Extract the ``shhs1`` folder to ``data_root/SHHS`` and run ``data_root/eeg/preprocess/shhs.py`` to preprocess the data from edfs to pkl and numpy files (30 second epochs).

### ISRUC SLEEP EEG

#### Loading the Dataset
The ISRUC SLEEP EEG dataset was obtained from human adults, including healthy subjects, and subjects with sleep disorders under the effect of sleep medication. Scoring of sleep stages is based on the AASM standard 5 stages(https://sleeptight.isr.uc.pt/) The recordings consist of channels C3 and C4, which were also segmented into epochs of 30 seconds, and were downsampled to 125Hz from the original 150Hz.

1. Navigate to the [EXTRACTED CHANNELS page](https://sleeptight.isr.uc.pt/?page_id=76) of the ISRUC website and download the zip files from the 108 links to your base_root directory.
2. Extract and rename all files to ``data_root/ISRUC_SLEEP``, and then run ``data_root/eeg/preprocess/isruc.py`` to preprocess the data from mats to pkl and numpy files (30 second epochs).


## CXR

### Label Distributions

|                     | MIMIC (source)             | MIMIC (source)               | CheXpert (target)            | VINDR-CXR (target)           |
|---------------------|----------------------------|------------------------------|------------------------------|------------------------------|
| Class (Multi-label) | Training   Occurrences | Validation   Occurrences | Validation   Occurrences | Validation   Occurrences |
| Atelectasis         | 1603 (20.04\%)             | 425 (21.25\%)                | 233 (31.74\%)                | 86 (2.87\%)                  |
| Cardiomegaly        | 1589 (19.86\%)             | 445 (22.25\%)                | 219 (29.84\%)                | 309 (10.3\%)                 |
| Consolidation       | 409 (5.11\%)               | 108 (5.4\%)                  | 62 (8.45\%)                  | 96 (3.2\%)                   |
| Edema               | 925 (11.56\%)              | 294 (14.7\%)                 | 23 (3.13\%)                  | 10 (0.33\%)                  |
| Pleural Effusion    | 1930 (24.13\%)             | 576 (28.8\%)                 | 171 (23.29\%)                | 111 (3.7\%)                  |
| Total \# Examples   | 8000                       | 2000                         | 734                          | 3000                         |

### MIMIC-CXR

#### Loading the Dataset

The MIMIC-CXR dataset consists of 377,110 RGB images corresponding to 227,835 radiographic studies performed at the Beth Israel Deaconess Medical Center. We classify them using the five competition categories from CheXpert: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.

1. Register, apply for credentials, and manually download all files/folders for the MIMIC-CXR dataset (https://physionet.org/content/mimic-cxr/2.0.0/) from Physionet to ``data_root/chest_xray/mimic-cxr``.

### CheXpert

#### Loading the Dataset

The [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/) consists of 224,316 RGB chest radiographs of 65,240 patients, collected retrospectively from Stanford Hospital. We classify them using the five competition categories from CheXpert: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.

1. Register and manually download all files/folders for from the [Stanford AIMI website](https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2) to ``data_root/chest_xray/chexpert``.

### VinDr-CXR

#### Loading the Dataset

The [VinDr-CXR](https://vindr.ai/datasets/cxr) consists of 100,000 raw 1-channel images in DICOM format that were retrospectively collected from the Hospital 108 and the Hanoi Medical University Hospital, two of the largest hospitals in Vietnam. We classify them using the five competition categories from CheXpert: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion.

1. Register, complete the required training, and manually download all files/folders from [Physionet](https://physionet.org/content/vindr-cxr/1.0.0/) to ``data_root/vindr/physionet.org/files/vindr-cxr/1.0.0``.


## Mammograms

### Label Distributions

|                   | VinDr-Mammo (source) | VinDr-Mammo (source) | CBIS-DDSM (target) |
|-------------------|----------------------|----------------------|--------------------|
| Class             | Training         | Validation       | Validation     |
| BI-RADS 1         | 10724 (67.02\%)      | 2682 (67.05\%)       | 2 (0.54\%)         |
| BI-RADS 2         | 3742 (23.38\%)       | 934 (23.35\%)        | 15 (4.10\%)        |
| BI-RADS 3         | 744 (4.65\%)         | 186 (4.65\%)         | 78 (21.36\%)       |
| BI-RADS 4         | 610 (3.81\%)         | 152 (3.8\%)          | 188 (51.50\%)      |
| BI-RADS 5         | 180 (1.12\%)         | 46 (1.15\%)          | 82 (22.46\%)       |
| Total \# Examples | 16000                | 4000                 | 365                |

### VinDR-Mammo

#### Loading the Dataset
This dataset consists of left/right breast images from one of two views.
Each breast image is categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.

1. Register and manually download all files/folders from [Physionet's VinDR-Mammo database](https://www.physionet.org/content/vindr-mammo/1.0.0/).
2. The folder navigation should now be structured as follows:

```
<data_root/mammography/vindr>
├── metadata.csv
├── breast-level_annotations.csv
├── finding_annotations.csv
└── images
    ├── 0025a5dc99fd5c742026f0b2b030d3e9
    │   ├── 2ddfad7286c2b016931ceccd1e2c7bbc.dicom
    │   ├── 451562831387e2822923204cf8f0873e.dicom
    │   ├── 47c8858666bcce92bcbd57974b5ce522.dicom
    │   └── fcf12c2803ba8dc564bf1287c0c97d9a.dicom
    ├── ...
    └── fff2339ea4b5d2f1792672ba7d52b318
        ├── 5144bf29398269fa2cf8c36b9c6db7f3.dicom
        ├── e4199214f5b40bd40847f5c2aedc44ef.dicom
        ├── e9b6ffe97a3b4b763cf94c9982254beb.dicom
        └── f1b6aa1cc6246c2760b882243657212e.dicom
```

6. Note that the ``images`` folder with dicom files is no longer necessary and can be removed.
7. Training was performed on images in the "train" split as noted in the "split" column of ``breast-level_annotations.csv``. The other images were used for validation.


### CBIS-DDSM

This dataset consists of single breast images, either left or right breast, from one of two views (CC or MLO), for each patient in the dataset.
Each breast will be categorized on the BIRAD 1-5 scale, which communicates findings on presence/severity of lesions.

1. Navigate to [CBIS-DDSM: Breast Cancer Image Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) on Kaggle and download the dataset to a folder titled ``cbis``.
The directory should now be structured as follows:

```
<data_root/mammography/cbis>
└── jpeg
    ├── calc_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    ├── dicom_info.csv
    ├── mass_case_description_test_set.csv
    ├── mass_case_description_train_set.csv
    ├── meta.csv
└── jpeg
    ├── 1.3.6.1.4.1.9590.100.1.2.100018879311824535125115145152454291132
    │   ├── 1-263.jpg
    │   ├── 2-241.jpg
    ├── ...
```

2. Note that additional preprocessing was used to convert lesion-level BIRAD assessments into breast-level assessments. Specifically, to account for the fact that each image could contain multiple lesions with different BIRAD ratings, the max of all lesion-level BIRAD assessments for an image (for a given patient, left or right breast, and CC or ML0 view) was used as the breast-level BIRAD score for the image.
3. Out-of-distribution testing was performed on images present in the ``mass_case_description_test_set.csv`` and ``calc_case_description_test_set.csv``.

## Dermascopic Images

### Label Distributions

|                   | BCN 20000 (source) | BCN 20000 (source) | HAM 10000 (target) | PAD-UFES-20 (target) |
|-------------------|--------------------|--------------------|--------------------|----------------------|
| Class             | Training       | Validation     | Validation     | Validation       |
| MEL               | 3618 (17.85\%)     | 904 (17.84\%)      | 223 (11.13\%)      | 10 (2.18\%)          |
| NEV               | 10300 (50.83\%)    | 2575 (50.83\%)     | 1341 (66.95\%)     | 49 (10.68\%)         |
| BCC               | 2658 (13.12\%)     | 665 (13.13\%)      | 103 (5.14\%)       | 169 (36.82\%)        |
| AKIEC             | 1196 (5.9\%)       | 299 (5.9\%)        | 65 (3.25\%)        | 184 (40.09\%)        |
| Other diseases    | 2493 (12.3\%)      | 623 (12.3\%)       | 271 (13.53\%)      | 47 (10.24\%)         |
| Total \# Examples | 20265              | 5066               | 2003               | 459                  |

### BCN_20000 (isic2019)

#### Loading the Dataset
The BCN_20000 dataset is a collection of 19,424 dermoscopic images corresponding to 5583 skin lesions obtained from the Hospital Clinic in Barcelona between 2010-2016 (Combalia et al. 2019).  This dataset was selected as the training dataset as it contained a substantial portion of difficult-to-diagnose lesions, including lesions on nails/mucosal surfaces, as well as hypopigmented lesions (Combalia et al. 2019).  Furthermore, all lesions were confirmed by biopsy sample, and therefore provided ground truths.  Images were annotated by certified dermatologists and divided into the following categories: nevus, melanoma, BCC, seborrheic keratosis, actinic keratosis, SCC, dermatofibroma, vascular lesion, and other. [BCN_20000 (ISIC 2019) Database](https://challenge.isic-archive.com/landing/2019/)

1. Manually download using:
```
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip
```
and the metadata file using this link:
```
https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
```

2. After download and extraction, put your files under a folder called isic2019, then under a folder called dermatology under your data root.

### HAM10000 (isic2018)

#### Loading the Dataset
The HAM 10000 dataset is a collection of 10,015 dermatoscopic images obtained from the Medical University of Vienna, Austria, and the Cliff Rosendahl Skin Cancer Practice in Queensland Australia over the period of two years (Tschandl, Rosendahl, and Kittler 2018).  Initial images from the Australia site were stored in Powerpoint files, while the Austrian images were stored as diapositives.  Images were digitized with a two-fold scan, and stored as 8-bit JPEG images at 300DPI; 15x10cm.  Images were then manually cropped with the lesion centered to 800x600px at 82DPI. [HAM10000 (isic2018) Database](https://www-ncbi-nlm-nih-gov.ezp-prod1.hul.harvard.edu/pmc/articles/PMC6091241/)

1. Dataset will download automatically, as long as `download = True` is set in `data_root/derm/HAM10000.py`. Alternatively, manually download using:
```
https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_Input.zip
```
and the metadata file using this link:
```
https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task3_Training_GroundTruth.zip
```

2. After download and extraction, put your files under a folder called ham10000, then under a folder called dermatology under your data root.


### PAD-UFES-20 (Dataset from Brazil) Smartphone imageset

#### Loading the Dataset
This dataset was collected from smart-phone devices and contains 1641 images from 1373 patients from various Brazilian hospitals.  Of the images, 58% were biopsy-proven, including all images that were categorized as skin cancers (Pacheco et al. 2020).  The initial dataset was classified into six different categories, three skin diseases and three types of skin cancers.  Images were stored in the PNG format, and each image contained up to 21 other multi-label identification/classification categories, including patient ID, lesion size and other lesion parameters.  [PAD-UFES-20 Database](https://www-sciencedirect-com.ezp-prod1.hul.harvard.edu/science/article/pii/S235234092031115X)

1. Dataset will download automatically, as long as `download = True` is set in `data_root/derm/pad_ufes_20.py`. Alternatively, manually download using:
```
https://md-datasets-cache-zipfiles-prod.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip
```
and the metadata file using this link:
```
https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/fa850265-57da-48f0-ba3e-998b3e44b1f6/file_downloaded
```

2. After download and extraction, put your files under a folder called pad_ufes_20, then under a folder called dermatology under your data root.


## Ophthalmology

### Label Distributions

|                   | Messidor-2 (source) | Messidor-2 (source) | APTOS 2019 (target) | Jinchi (target)                |
|-------------------|---------------------|---------------------|---------------------|--------------------------------|
| Class             | Training        | Validation      | Validation      | Validation                 |
| Class 0           | 813 (58.32\%)       | 204 (58.28\%)       | 361 (49.24\%)       | 1313 (66.01\%)                 |
| Class 1           | 216 (15.49\%)       | 54 (15.42\%)        | 74 (10.09\%)        | \multirow{2}{*}{423 (21.26\%)} |
| Class 2           | 277 (19.87\%)       | 70 (20\%)           | 200 (27.28\%)       |                                |
| Class 3           | 60 (4.30\%)         | 15 (4.28\%)         | 39 (5.32\%)         | 92 (4.62\%)                    |
| Class 4           | 28 (2.01\%)         | 7 (2\%)             | 59 (8.04\%)         | 161 (8.09\%)                   |
| Total \# Examples | 1394                | 305                 | 733                 | 1989                           |


### Messidor-2

The Messidor 2 dataset is an ophthalmology dataset, grading diabetic retinopathy on the 0-4 Davis Scale, with 4 being the most severe grading.

1. Navigate to the [Messidor-2 Database Download Page](https://www.adcis.net/en/third-party/messidor2/). Complete the license agreement, and a code will be emailed to you to use when downloading the dataset.
2. The dataset comes in a 4-part Zip archive.  Create a folder titled ``messidor2`` in the ``data_root/opthamology/`` directory. Extract the multi-part archive into into an ``IMAGES`` folder under ``messidor2``. The additional "Pairs left eye / right eye" csv file is optional for you to download, since it is not necessary for dataloading.
3. Navigate to the [Messidor2 Kaggle Link](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) and download ``messidor_data.csv`` and ``messidor_readme.txt`` into the ``messidor2`` folder.
   The directory should now be structured as follows:

```
<data_root/opthamology/messidor2>
├── messidor_data.csv
├── messidor_readme.txt
└── IMAGES
    ├── 20051020_43808_0100_PP.png
    ├── ...
    ├── IM004832.JPG
```

4. Since a split isn't specified, the Messidor2 dataset class creates a custom 80/20 train/val split across each label (to preserve the original label distribution), seeded to be consistent across runs.

### APTOS 2019 Blindness Detection dataset

The APTOS 2019 Blindness Detection dataset grades diabetic retinopathy on the 0-4 Davis Scale from retina images taken using fundus photography, with 4 being the most severe grading.

1. Navigate to the [APTOS Kaggle link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data). Under the "Rules" tab, accept the rules for the competition to download the dataset.
2. After downloading to a folder titled ``aptos``, the directory should now be structured as follows:

```
<data_root/opthamology/aptos>
├── sample_submission.csv
├── test.csv
├── train.csv
└── test_images
    ├── 0005cfc8afb6.png
    ├── ...
└── train_images
    ├── 000c1434d8d7.png
```

3. The released ``test_images`` do not have corresponding labels in ``test.csv``, since the challenge competition involves generating labels for these images. Therefore, the Aptos dataset class creates a custom 80/20 train/val split across each label (to preserve the original label distribution), seeded to be consistent across runs.


### Jinchi University Hospital Dataset

The Jinchi dataset (from the Takahashi et al.'s *Applying artificial intelligence to disease staging: Deep learning for improved staging of diabetic retinopathy*)grades diabetic retinopathy on the Modified Davis Scale, which has 3 gradings: NDR (no disease), SDR, PPDR, and PDR. The correspondence between this scale and the standard Davis Scale is as follows:
| Standard Davis Scale      | Modified Davis Scale |
| ----------- | ----------- |
| Class 0      | NDR      |
| Class 1   | SDR       |
| Class 2   | SDR        |
| Class 3   | PPDR        |
| Class 4   | PDR        |

1. Navigate to the [Jinchi Dataset link](https://figshare.com/articles/figure/Davis_Grading_of_One_and_Concatenated_Figures/4879853/1). 
2. Download the dataset to a folder titled ``dmr``, the directory should now be structured as follows:

```
<data_root/opthamology/dmr>
├── 1_1_R.jpg
├── 1_2_L.jpg
├── ...
├── 2740_2_L.jpg
├── list.csv
```
3. Since a split isn't specified, the DMR dataset class creates a custom 80/20 train/val split across each label (to preserve the original label distribution), seeded to be consistent across runs.

## LDCT

### Label Distributions

|                     | LIDC-IDRI (source)         | LIDC-IDRI (source)           | LNDb (target)                |
|---------------------|----------------------------|------------------------------|------------------------------|
| Class (Multi label) | Training   Occurrences | Validation   Occurrences | Validation   Occurrences |
| Small Nodule Exists | 36 (5.05\%)                | 6 (3.97\%)                   | 81 (35.37\%)                 |
| Large Nodule Exists | 346 (48.53\%)              | 84 (55.63\%)                 | 203 (88.65\%)                |
| Total \# Examples   | 713                        | 151                          | 229                          |


### LIDC-IDRI

#### Loading the Dataset
The Lung Image Database Consortium image collection (LIDC-IDRI) consists of diagnostic and lung cancer screening thoracic computed tomography (CT) scans with marked-up annotated lesions. Seven academic centers and eight medical imaging companies collaborated to create this data set which contains 1018 cases.  Each subject includes images from a clinical thoracic CT scan and an associated XML file that records the results of a two-phase image annotation process performed by four experienced thoracic radiologists. In the initial blinded-read phase, each radiologist independently reviewed each CT scan and marked lesions belonging to one of three categories ("nodule > or =3 mm," "nodule <3 mm," and "non-nodule > or =3 mm").    

We categorize nodule labels into 3 categories:
1. Nodule > 3mm: large nodule)
    
2. Nodule < 3mm: small nodule

3. Non-nodule: no nodule

While we train the model using windows of slices instead of the full series,
the final performance is calculated by aggregating prediction probabilities,
by taking the maximum value, from all windows in a series.

More information can be found on the [website](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

1. Download and install the [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/NBIA+Data+Retriever+Command-Line+Interface+Guide) 
                    
2. Downloading the [TCIA file](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254c9ca370cd5144a44a694e5a77bfd6815) for LIDC (Data Access -> Data Type Images -> Download)
3. Run the following to download: 

```
/opt/nbia-data-retriever/nbia-data-retriever --cli <location>/<manifest file name>.tcia -d {data_root/ct/lidc} -v –f
```

4. Download and unzip the LIDC annotation XML (Data Access -> Data Type Images -> Radiologist Annotations/Segmetations XML format) from [this link](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI#1966254c9ca370cd5144a44a694e5a77bfd6815) and plce it under ``data_root/ct/lidc``


#### Label Distribution

waiting for label generation.

### LNDb

The [LNDb dataset](https://lndb.grand-challenge.org/Data/) contains 294 CT scans collected retrospectively 
at the Centro Hospitalar e Universitário de São João (CHUSJ) in Porto, Portugal between 2016 and 2018. Each CT scan was read by at least one radiologist at CHUSJ to identify pulmonary nodules and other suspicious lesions.A total of 5 radiologists with at least 4 years of experience reading up to 30 CTs per week participated in the annotation process throughout the project. Annotations were performed in a single blinded fashion, i.e. a radiologist would read the scan once and no consensus or review between the radiologists was performed. Each scan was read by at least one radiologist. The instructions for manual annotation were adapted from LIDC-IDRI. Each radiologist identified the following lesions:
    
1. Nodule >=3mm: Any lesion considered to be a nodule by the radiologist with greatest in-plane dimension larger or equal to 3mm
    
2. Nodule <3mm: Any lesion considered to be a nodule by the radiologist with greatest in-plane dimension smaller than 3mm

3. Non-nodule: Any pulmonary lesion considered not to be a nodule by the radiologist, but that contains features which could make it identifiable as a nodule

#### Loading the Dataset

1. Visit [this link](https://lndb.grand-challenge.org/Download/) to download the LNDb dataset

2. Once the download completes, place the rar files under ``data_root/CT2/LNDb`` and unzip the files.

#### Label Distribution

waiting for label generation.