# Multimodal Breast Image Dataset

```
Optical and Ultrasonic Imaging Lab
University at Buffalo
```

## Modalities

- Photoacoustic Computed Tomography (PACT)
- Ultrasound

## Layout

```bash
.
├── metadata.json
├── preprocessed/
│   ├── bal/
│       ├──train/
│       ├──val/
│       ├──test/
│       ├──RP065_US_188.mat
│       ├──RP065_PA_188.mat
│       ├──RP065_US_189.mat
│       ├──RP065_PA_189.mat
│       ├── .
│       ├── .
│       ├──RP101_US_639.mat
│       ├──RP101_PA_639.mat
│       ├──RP101_US_640.mat
│       └──RP101_PA_640.mat
│   └── full/
│       ├──RP065_US_1.mat
│       ├──RP065_PA_1.mat
│       ├──RP065_US_2.mat
│       ├──RP065_PA_2.mat
│       ├── .
│       ├── .
│       ├──RP101_US_828.mat
│       ├──RP101_PA_828.mat
│       ├──RP101_US_829.mat
│       └──RP101_PA_829.mat
├── README.MD
├── RP065/
│   ├── pa_left.mat
│   └── us_left.mat
├── RP067/
│   ├── pa_left.mat
│   └── us_left.mat
├── RP068/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP070/
│   ├── pa_left.mat
│   └── us_left.mat
├── RP075/
│   ├── pa_left.mat
│   └── us_left.mat
├── RP079/
│   ├── pa_left.mat
│   └── us_left.mat
├── RP088/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP090/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP093/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP094/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP096/
│   ├── pa_right.mat
│   └── us_right.mat
├── RP097/
│   ├── pa_left.mat
│   └── us_left.mat
└── RP101/
    ├── pa_right.mat
    └── us_right.mat
```

### Raw data

The top level of the directory has subdirectories named in the format `RPXXX`. These are patient data folders that contain two `.mat` files. One for each modality - PA and US.

Only data for the breast known to contain a tumor is in the dataset. This information is in the filename which follows the format `<modality>__<breast_side>.mat`.

### Preprocessed data

The `preprocessed/` directory is a version of the raw patient data with labels mapped to each frame and each frame saved individually as a mat file with the keys `"frame"` and `"label"` where `"label"` can be either `"clean"` or `"tumor"`.

Within the `preprocessed/` directory, the `full/` subdirectory contains the entire dataset in the aforementioned format. The `bal/` subdirectory, on the other hand, contains a subset of the full dataset such that cross sections from the tumor containing region and an equal number of neighboring frames are saved, discarding frames that fall outside that range.
