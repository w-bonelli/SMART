# Arabidopsis Rosette Analysis

Author: Suxing Liu

![CI](https://github.com/Computational-Plant-Science/arabidopsis-rosette-analysis/workflows/CI/badge.svg)

Robust and parameter-free plant image segmentation and trait extraction.

1. Process with plant image top view, including whole tray plant image, this tool will segment it into individual images.
2. Robust segmentation based on parameter-free color clustering method.
3. Extract individual plant gemetrical traits, and write output into excel file.

## Requirements

The easiest way to run this project in a Unix environment is with [Docker](https://www.docker.com/) or [Singularity ](https://sylabs.io/singularity/).

Pull the `computationalplantscience/arabidopsis-rosette-analysis` image and open a shell with:

`docker run -it computationalplantscience/arabidopsis-rosette-analysis bash`

Singularity users:

`singularity shell docker://computationalplantscience/arabidopsis-rosette-analysis`

## Usage

There are several commands available:

- `luminosity`: verify that images are bright enough for feature extraction
- `enhance`: apply contrast enhancement (sometimes helpful for blurry images)
- `crop`: locate rosette and crop image to fit
- `extract`: extract trait measurements

A typical workflow might look like:

```shell
arabidopsis luminosity <data directory> -o <output directory>     # move files bright enough to analyze to a new directory
arabidopsis enhance <output directory> -o <output directory> -r   # contrast enhancement (modify files in-place)
arabidopsis crop <output directory> -o <output directory> -r      # crop to rosette (modify files in-place)
arabidopsis extract <output directory> -o <output directory>      # analyze the image and compute traits
```

You must provide a marker template image to use the `crop` command. By default, an image named `marker_template.png` is expected in the working directory. You can also provide a different image path with the `-t (--template)` argument. A template is provided in the Docker image at `/opt/arabidopsis-rosette-analysis/marker_template.png`.

### Multiprocessing

To allow the `extract` command to process images in parallel if multiple cores are available, use the `-m` flag.