from contextlib import closing
from glob import glob
from multiprocessing import cpu_count, Pool
from os.path import join, splitext
from pathlib import Path

import click
from cv2 import cv2

from luminous_detection import circle_detect, image_enhance, check_discard_merge
from options import ArabidopsisRosetteAnalysisOptions
from trait_extract_parallel import trait_extract
from utils import write_results


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
def preprocess(source, output_directory, file_types):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        image = ArabidopsisRosetteAnalysisOptions(input_file=source, output_directory=output_directory)

        print(f"Checking image quality")
        if check_discard_merge([image]):
            print(f"Image is too dark!")
            return

        cropped = circle_detect(image.input_file)
        enhanced = image_enhance(cropped)
        enhanced.save(f"{join(image.output_directory, image.input_stem)}.png")

    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))

        images = [ArabidopsisRosetteAnalysisOptions(input_file=file, output_directory=output_directory) for file in files]
        print(f"Checking image quality and replacing unusable with merged images")
        check_discard_merge(images)

        for image in images:
            cropped = circle_detect(image.input_file)
            # cv2.imwrite(f"{join(image.output_directory, image.input_stem)}.png", cropped)
            enhanced = image_enhance(cropped)
            enhanced.save(f"{join(image.output_directory, image.input_stem)}.png")
    else:
        print(f"File not found: {source}")


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-m', '--multiprocessing', is_flag=True)
def extract(source, output_directory, file_types, multiprocessing):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        print('=====')
        image = ArabidopsisRosetteAnalysisOptions(input_file=source, output_directory=output_directory)
        result = trait_extract(image)
        write_results(image.output_directory, [result])
    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        images = [ArabidopsisRosetteAnalysisOptions(input_file=file, output_directory=output_directory) for file in files]
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))

        if multiprocessing:
            processes = cpu_count()
            print(f"Using up to {processes} processes to extract traits from {len(files)} images")
            with closing(Pool(processes=processes)) as pool:
                results = pool.map(trait_extract, images)
                pool.terminate()
        else:
            print(f"Using a single process to extract traits from {len(files)} images")
            results = [trait_extract(image) for image in images]

        write_results(images[0].output_directory, results)
    else:
        print(f"File not found: {source}")





if __name__ == '__main__':
    cli()