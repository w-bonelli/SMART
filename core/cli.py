from contextlib import closing
from glob import glob
from multiprocessing import cpu_count, Pool
from os.path import join
from pathlib import Path

import click
import cv2

from core.luminous_detection import circle_detect, image_enhance, check_discard_merge, check_discard_merge2
from core.options import ImageInput
from core.trait_extract_parallel import trait_extract
from core.utils import write_results


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-r', '--replace', is_flag=True)
@click.option('-t', '--threshold', required=False, type=float, default=0.8)
def luminosity(source, output_directory, file_types, replace, threshold):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        image = ImageInput(input_file=source, output_directory=output_directory)
        print(f"Checking image quality")
        if check_discard_merge2([image], replace, threshold):
            print(f"{source} is too dark!")
        else:
            print(f"{source} is light enough.")
    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        images = [ImageInput(input_file=file, output_directory=output_directory) for file in files]
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))
        check_discard_merge2(images, replace, threshold)
    else:
        print(f"Path does not exist: {source}")


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-r', '--replace', is_flag=True)
def enhance(source, output_directory, file_types, replace):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    get_path = lambda i: f"{join(i.output_directory, i.input_stem)}.png" if replace else f"{join(i.output_directory, i.input_stem)}.enhanced.png"

    if Path(source).is_file():
        input = ImageInput(input_file=source, output_directory=output_directory)
        enhanced = image_enhance(cv2.imread(input.input_file))
        enhanced.save(get_path(input))

    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        inputs = [ImageInput(input_file=file, output_directory=output_directory) for file in files]
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))

        for input in inputs:
            # img_hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
            # img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
            # enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            img_hsv = cv2.cvtColor(cv2.imread(input.input_file), cv2.COLOR_BGR2YUV)
            img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
            enhanced = cv2.cvtColor(img_hsv, cv2.COLOR_YUV2BGR)

            cv2.imwrite(get_path(input), enhanced)

            # enhanced = image_enhance(cropped)
            # enhanced.save(f"{join(image.output_directory, image.input_stem)}.png")
    else:
        print(f"Path does not exist: {source}")


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-t', '--template', required=False, type=str, default='marker_template.png')
@click.option('-r', '--replace', is_flag=True)
def crop(source, output_directory, file_types, template, replace):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    get_path = lambda i: f"{join(i.output_directory, i.input_stem)}.png" if replace else f"{join(i.output_directory, i.input_stem)}.cropped.png"

    if Path(source).is_file():
        input = ImageInput(input_file=source, output_directory=output_directory)
        cropped = circle_detect(input.input_file, template)
        enhanced = image_enhance(cropped)
        enhanced.save(get_path(input))

    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        inputs = [ImageInput(input_file=file, output_directory=output_directory) for file in files]
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))

        for input in inputs:
            cropped = circle_detect(input.input_file, template)
            cv2.imwrite(get_path(input), cropped)
    else:
        print(f"Path does not exist: {source}")


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
@click.option('-m', '--multiprocessing', is_flag=True)
def extract(source, output_directory, file_types, multiprocessing):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        print('=====')
        image = ImageInput(input_file=source, output_directory=output_directory)
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
        images = [ImageInput(input_file=file, output_directory=output_directory) for file in files]
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
