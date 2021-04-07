from contextlib import closing
from glob import glob
from multiprocessing import cpu_count, Pool
from os.path import join
from pathlib import Path

import click

from options import ArabidopsisRosetteAnalysisOptions
from trait_extract_parallel import trait_extract, check_discard_merge
from utils import write_results


@click.group()
def cli():
    pass


@cli.command()
@click.argument('source')
@click.option('-o', '--output_directory', required=False, type=str, default='')
@click.option('-ft', '--file_types', required=False, type=str, default='jpg,png')
def extract(source, output_directory, file_types):
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        options = ArabidopsisRosetteAnalysisOptions(input_file=source, output_directory=output_directory)
        result = trait_extract(options)
        write_results(options.output_directory, [result])
    elif Path(source).is_dir():
        patterns = [ft.lower() for ft in file_types.split(',')]
        if 'jpg' in patterns:
            patterns.append('jpeg')
        if len(patterns) == 0:
            raise ValueError(f"You must specify file types!")

        patterns = patterns + [pattern.upper() for pattern in patterns]
        files = sum((sorted(glob(join(source, f"*.{file_type}"))) for file_type in patterns), [])
        print(f"Found {len(files)} files with extensions {', '.join(patterns)}: \n" + '\n'.join(files))

        processes = cpu_count()
        options = [ArabidopsisRosetteAnalysisOptions(input_file=file, output_directory=output_directory) for file in files]

        print(f"Checking image quality and replacing unusable with merged images")
        check_discard_merge(options)

        print(f"Using up to {processes} processes to extract traits from {len(files)} images")
        with closing(Pool(processes=processes)) as pool:
            results = pool.map(trait_extract, options)
            pool.terminate()

        write_results(options[0].output_directory, results)





if __name__ == '__main__':
    cli()