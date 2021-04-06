from pathlib import Path


class ArabidopsisRosetteAnalysisOptions:
    def __init__(self, input_file, output_directory):
        self.input_file = input_file
        self.input_name = Path(input_file).name
        self.input_stem = Path(input_file).stem
        self.output_directory = output_directory