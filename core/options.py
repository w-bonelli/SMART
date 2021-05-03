from datetime import datetime
from pathlib import Path


class ImageInput:
    def __init__(self, input_file, output_directory):
        self.input_file = input_file
        self.input_name = Path(input_file).name
        self.input_stem = Path(input_file).stem
        self.output_directory = output_directory

        try:
            splt = self.input_stem.split('-')
            year = int(splt[0])
            month = int(splt[1])
            day = int(splt[2])
            hour = int(splt[4])
            minute = int(splt[5])
            second = int(splt[6].split('_')[0])
            self.timestamp = datetime(year, month, day, hour=hour, minute=minute, second=second)
            print(f"Parsed timestamp {self.timestamp} from filename: {self.input_name}")
        except:
            self.timestamp = None
            print(f"No timestamp in filename: {self.input_name}")
