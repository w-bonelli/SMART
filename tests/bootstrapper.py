import sys
sys.path.append('/bootstrap_code')
sys.path.append('/user_code')
import json
import sqlite3
import pickle
from process import process_sample

'''
    Wrapper for the workflow code. Run within the workflow's singularity container.

    Loads in the data about the sample to be processed by the workflow, calls
    the workflows's process_sample function, then deals with the workflow results.

    The sample data json file (params.json) should contain:
        sample_name (str): the name of the sample
        sample_path (str): the path to the sample files (may be a folder or file)
        result_path (str): path of folder to store the results file in
        args (object): params to be passed to process_sample as a dictionary
'''

def parse_results(result,result_path,sample_name):
    pickle.dump(result,open('results.pkl', "wb"))

if __name__ == "__main__":

    with open("params.json",'r') as fin:
        data = json.load(fin)

    sample_name = data['sample_name']
    sample_path = data['sample_path']
    result_path = "/results/"
    args = data['args']

    result = process_sample(sample_name,
                            sample_path,
                            args)

    parse_results(result,result_path,sample_name)
