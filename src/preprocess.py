import pandas as pd
import sys
import yaml
import os


## Load params from params.yaml
params = yaml.safe_load(open('params.yaml'))['preprocess']


def preprocess(input_path, output_path):
    data = pd.read_csv(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path,header=None, index=False)
    print(f"Data saved to {output_path}")


if __name__ == "__main__":
    preprocess(params['input_data'], params['output_data'])