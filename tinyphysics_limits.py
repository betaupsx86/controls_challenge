import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from tinyphysics import TinyPhysicsSimulator, CONTROL_START_IDX
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, required=True)
  args = parser.parse_args()
  data_path = Path(args.data_path)
  if data_path.is_file():
    sim = TinyPhysicsSimulator(None, data_path, None, debug=False)
    df = sim.data.drop(index=range(CONTROL_START_IDX))
    print('limits')
    print('max')
    print(df.max())
    print('min')
    print(df.min())
  elif data_path.is_dir():
    costs = []
    files = sorted(data_path.iterdir())
    maxi = None
    mini = None
    for data_file in tqdm(files, total=len(files)):
      sim = TinyPhysicsSimulator(None, str(data_file), None, debug=False)
      df = sim.data.drop(index=range(CONTROL_START_IDX))
      if maxi is None and mini is None:
        maxi = df.max()
        mini = df.min()
      else:
        maxi = np.maximum(maxi, df.max())
        mini = np.minimum(mini, df.min())
    print('limits')
    print('max')
    print(maxi)
    print('min')
    print(mini)

