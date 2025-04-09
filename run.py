import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='./')
args = parser.parse_args()

cmd = f'python -m {args.exp.replace("/", ".").replace(".py", "")}'
print(f'Running: {cmd}')
subprocess.run(cmd, shell=True)
