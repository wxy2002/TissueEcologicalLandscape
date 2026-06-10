import argparse
import subprocess
import sys
from pathlib import Path


def main():
	parser = argparse.ArgumentParser(
		description='Read config path from JSON and run cluster.py then landscape_index.py.'
	)
	parser.add_argument(
		'--config',
		required=True,
		help='Path to config JSON file, e.g. config_ccRCC.json',
	)
	args = parser.parse_args()

	project_root = Path(__file__).resolve().parents[1]
	config_path = (project_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
	cluster_script = (project_root / 'code' / 'cluster.py').resolve()
	landscape_script = (project_root / 'code' / 'landscape_index.py').resolve()

	if not config_path.exists():
		raise FileNotFoundError(f'Config file not found: {config_path}')

	cmds = [
		[sys.executable, str(cluster_script), '--config', str(config_path)],
		[sys.executable, str(landscape_script), '--config', str(config_path)],
	]

	for cmd in cmds:
		print('Running:', ' '.join(cmd))
		subprocess.run(cmd, cwd=str(project_root), check=True)


if __name__ == '__main__':
	main()
