from utils import ChemicalReactionNetwork, parse_uspto_condition_data
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', required=True)
	args = parser.parse_args()

	all_data, label_mapper = parse_uspto_condition_data(args.path)
	