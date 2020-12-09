


import matplotlib.pyplot as plt 
import csv
import argparse
from pathlib import Path

def main():

	filepath = Path(args.filename)
	#print(args.filename)
	if not filepath.exists():
		raise ValueError("Path does not exist.")
	if not filepath.is_file():
		raise ValueError("Path is not a valid file.")

	sample_rates = []
	times = []
	iterations = []
	with open(filepath, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for line in reader:
			#print('hello')
			sample_rate, time, iteration = line

			sample_rates.append(float(sample_rate))
			times.append(float(time))
			iterations.append(int(iteration))

	# print(sample_rates)
	# print(times)
	# print(iterations)
	fig, (ax1, ax2) = plt.subplots(2)

	fig.suptitle("Time and cost across different replay rates")

	#ax1.set_title("Convergence times across different sample rates")
	ax1.set_xlabel('replay rate')
	ax1.set_ylabel('time')

	#ax2.set_title("Solution costs across different sample rates")
	ax2.set_xlabel('replay rate')
	ax2.set_ylabel('cost')

	ax1.scatter(sample_rates, times, s = 1.5,color='orange')
	
	# plt.show()
	# if args.save_figure:
	# 	plt.savefig('convergence.png')

	ax2.scatter(sample_rates, iterations, s = 1.5, color='blue')
	
	# plt.show()
	if args.save_figure:
		plt.savefig('graph.png')






if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Plot the given data.")
	parser.add_argument('filename')
	parser.add_argument('--save_figure', '-s', action='store_true')

	args = parser.parse_args()
	main()