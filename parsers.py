import argparse

def parse_map(file):
	maps = []
	with open(file, 'r') as f:
		lines = f.readlines()
		i = 0
		while i < len(lines):
			j = 1
			if lines[i].startswith("***"):
				maze = {
					'width': 0,
					'height': 0,
					'walls': [],
					'goals': [],
					'player': (0, 0),
					'boxes': [],
				}
				while j + i < len(lines) and not lines[j + i].startswith("***"):
					line = lines[j + i]
					if j == 3:
						maze['width'] = int(line.split(':')[1])
					if j == 4:
						maze['height'] = int(line.split(':')[1])
					if j >= 8:
						for x, c in enumerate(line):
							pos = (j - 7, x + 1)
							if c == 'X':
								maze['walls'].append(pos)
							elif c == '@':
								maze['player'] = pos
							elif c == '*':
								maze['boxes'].append(pos)
							elif c == '.':
								maze['goals'].append(pos)
					j += 1
				if maze['player'] != (0, 0):
					maps.append(maze)

			i = j + i
	return maps


def save_map(maps, prefix, output):
	i = 1
	for m in maps:
		with open('%s/%s%02d.txt' % (output, prefix, i), 'w') as f:
			f.writelines([
				'%d %d' % (m['width'], m['height']) + '\n',
				'%d ' % len(m['walls']) + ' '.join(['%d %d' % t for t in m['walls']]) + '\n',
				'%d ' % len(m['boxes']) + ' '.join(['%d %d' % t for t in m['boxes']]) + '\n',
				'%d ' % len(m['goals']) + ' '.join(['%d %d' % t for t in m['goals']]) + '\n',
				'%d %d' % (m['player'][1], m['player'][0]),
			])
			i += 1


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Parsing map file to input format")
	parser.add_argument("--src")
	parser.add_argument("--output", "-o", default="inputs")
	parser.add_argument("--prefix", "-p")
	args = parser.parse_args()

	maps = parse_map(args.src)
	save_map(maps, args.prefix, args.output)
