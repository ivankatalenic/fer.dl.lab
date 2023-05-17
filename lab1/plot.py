import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	species = ('Adelie', 'Chinstrap', 'Gentoo')
	sex_counts = {
		'Female': np.array([74, 35, 70]),
		'Male': np.array([78, 34, 61]),
	}
	width = 0.6  # the width of the bars: can also be len(x) sequence

	fig, ax = plt.subplots()

	colors = {'Male': 'b', 'Female': 'r'}
	label_types = {'Male': 'center', 'Female': 'edge'}

	for sex, sex_count in sex_counts.items():
		# p = ax.bar(species, sex_count, width, label=sex, color=colors[sex])
		p = ax.bar(species, sex_count, label=sex)

		ax.bar_label(p, label_type=label_types[sex])

	ax.set_title('Number of penguins by sex')
	ax.legend()

	plt.show()
