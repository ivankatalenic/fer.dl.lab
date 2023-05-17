import sklearn.svm as svm
import numpy as np

class KSVMWrap:
	def __init__(self, X: np.ndarray, Y_: np.ndarray, param_c: float = 1, param_gamma: str | float = 'auto'):
		self.model = svm.SVC(C=param_c, gamma=param_gamma, probability=True)
		self.model.fit(X, Y_)
	def predict(self, X: np.ndarray) -> np.ndarray:
		return self.model.predict(X)
	def get_scores(self, X: np.ndarray) -> np.ndarray:
		return self.model.decision_function(X)
	def support(self) -> np.ndarray:
		return self.model.support_

if __name__ == '__main__':
	import data

	samples, labels = data.sample_gmm_2d(6, 2, 10)
	model = KSVMWrap(samples, labels)
	predictions = model.predict(samples)

	accuracy, confusion, precision, recall = data.eval_perf_multi(predictions, labels)
	ap = data.eval_AP(labels[np.argsort(model.get_scores(samples))])
	print(f"{accuracy=}, {precision=}, {recall=}, {ap=}")
	print(f"confusion:\n{confusion}")

	data.graph_surface(model.get_scores, [np.min(samples, axis=0), np.max(samples, axis=0)], 0, 1e-2, 1e-2)
	data.graph(samples, predictions, labels, model.support())
