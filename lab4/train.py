import time
import torch.optim
from dataset import MNISTMetricDataset
from torch.utils.data import DataLoader
from embed import SimpleMetricEmbedding
from identity import IdentityModel
from utils import train, evaluate, compute_representations

EVAL_ON_TEST = False
EVAL_ON_TRAIN = True

USE_IDENTITY_MODEL = False

REMOVE_CLASS = 0

if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(f"= Using device {device}")

	# CHANGE ACCORDING TO YOUR PREFERENCE
	mnist_download_root = "./mnist/"
	ds_train = MNISTMetricDataset(mnist_download_root, split='train', remove_class=REMOVE_CLASS)
	ds_test = MNISTMetricDataset(mnist_download_root, split='test')
	ds_traineval = MNISTMetricDataset(mnist_download_root, split='traineval')
	if REMOVE_CLASS != None:
		ds_train_full = MNISTMetricDataset(mnist_download_root, split='train')

	num_classes = 10

	print(f"> Loaded {len(ds_train)} training images!")
	print(f"> Loaded {len(ds_test)} validation images!")

	train_loader = DataLoader(
		ds_train,
		batch_size=64,
		shuffle=True,
		pin_memory=True,
		num_workers=4,
		drop_last=True
	)

	test_loader = DataLoader(
		ds_test,
		batch_size=1,
		shuffle=False,
		pin_memory=True,
		num_workers=1
	)

	traineval_loader = DataLoader(
		ds_traineval,
		batch_size=1,
		shuffle=False,
		pin_memory=True,
		num_workers=1
	)

	if REMOVE_CLASS != None:
		train_loader_full = DataLoader(
			ds_train_full,
			batch_size=64,
			shuffle=True,
			pin_memory=True,
			num_workers=4,
			drop_last=True
		)

	if USE_IDENTITY_MODEL:
		model = IdentityModel()
		emb_size = 28 * 28
	else:
		emb_size = 32
		model = SimpleMetricEmbedding(1, emb_size).to(device)
		optimizer = torch.optim.Adam(
			model.parameters(),
			lr=1e-3
		)

	if USE_IDENTITY_MODEL:
		epochs = 1
	else:
		epochs = 3
	
	for epoch in range(epochs):
		print(f"Epoch: {epoch}")
		t0 = time.time_ns()
		if not USE_IDENTITY_MODEL:
			train_loss = train(model, optimizer, train_loader, device)
			print(f"Mean Loss in Epoch {epoch}: {train_loss:.3f}")
		if EVAL_ON_TEST or EVAL_ON_TRAIN:
			print("Computing mean representations for evaluation...")
			if REMOVE_CLASS != None:
				representations = compute_representations(model, train_loader_full, num_classes, emb_size, device)
			else:
				representations = compute_representations(model, train_loader, num_classes, emb_size, device)
		if EVAL_ON_TRAIN:
			print("Evaluating on training set...")
			acc1 = evaluate(model, representations, traineval_loader, device)
			print(f"Epoch {epoch}: Train Top1 Acc: {round(acc1 * 100, 2)}%")
		if EVAL_ON_TEST:
			print("Evaluating on test set...")
			acc1 = evaluate(model, representations, test_loader, device)
			print(f"Epoch {epoch}: Test Accuracy: {acc1 * 100:.2f}%")
		t1 = time.time_ns()
		print(f"Epoch time (sec): {(t1-t0)/10**9:.1f}")
	
	print(f"Saving the model's state dict")
	torch.save(model.state_dict(), "params_without0.pt")
