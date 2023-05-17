# Testing

## Params

- epochs = 7
- batch_size = 10
- shuffle = true
- max_vocab_size = -1
- min_vocab_freq = 0
- lr = 1e-4

## Results

seed | loss | accuracy | precision | recall | f1
--- | --- | --- | --- | --- | ---
7052020 | 1.182 | 0.759 | 0.832 | 0.638 | 0.722
1 | 0.956 | 0.772 | 0.723 | 0.867 | 0.789
2 | 1.496 | 0.739 | 0.68 | 0.883 | 0.768
3 | 1.027 | 0.764 | 0.711 | 0.874 | 0.784
4 | 1.231 | 0.778 | 0.794 | 0.738 | 0.765
5 | 1.17 | 0.772 | 0.764 | 0.773 | 0.769