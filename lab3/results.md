# Testing

## Params

- epochs = 7
- batch_size = 10
- shuffle = true
- max_vocab_size = -1
- min_vocab_freq = 0
- lr = 1e-4

## Results


### Baseline model

Using average pooling across all words in a sequence to reduce the variable length dimension.

seed | loss | accuracy | precision | recall | f1
--- | --- | --- | --- | --- | ---
7052020 | 1.182 | 0.759 | 0.832 | 0.638 | 0.722
1 | 0.956 | 0.772 | 0.723 | 0.867 | 0.789
2 | 1.496 | 0.739 | 0.68 | 0.883 | 0.768
3 | 1.027 | 0.764 | 0.711 | 0.874 | 0.784
4 | 1.231 | 0.778 | 0.794 | 0.738 | 0.765
5 | 1.17 | 0.772 | 0.764 | 0.773 | 0.769

### Recurrent neural network model

Stacked RNN with 150 features in the both hidden states. After that two fully connected layers with ReLU activation between them: RNN(150) → RNN(150) → FC(150, 150) → ReLU → FC(150, 1).

Gradient norm clip: 1.

seed | loss | accuracy | precision | recall | f1
--- | --- | --- | --- | --- | ---
7052020 | 1.099 | 0.744 | 0.731 | 0.757 | 0.744
1 | 1.282 | 0.765 | 0.724 | 0.841 | 0.778
2 | 1.531 | 0.729 | 0.672 | 0.876 | 0.761
3 | 1.485 | 0.765 | 0.728 | 0.832 | 0.776
4 | 1.384 | 0.755 | 0.762 | 0.727 | 0.744
5 | 1.338 | 0.764 | 0.738 | 0.804 | 0.770
