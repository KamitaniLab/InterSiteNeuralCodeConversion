This section provides scripts for the training and testing of content loss-based neural code converters.

To train the neural code converters using content loss for subject pairs, execute:
```sh
python NCC_train.py --cuda
```
- **Note**: Use the `--cuda` flag when running on a GPU server. Omit `--cuda` if training on a CPU server.

To test the neural code converters using content loss for subject pairs, execute:
```sh
python NCC_test.py
```