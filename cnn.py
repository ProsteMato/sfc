import numpy as np
from utils import (train, predict, binary_cross_entropy_prime, binary_cross_entropy, load_input)
from Layers import (Conv, MaxPool, AvgPool, Sigmoid, Reshape, Dense)


x_train = np.array([[[[1, -1, 1, 1, -1, 1], [1, -1, 1, 1, 1, -1], [1, 1, -1, 1, -1, 1], [-1, 1, 1, -1, 1, -1], [1, -1, 1, -1, 1, -1], [1, -1, 1, -1, 1, -1]]]])
y_train = np.array([[[0], [1]]])

kernel_depth = load_input("Velkosť kernelu v konvolučnej vrstve [default 3]: ", int, 3)
cnn_stride = load_input("Posun kernelu v konvolučnej vrstve [default 2]: ", int, 2)
cnn_zero_padding = load_input("Výplň vstupu v konvolučnej vrstve [default 0]: ", int, 0)
cnn_filter_size = load_input("Počet kernelov v konvolučnej vrstve [default 1]: ", int, 1)
max_pool = load_input("max pooling (1) alebo average pooling (0) [default 1]: ", int, 1)
input_shape = (1, 6, 6)
dense_neurons = 2
epochs = load_input("Počet epoch [default 100]: ", int, 100)
learning_rate = load_input("Learning rate [default 0.1]: ", float, 0.1)
verbose = load_input("Výpis erroru po epoche [default True]: ", lambda v: v.lower == "true", True)
print_calculation = load_input("Výpis výpočtov [default True]: ", lambda v: v.lower == "true", True)

print()
print("Nastavenie:")
print()
print(f"Kernel depth: {kernel_depth}")
print(f"Posun: {cnn_stride}")
print(f"Padding: {cnn_zero_padding}")
print(f"Počet kerelov: {cnn_filter_size}")
print(f"MaxPool(1)/AvgPool(0): {max_pool}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {learning_rate}")
print(f"Výpis erroru po epoche: {verbose}")
print(f"Výpis výpočtou: {print_calculation}")
print()


input_depth, input_height, input_width = input_shape

cnn_output_height = int((input_height - kernel_depth + 2 * cnn_zero_padding) / cnn_stride + 1)
cnn_output_widht = int((input_width - kernel_depth + 2 * cnn_zero_padding) / cnn_stride + 1)

# neural network
network = [
    Conv(input_shape, cnn_filter_size, (kernel_depth, kernel_depth), cnn_stride, cnn_zero_padding),
    Sigmoid(),
    MaxPool((cnn_filter_size, cnn_output_height, cnn_output_widht), 2) if max_pool == 1 else AvgPool((cnn_filter_size, cnn_output_height, cnn_output_widht), 2),
    Reshape((cnn_filter_size, int(cnn_output_height / 2), int(cnn_output_widht / 2)), (cnn_filter_size * int(cnn_output_height / 2) * int(cnn_output_widht / 2), 1)),
    Dense(cnn_filter_size * int(cnn_output_height / 2) * int(cnn_output_widht / 2), dense_neurons),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=epochs,
    learning_rate=learning_rate,
    print_calculations=print_calculation,
    verbose=verbose
)

print("\nVýsledná predikcia na trénovacom dáte:")
print(predict(network, x_train[0], False))
print()
