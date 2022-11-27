import numpy as np
from utils import (train, predict, binary_cross_entropy_prime, binary_cross_entropy, load_input)
from Layers import (Conv, MaxPool, AvgPool, Sigmoid, Reshape, Dense)




kernel_depth = load_input("Velkosť kernelu v konvolučnej vrstve [min 1, default 3]: ", int, 3, lambda v: int(v) > 0)
cnn_stride = load_input("Posun kernelu v konvolučnej vrstve [min 1, default 2]: ", int, 2, lambda v: int(v) > 0)
cnn_zero_padding = load_input("Výplň vstupu v konvolučnej vrstve [min 0, default 0]: ", int, 0, lambda v: int(v) >= 0)
cnn_filter_size = load_input("Počet kernelov v konvolučnej vrstve [min 0, default 1]: ", int, 1, lambda v: int(v) > 0)
max_pool = load_input("max pooling (1) alebo average pooling (0) [default 1]: ", int, 1, lambda v: v == "0" or v == "1")
dense_neurons = 2
epochs = load_input("Počet epoch [min 1, default 100]: ", int, 100, lambda v: int(v) > 0)
learning_rate = load_input("Learning rate [default 0.1]: ", float, 0.1, lambda v: float(v) > 0)
verbose = load_input("Výpis erroru po epoche [default True]: ", lambda v: v.lower == "true", True)
print_calculation = load_input("Výpis výpočtov [default True]: ", lambda v: v.lower == "true", True)

min_input_size = cnn_stride + kernel_depth - 2 * cnn_zero_padding

input_depth = load_input("Veľkosť dimenzie vstupu [min 1, default 1]: ", int, 1, lambda v: int(v) > 0)
input_size = load_input(f"Výška a šírka vstupu [minimum {min_input_size}, default {min_input_size+2}]: ", int, min_input_size + 2, lambda v: v >= min_input_size)

input_shape = (input_depth, input_size, input_size)

x_train = np.zeros((1, input_depth, input_size, input_size))
y_train = np.zeros((1, 2, 1))

print("Nastavenie vstupnej matice: ")
for i in range(input_depth):
    print(f"Nastavenie {i+1}. dimenzie matice: ")
    for j in range(input_size):
        x_train[0, i, j] = np.array([int(x) for x in input(f"Nastavenie {j+1} riadku matice [napr. 1 2 -1 -3]: ").split()])
        
trieda = load_input("Vstup bude patriť do triedy [0/1]: ", int, 0, lambda v: v == "0" or v == "1")

y_train[0, trieda] = 1

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
