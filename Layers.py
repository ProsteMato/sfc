import numpy as np
from utils import (next_step, print_operation, print_with_line, print_matrix)

class Conv:
    def __init__(self, input_shape, kernel_size, kernel_shape = (3, 3), stride = 1, padding = 0) -> None:
        input_depth, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.kernel = kernel_shape
        self.kernel_shape = (kernel_size, input_depth, *kernel_shape)
        self.output_shape = (kernel_size, int((input_height - kernel_shape[0] + 2 * padding) / stride + 1), int((input_width - kernel_shape[0] + 2 * padding) / stride + 1))
        self.biases = np.random.rand(*self.output_shape)
        self.kernels = np.random.rand(*self.kernel_shape)
        self.stride = stride
        self.zero_padding = padding
        
    
    def forward(self, input, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'Conv: Forward stride: {self.stride} padding: {self.zero_padding} kernel_size {self.kernel_size} kernel: {self.kernel}', verbose)
        print_with_line(f'Skopírovanie biasov na výstup:', verbose)
        self.input = input
        self.output = np.copy(self.biases)
        
        for i in range(self.kernel_size):
            for j in range(self.input_depth):
                print_with_line(f'Bias pre {i+1}. filter:', verbose)
                print_matrix(self.output[i], verbose)
                next_step(verbose)
                if (self.zero_padding > 0):
                    print_with_line(f"Pridanie zero padding vstupnej matici", verbose)
                padded_input = np.pad(input[j], self.zero_padding)
                if (self.zero_padding > 0):
                    print_matrix(padded_input, verbose)
                print_with_line(f'Výpočet konvolúcii na {j+1}. matici:', verbose)
                print_matrix(padded_input, verbose)
                print_with_line(f'S {i+1}. filtrom:', verbose)
                print_matrix(self.kernels[i, j], verbose)
                next_step(verbose)
                windowed_input = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape = self.kernel)[::self.stride, ::self.stride]
                for index_row, windowed_steps in enumerate(windowed_input):
                    for index_col, window in enumerate(windowed_steps):
                        print_with_line(f"Prenásobenie okna filtrom:", verbose)
                        print_operation(window, "*", self.kernels[i, j], None, verbose)
                        next_step(verbose)
                        multiply = np.multiply(window, self.kernels[i, j])
                        print_with_line(f"Vypočítanie a pričítanie konvolúcie k pozícii [{i}][{index_row}][{index_col}]:", verbose)
                        suma = np.sum(multiply)
                        print_operation(multiply, "suma je:", np.array([[suma]]), None, verbose)
                        self.output[i][index_row][index_col] += suma
                        next_step(verbose)
            print_with_line(f"Výsledná matica pre {i+1}. filter je:", verbose)
            print_matrix(self.output[i], verbose)
            next_step(verbose)
                    
                        
        if verbose: print(f'================================================')
        next_step(verbose)
        
        return self.output
    
    def backward(self, output_grads, learning_rate, verbose) -> None:
        if verbose: print(f'================================================')
        print_with_line(f'Conv: Backpropagation stride: {self.stride} padding: {self.zero_padding} kernel_size {self.kernel_size} kernel: {self.kernel}', verbose)
        kernel_grads = np.zeros(self.kernels.shape)
        input_grads = np.zeros(self.input.shape)
        
        for i in range(self.kernel_size):
            padded_output_grands = np.pad(output_grads[i], self.kernel[0] - 1)
            windowed_output_grads = np.lib.stride_tricks.sliding_window_view(padded_output_grands, window_shape = self.kernel)[::self.stride, ::self.stride]
            print_with_line(f'Gradient', verbose)
            print_matrix(output_grads[i], verbose)
            next_step(verbose)
            print_with_line(f'Ohraničený gradient nulami pre jednoduchšie počítanie výstupných gradientov', verbose)
            print_matrix(padded_output_grands, verbose)
            next_step(verbose)
            for j in range(self.input_depth):
                padded_input = np.pad(self.input[j], self.zero_padding)
                windowed_input = np.lib.stride_tricks.sliding_window_view(padded_input, window_shape = output_grads[i].shape)[::self.stride + int((self.input_width - output_grads[i].shape[0] + 2 * self.zero_padding) / (self.kernel[0] - 1) - 1), ::self.stride + int((self.input_height - output_grads[i].shape[1] + 2 * self.zero_padding) / (self.kernel[1] - 1) - 1)]
                print_with_line(f'Výpočet gradientov pre {i+1}. filter, gradient ako filter:', verbose)
                print_matrix(output_grads[i], verbose)
                next_step(verbose)
                print_with_line(f'Bude prechádzať maticu:', verbose)
                print_matrix(padded_input, verbose)
                next_step(verbose)
                
                for index_row, windowed_steps in enumerate(windowed_input):
                    for index_col, window in enumerate(windowed_steps):
                        print_with_line(f'Výpočet gradientu pre {i+1} filter na pozícii [{i}, {j}][{index_row}][{index_col}]', verbose)
                        print_operation(window, "*", output_grads[i], None, verbose)
                        next_step(verbose)
                        print_with_line(f'Výsledok násobenia:', verbose)
                        multiply = np.multiply(window, output_grads[i])
                        print_matrix(multiply, verbose)
                        next_step(verbose)
                        print_with_line(f'Sčítanie:', verbose)
                        sum = np.sum(multiply)
                        print_operation(multiply, "suma je:", np.array([[sum]]), None, verbose)
                        next_step(verbose)
                        kernel_grads[i, j][index_row][index_col] = sum
                        
                print_with_line(f'Výsledný gradient pre {i + 1} filter', verbose)
                print_matrix(kernel_grads[i, j], verbose)
                next_step(verbose)
                
                print_with_line(f'Výpočet výstupných gradientov', verbose)
                for index_row, windowed_steps in enumerate(windowed_output_grads):
                    for index_col, window in enumerate(windowed_steps):
                        print_with_line(f'Výpočet výstupného gradientu na pozíciu [{j}][{index_row}][{index_col}]', verbose)
                        print_with_line(f'Násobanie okna filtrom:', verbose)
                        print_operation(window, "*", self.kernels[i, j], None, verbose)
                        next_step(verbose)
                        multiply = np.multiply(window, self.kernels[i, j])
                        print_with_line(f'Suma a pričítanie na pozíciu [{j}][{index_row}][{index_col}]', verbose)
                        sum = np.sum(multiply)
                        print_operation(multiply, "suma je:", np.array([[sum]]), None, verbose)
                        next_step(verbose)
                        input_grads[j][index_row][index_col] += sum
                print_with_line(f'Výstupné gradienty pre {j} dimenziu', verbose)
                print_matrix(input_grads[j], verbose)
                next_step(verbose)
        
        for j in range(self.input_depth):
            input_grads[j] = np.rot90(input_grads[j], 2)
        
        print_with_line(f'Výpočet nových filtrov.', verbose)
        for i in range(self.kernel_size):
            for j in range(self.input_depth):
                print_with_line(f'Výpočet {i+1}. filtru', verbose)
                print_operation(kernel_grads[i, j], "*", None, learning_rate, verbose)
                print_with_line(f'Výsledok výpočtu {i+1}. filtru', verbose)
                self.kernels[i, j] -= learning_rate * kernel_grads[i, j]
                print_matrix(self.kernels[i, j], verbose)
                next_step(verbose)
        
        print_with_line(f'Výpočet nových biasov.', verbose)
        for i in range(self.kernel_size):
            print_with_line(f'Výpočet {i+1}. biasu', verbose)
            print_operation(output_grads[i], "*", None, learning_rate, verbose)
            print_with_line(f'Výsledok výpočtu {i+1}. biasu', verbose)
            self.biases[i] -= learning_rate * output_grads[i]
            print_matrix(self.biases[i], verbose)
            next_step(verbose)
        
        if verbose: print(f'================================================')
        next_step(verbose)
        
        return input_grads
    
    
class MaxPool():
    def __init__(self, input_shape, depth):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_shape = (input_depth, int(input_height / depth), int(input_width / depth))
        self.kernel = (depth, depth)
        self.indices = np.zeros(self.output_shape)
    
    def forward(self, input, verbose):
        self.input = input
        if verbose: print(f'================================================')
        print_with_line(f'MaxPool: Forward (2, 2)', verbose)
        
        help_indices = np.arange(0, self.input_height * self.input_width).reshape((self.input_height, self.input_width))
        output = np.zeros(self.output_shape)
        
        for i in range(self.input_depth):
            print_with_line(f'Počítanie maxima pre jednotlivé okna vstupnej matice:', verbose)
            print_matrix(input[i], verbose)
            next_step(verbose)
            windowed_input = np.lib.stride_tricks.sliding_window_view(input[i], window_shape = self.kernel)[::2, ::2]
            windowed_indices = np.lib.stride_tricks.sliding_window_view(help_indices, window_shape = self.kernel)[::2, ::2]
            for index_row, (w_s, w_i_s) in enumerate(zip(windowed_input, windowed_indices)):
                for index_col, (window, idices_step) in enumerate(zip(w_s, w_i_s)):
                    print_with_line(f'Počítanie maxima pre pozíciu [{i}][{index_row}][{index_col}]:', verbose)
                    maximum = window.max()
                    print_operation(window, "maximum je: ", np.array([[maximum]]), None, verbose)
                    output[i][index_row][index_col] = maximum
                    self.indices[i][index_row][index_col] = idices_step.reshape(-1)[window.argmax()]
                    next_step(verbose)
            print_with_line(f'Výstupná matica maxím:', verbose)
            print_matrix(output[i], verbose)
            next_step(verbose)
                    
        if verbose: print(f'================================================')
        next_step(verbose)
        return output
            
        
    
    def backward(self, output_grads, learning_rate, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'MaxPool: Backpropagation (2, 2)', verbose)
        print_with_line(f'Spropagovanie gradientov na pozície maxím vstupného vektoru', verbose)
        input_grads = np.zeros(self.input_shape).reshape(self.input_depth, -1)
        indices = self.indices.reshape(self.input_depth, -1).astype(int)
        grads = output_grads.reshape(self.input_depth, -1)
        for i in range(self.input_depth):
            print_with_line(f'Gradient pre {i+1}. maticu: ', verbose)
            print_matrix(output_grads[i], verbose)
            print_with_line(f'Pozície maxím pre {i+1}. maticu:', verbose)
            print_matrix(self.indices[i], verbose)
            print_with_line(f'Propagácia pre {i+1}. maticu:', verbose)
            for j, k in enumerate(indices[i]):
                input_grads[i][k] += grads[i][j]
            print_matrix(input_grads[i].reshape((self.input_shape[1], self.input_shape[2])), verbose)
        input_grads = input_grads.reshape(self.input_shape)
        if verbose: print(f'================================================')
        next_step(verbose)
        return input_grads
            


class AvgPool():
    def __init__(self, input_shape, depth):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.input_height = input_height
        self.input_width = input_width
        self.output_shape = (input_depth, int(input_height / depth), int(input_width / depth))
        self.kernel = (depth, depth)
    
    def forward(self, input, verbose):
        output = np.zeros(self.output_shape)
        
        if verbose: print(f'================================================')
        print_with_line(f'Forward: AvgPool (2, 2)', verbose)
        
        for i in range(self.input_depth):
            print_with_line(f'Počítanie priemeru pre jednotlivé okná vstupnej matice:', verbose)
            print_matrix(input[i], verbose)
            next_step(verbose)
            windowed_input = np.lib.stride_tricks.sliding_window_view(input[i], window_shape = self.kernel)[::2, ::2]
            for index_row, w_s in enumerate(windowed_input):
                for index_col, window in enumerate(w_s):
                    print_with_line(f'Výpočet primeru pre pozíciu [{i}][{index_row}][{index_col}]:', verbose)
                    average = np.average(window)
                    print_operation(window, "priemer je:", np.array([[average]]), None, verbose)
                    output[i][index_row][index_col] = average
                    next_step(verbose)
            print_with_line(f'Výstupná matica priemerov:', verbose)
            print_matrix(output[i], verbose)
            next_step(verbose)
        if verbose: print(f'================================================')
        next_step(verbose)
        return output
    
    def backward(self, output_grads, learning_rate, verbose):
        input_grads = np.ones(self.input_shape)
        
        if verbose: print(f'================================================')
        print_with_line(f'Backpropagation: AvgPool (2, 2)', verbose)
        print_with_line(f'Priemerné rozdelenie gradientu medzi spriemerované prvky okien:', verbose)
        
        for i in range(self.input_depth):
            print_with_line(f'{i+1}. matica gradientov:', verbose)
            print_matrix(output_grads[i], verbose)
            next_step(verbose)
            print_with_line(f'Expanzia hodnôt gradientu na veľkosť filtra:', verbose)
            expanded_output_grads = np.kron(output_grads[i], np.ones(self.kernel))
            print_matrix(expanded_output_grads, verbose)
            next_step(verbose)
            print_with_line(f'Pridanie hodnôť gradientu do výslednej matice:', verbose)
            input_grads[i][0:expanded_output_grads.shape[0], 0:expanded_output_grads.shape[1]] = expanded_output_grads
            print_matrix(input_grads[i], verbose)
            next_step(verbose)
            print_with_line(f'Spriemerovanie matice podľa velkosti filtra:', verbose)
            input_grads[i] = input_grads[i] * (1 / (self.output_shape[1] * self.output_shape[2]))
            print_matrix(input_grads[i], verbose)
            next_step(verbose)
                    
        if verbose: print(f'================================================')
        next_step(verbose)
        return input_grads
                

class Sigmoid():
    def __sigmoid(self, input):
        return 1 / (1 + np.exp(np.negative(input)))
    
    def forward(self, input, verbose):
        self.input = input
        if verbose: print(f'================================================')
        print_with_line(f'Forward: sigmoid', verbose)
        sigmoid_input = self.__sigmoid(input)
        for i in range(input.shape[0]):
            print_with_line(f'Použitie funkcie sigmoidy na {i+1}. dimenziu vstupnej matice:', verbose)
            print_matrix(input[i], verbose)
            next_step(verbose)
            print_with_line(f'Výsledok', verbose)
            print_matrix(sigmoid_input[i], verbose)
        if verbose: print(f'================================================')
        next_step(verbose)
        return sigmoid_input
    
    def backward(self, output_grads, learning_rate, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'Backward: sigmoid', verbose)
        s = self.__sigmoid(self.input)
        input_grads = np.multiply(output_grads, (s * (1 - s)))
        for i in range(self.input.shape[0]):
            print_with_line(f'Použitie funkcie sigmoidy na {i+1}. dimenziu vstupnej matice:', verbose)
            print_matrix(self.input[i], verbose)
            next_step(verbose)
            print_with_line(f'Výsledok', verbose)
            print_matrix(s[i], verbose)
            print_with_line(f'Vynásovanie matice gradientov s deriváciou sigmoidy:', verbose)
            print_operation(output_grads[i], "*", (s * (1 - s))[i], None, verbose)
            next_step(verbose)
            print_with_line(f'Výsledok', verbose)
            print_matrix(input_grads[i], verbose)
        if verbose: print(f'================================================')
        next_step(verbose)
        return input_grads
        
class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
    
    def forward(self, input, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'Forward: reshape {self.input_shape}, {self.output_shape}', verbose)
        reshaped_input = input.reshape(self.output_shape)
        for i in range(self.input_shape[0]):
            print_with_line(f'Zmena tvaru matice pre filter {i+1}:', verbose)
            next_step(verbose)
            print_matrix(input[i], verbose)
            print_with_line(f'Výsledok', verbose)
            print_matrix(reshaped_input[i], verbose)
        if verbose: print(f'================================================')
        next_step(verbose)
        return reshaped_input
    
    def backward(self, output_grad, learning_rate, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'Backward: reshape {self.input_shape}, {self.output_shape}', verbose)
        print_with_line(f'Zmena tvaru matice', verbose)
        print_matrix(output_grad, verbose)
        reshaped_input = output_grad.reshape(self.input_shape)
        next_step(verbose)
        for i in range(self.input_shape[0]):
            print_with_line(f'Výsledok pre dimenziu {i+1}', verbose)
            print_matrix(reshaped_input[i], verbose)
        if verbose: print(f'================================================')
        next_step(verbose)
        return reshaped_input
    
class Dense():
    def __init__(self, input_shape, output_shape):
        self.weights = np.random.randn(output_shape, input_shape)
        self.biases = np.random.randn(output_shape, 1)
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    
    def forward(self, input, verbose):
        self.input = input
        if verbose: print(f'================================================')
        print_with_line(f'Forward: Dense z {self.input_shape} do {self.output_shape}', verbose)
        next_step(verbose)
        print_with_line(f'Násobenie matice váh s maticou vstupu a pričítanie biasu', verbose)
        print_operation(self.weights, "*", input, None, verbose)
        weight_input = np.dot(self.weights, input)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        print_matrix(weight_input, verbose)
        next_step(verbose)
        print_with_line(f'Pričítanie biasu (ku každému prvku z matice váh sa pričíta každý prvok z matice biasov):', verbose)
        print_operation(weight_input, "+", self.biases, None, verbose)
        weight_input_biases = weight_input + self.biases
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        print_matrix(weight_input_biases, verbose)
        next_step(verbose)
        if verbose: print(f'================================================')
        
        
        return weight_input_biases
    
    def backward(self, output_grads, learning_rate, verbose):
        if verbose: print(f'================================================')
        print_with_line(f'Backpropagation: Dense z {self.output_shape} do {self.input_shape}', verbose)
        print_with_line(f'Násobenie gradientov so vstupom:', verbose)
        next_step(verbose)
        print_operation(output_grads, "*" , self.input.T, None, verbose)
        output_grads_input = np.dot(output_grads, self.input.T)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        print_matrix(output_grads_input, verbose)
        next_step(verbose)
        print_with_line(f'Násobenie vynásobených gradientov so vstupom s learning rate:', verbose)
        print_operation(output_grads_input, "*", None, learning_rate, verbose)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        learning_output_grads_input = learning_rate * output_grads_input
        print_matrix(learning_output_grads_input, verbose)
        next_step(verbose)
        print_with_line(f'Odčítanie stávajúcich váh s prechozím výsledkom:', verbose)
        print_operation(self.weights, "-", learning_output_grads_input, None, verbose)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        self.weights -= learning_output_grads_input
        print_matrix(self.weights, verbose)
        next_step(verbose)
        print_with_line(f'Vynásobenie learning rate s gradientmi:', verbose)
        learning_grads = learning_rate * output_grads
        print_operation(output_grads, "*", None, learning_rate, verbose)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        print_matrix(learning_grads, verbose)
        next_step(verbose)
        print_with_line(f'Odčítanie výsledku od stávajúcich biaseov:', verbose)
        print_operation(self.biases, "-", learning_grads, None, verbose)
        next_step(verbose)
        print_with_line(f'Výsledok:', verbose)
        print_matrix(self.biases, verbose)
        self.biases -= learning_rate * output_grads
        
        return np.dot(self.weights.T, output_grads)
        
        