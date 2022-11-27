import numpy as np

def print_matrix(arr, verbose):
    if not verbose: return
    
    for i in range(arr.shape[0]):
        if len(arr.shape) > 1:
            for j in range(arr.shape[1]):
                print(" |", "{:>6.2f}".format(arr[i, j]), end="")
        else:
            print(" |", "{:>6.2f}".format(arr[i]), end="")
        print(" |")
        

def print_with_line(text, verbose):
    if verbose:
        print(f'------------------------------------------------')
        print(text)

def print_operation(arr1, operation, arr2, constant = None, verbose = True):
    if not verbose: return
    
    if (arr1 is not None and (arr2 is None or arr1.shape[0] >= arr2.shape[0])):
        center = int(np.floor(arr1.shape[0] / 2))
        if (arr2 is not None):
            padding = int(np.ceil((arr1.shape[0] - arr2.shape[0]) / 2))
        
        for i in range(arr1.shape[0]):
            if constant is not None and i == center:
                print("{:>5}".format(constant), " * ", end="")
            
            if constant is not None and i != center:
                print(" " * 8, end="")
            
            if (len(arr1.shape) > 1):
                    
                for j in range(arr1.shape[1]):
                    print(" |", "{:>5.2f}".format(arr1[i, j]), end="")
            else:
                print(" |", "{:>5.2f}".format(arr1[i]), end="")
            
            if (arr2 is not None):
                if (i == center):
                    print(" |", end="")
                    print(" " * 3, end="")
                    print(operation, end="")
                    
                    if (i >= padding and i - padding < arr2.shape[0]):
                        print(" " * 3, end="")
                        if len(arr2.shape) > 1:
                            for j in range(arr2.shape[1]):
                                print(" |", "{:>5.2f}".format(arr2[i - padding, j]), end="")
                        else:
                            print(" |", "{:>5.2f}".format(arr2[i - padding]), end="")
                else:
                    if (i >= padding and i - padding < arr2.shape[0]):
                        print(" |", end="")
                        print(" " * 7, end="")
                        if len(arr2.shape) > 1:
                            for j in range(arr2.shape[1]):
                                print(" |", "{:>5.2f}".format(arr2[i - padding, j]), end="")
                        else:
                            print(" |", "{:>5.2f}".format(arr2[i - padding]), end="")
            print(" |")
    
    else:
        center = int(np.floor(arr2.shape[0] / 2))
        if (arr1 is not None):
            padding = int(np.ceil((arr2.shape[0] - arr1.shape[0]) / 2))
        
        for i in range(arr2.shape[0]):
            if constant is not None and i == center:
                print("{:>5}".format(constant), " * ", end="")
            
            if constant is not None and i != center:
                print(" " * 9, end="")
            
            if (arr1 is not None):
                if (i >= padding and i - padding < arr1.shape[0]):
                    if len(arr1.shape) > 1:
                        for j in range(arr1.shape[1]):
                            print(" |", "{:>5.2f}".format(arr1[i - padding, j]), end="")
                    else:
                        print(" |", "{:>5.2f}".format(arr1[i - padding]), end="")
                else:
                    for _ in range(arr1.shape[1]):
                        print(" " * 8, end="")
                    print("  ", end="")
                     
            if (i == center):
                if (arr1 is not None): 
                    print(" |", end="")
                    print(" " * 3, end="")
                    print(operation, end="")
                    print(" " * 3, end="")
                else:
                    print(" " * 7, end="")
                if len(arr2.shape) > 1:
                    for j in range(arr2.shape[1]):
                        print(" |", "{:>5.2f}".format(arr2[i, j]), end="")
                else:
                    print(" |", "{:>5.2f}".format(arr2[i]), end="")
            else:
                if (arr1 is not None):
                    if (i >= padding and i - padding < arr1.shape[0]):
                        print(" |", end="")
                print(" " * 7, end="")
                if len(arr2.shape) > 1:
                    for j in range(arr2.shape[1]):
                        print(" |", "{:>5.2f}".format(arr2[i, j]), end="")
                else:
                    print(" |", "{:>5.2f}".format(arr2[i]), end="")
            print(" |")
            
def next_step(verbose):
    if not verbose: return
    input("Ďalší krok [stlačte ľubovolné tlačítko]")
    
def skip_calculation_for_n_epochs():
    return load_input("Vynechať vypisovanie výpočtu pre n epocho [default 0]: ", int, 0)
    
def load_input(text, callback, default, condition = None):
    i = input(text)
    if (i == ""):
        return default
    else:
        if condition is None or condition(i):
            return callback(i)
        else:
            return load_input(f"Nesprávna hodnota '{i}' skúste znova. " + text, callback, default, condition)
    

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def predict(network, input, verbose):
    output = input
    for layer in network:
        output = layer.forward(output, verbose)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, print_calculations = True, verbose = True):
    skip = 0
    first_print_calculations = print_calculations
    for e in range(epochs):
        if first_print_calculations:
            print_calculations = not skip > 0
        skip -= (skip > 0) * 1
        error = 0
        for x, y in zip(x_train, y_train):
            
            if print_calculations: print(f'================================================')
            print_with_line(f"Epocha: {e+1}", print_calculations)
            print_with_line("Vstup do siete:", print_calculations)
            if print_calculations: print(x)
            print_with_line("Očakávaný výstup", print_calculations)
            if print_calculations: print(y)
            if print_calculations: print(f'================================================')
            output = predict(network, x, print_calculations)

            error += loss(y, output)

            grad = loss_prime(y, output)
            for i, layer in enumerate(reversed(network)):
                grad = layer.backward(grad, learning_rate, print_calculations)

        error /= len(x_train)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")
        
        if print_calculations: 
            skip = skip_calculation_for_n_epochs()
                