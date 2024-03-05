import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import os

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.random.rand(num_inputs)
        self.learning_rate = learning_rate

    def train(self, X, y, max_epochs=100):
        epoch = 0
        while epoch < max_epochs:
            error_count = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    error_count += 1
            if error_count == 0:
                print("Entrenamiento completado en la época", epoch)
                break
            epoch += 1
        print("Entrenamiento finalizado")

    def predict(self, inputs):
        activation = np.dot(inputs, self.weights)
        return 1 if activation >= 0 else 0

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    X = []
    y = []
    for line in lines:
        data = line.strip().split(',')
        X.append([float(x) for x in data[:-1]])
        y.append(int(data[-1]))
    return np.array(X), np.array(y)

def train_test_split_random(X, y, train_percentage):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage / 100, random_state=42)
    return X_train, X_test, y_train, y_test

def train_test_split_stratified(X, y, train_percentage):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_percentage / 100, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def k_fold_cross_validation(X, y, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    return kf.split(X, y)

def stratified_k_fold_cross_validation(X, y, k):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    return skf.split(X, y)

def random_sampling(X, y, train_percentage):
    num_samples = int(len(X) * (train_percentage / 100))
    indices = np.random.choice(len(X), num_samples, replace=False)
    X_train, X_test = X[indices], np.delete(X, indices, axis=0)
    y_train, y_test = y[indices], np.delete(y, indices, axis=0)
    return X_train, X_test, y_train, y_test

def plot_decision_boundary(X, y, perceptron, technique_name):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.xlabel('X1', fontsize=10)
    plt.ylabel('X2', fontsize=10)
    plt.title(technique_name, fontsize=12)
    x_values = np.linspace(-1.5, 1.5, 100)
    y_values = -(perceptron.weights[0] * x_values) / perceptron.weights[1]
    plt.plot(x_values, y_values, label='Recta de separación')
    plt.legend()
    plt.show()

def main():
    datasets = ["spheres2d10.csv", "spheres2d50.csv", 
                "spheres2d70.csv"]
    techniques = {
        "spheres2d10.csv": ["train_test_split_random", "random_sampling"],
        "spheres2d50.csv": ["train_test_split_stratified", "stratified_k_fold_cross_validation"],
        "spheres2d70.csv": ["k_fold_cross_validation"]
    }

    dataset_results = {}  
    overall_results = {}

    for dataset in datasets:
        dataset_name = os.path.basename(dataset)
        print(f"Dataset: {dataset_name}")

        while True:
            train_percentage_str = input("Ingrese el porcentaje de datos destinados para entrenamiento (0-100): ")
            try:
                train_percentage = float(train_percentage_str)
                if 0 <= train_percentage <= 100:
                    if train_percentage == 100:
                        print("Error: El porcentaje de entrenamiento es del 100%, no hay espacio para el conjunto de prueba.")
                    else:
                        break
                else:
                    print("El porcentaje debe estar en el rango de 0 a 100")
            except ValueError:
                print("Por favor, ingrese un valor numérico válido")

        X, y = read_data(dataset)

        dataset_results[dataset_name] = {}
        overall_results[dataset_name] = {}

        for technique in techniques[dataset_name]:
            split_method = globals()[technique]
            print(f"Técnica: {technique}")

            if "cross_validation" in technique:
                k = 5
                splits = split_method(X, y, k)
                accuracies = []

                for fold, (train_indices, test_indices) in enumerate(splits):
                    X_train, X_test = X[train_indices], X[test_indices]
                    y_train, y_test = y[train_indices], y[test_indices]

                    perceptron = Perceptron(num_inputs=len(X_train[0]))
                    perceptron.train(X_train, y_train)

                    correct_predictions = sum(1 for i in range(len(X_test)) if perceptron.predict(X_test[i]) == y_test[i])
                    accuracy = correct_predictions / len(X_test)
                    accuracies.append(accuracy)

                    plot_decision_boundary(X_train, y_train, perceptron, technique.replace("_", " ").title())

                dataset_results[dataset_name][technique] = np.mean(accuracies)

                print(f"Precisión: {np.mean(accuracies)}")
                print()

            else:
                X_train, X_test, y_train, y_test = split_method(X, y, train_percentage)
                perceptron = Perceptron(num_inputs=len(X_train[0]))
                perceptron.train(X_train, y_train)
                correct_predictions = sum(1 for i in range(len(X_test)) if perceptron.predict(X_test[i]) == y_test[i])
                accuracy = correct_predictions / len(X_test)
                dataset_results[dataset_name][technique] = accuracy

                print(f"Precisión: {accuracy}")
                print()
                plot_decision_boundary(X_train, y_train, perceptron, technique.replace("_", " ").title())

            overall_results[dataset_name] = dataset_results[dataset_name]

    for dataset, results in dataset_results.items():
        plt.figure()
        labels = [technique.replace("_", " ").title() for technique in results.keys()]
        plt.bar(labels, results.values(), color=['blue', 'orange', 'green', 'red'])
        plt.title(f'Rendimiento de Técnicas por Dataset: {dataset}')
        plt.xlabel('Técnicas')
        plt.ylabel('Precisión')
        plt.ylim(0, 1)
        plt.xticks(fontsize=8)
        plt.show()

    plt.figure()
    for dataset, results in overall_results.items():
        plt.bar(results.keys(), results.values(), alpha=0.5, label=dataset)
    plt.title('Rendimiento de todas las Técnicas')
    plt.xlabel('Técnicas')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)
    plt.xticks(rotation=0, fontsize=8)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
