import math
import matplotlib.pyplot as plt
import sys

sys.setrecursionlimit(100000)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
        else:
            return True

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def plot_sigmoid(xs):
    w1, w2, w3, b1, b2, b3 = xs
    x_values = []
    y_values = []
    for i in range(-100, 101):
        x = i / 10.0
        y = sigmoid(w3 * sigmoid(w2 * sigmoid(w1 * x + b1) + b2) + b3)
        x_values.append(x)
        y_values.append(y)
    plt.plot(x_values, y_values)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Função Sigmoide')
    plt.grid(True)
    plt.show()

def plot_gradient(xs):
    x = [i / 10 for i in range(-100, 101)]
    y = [neural([(xi, 0)], xs)[0] for xi in x]

    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('gradiente')
    plt.title('Gradiente')
    plt.grid(True)
    plt.show()

def neural(ts, xs):
    w1 = xs[0]
    w2 = xs[1]
    b1 = xs[2]
    b2 = xs[3]
    w3 = xs[4]
    b3 = xs[5]

    def f(xi):
        return sigmoid(w3 * sigmoid(w2 * sigmoid(w1 * xi + b1) + b2) + b3)

    def f_prime(xi):
        return 1 if is_prime(round(xi * 1000)) else 0

    def f_parity(xi):
        return 1 if xi % 2 == 0 else 0

    dedw1 = sum([
        -(yi - f(xi)) * f(xi) * (1 - f(xi)) * w3 * w2 * sigmoid(w1 * xi + b1) * (1 - sigmoid(w1 * xi + b1)) * xi
        for xi, yi in ts
    ])

    dedw2 = sum([
        -(yi - f(xi)) * f(xi) * (1 - f(xi)) * w3 * sigmoid(w2 * sigmoid(w1 * xi + b1) + b2) * (1 - sigmoid(w2 * sigmoid(w1 * xi + b1) + b2)) * sigmoid(w1 * xi + b1)
        for xi, yi in ts
    ])

    dedw3 = sum([
        -(yi - f(xi)) * f(xi) * (1 - f(xi)) * sigmoid(w2 * sigmoid(w1 * xi + b1) + b2)
        for xi, yi in ts
    ])

    dedb1 = sum([
        -(yi - f(xi)) * f(xi) * (1 - f(xi)) * w3 * w2 * sigmoid(w1 * xi + b1) * (1 - sigmoid(w1 * xi + b1))
        for xi, yi in ts
    ])

    dedb2 = sum([
        -(yi - f(xi)) * f(xi) * (1 - f(xi)) * w3 * sigmoid(w2 * sigmoid(w1 * xi + b1) + b2) * (1 - sigmoid(w2 * sigmoid(w1 * xi + b1) + b2))
        for xi, yi in ts
    ])

    dedb3 = sum([
        -(f_parity(xi) - f(xi)) * f(xi) * (1 - f(xi))
        for xi, _ in ts
    ])

    return [dedw1, dedw2, dedw3, dedb1, dedb2, dedb3]

def descentV(grad, lr, i, err, xts):
    tol = 10 ** -6
    if err < tol:
        return xts, i, err
    else:
        dfdxs = grad(xts)
        xsnovo = [xt - lr * grad for xt, grad in zip(xts, dfdxs)]
        errnovo = sum([(xnovo - xt)**2 for xnovo, xt in zip(xsnovo, xts)])
        return descentV(grad, lr, i + 1, errnovo, xsnovo)

def predict(xi, xs):
    w1 = xs[0]
    w2 = xs[1]
    b1 = xs[2]
    b2 = xs[3]
    w3 = xs[4]
    b3 = xs[5]

    def f(xi):
        return sigmoid(w3 * sigmoid(w2 * sigmoid(w1 * xi + b1) + b2) + b3)

    return f(xi)

def numbersClassification(num, xs):
    score = predict(num / 1000, xs)
    is_primo = is_prime(num)
    is_par = False if num < 0 else num % 2 == 0
    print(f"Número {num} é primo:", is_primo)
    print(f"Número {num} é par:", is_par)

def main():
    w1 = float(input("Digite o valor de w1: "))
    w2 = float(input("Digite o valor de w2: "))
    w3 = float(input("Digite o valor de w3: "))
    b1 = float(input("Digite o valor de b1: "))
    b2 = float(input("Digite o valor de b2: "))
    b3 = float(input("Digite o valor de b3: "))
    lr = float(input("Digite o valor de lr (taxa de aprendizado): "))
    err = float(input("Digite o valor de err: "))
    xs, it, error = descentV(lambda xs: neural([(xi, is_prime(xi * 1000)) for xi in range(1000)], xs), lr, 0, err, [w1, w2, b1, b2, w3, b3])
    print("Descent")
    print(xs)
    print("Passos:", it)
    print("Último erro quadrático: ", error)
    numeros = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 55, 988, 2300, -120, -2900, 10501, 41, 76]


    # p = descentV(lambda xs: neural([(n / 1000, is_prime(n)) for n in range(1, 1001)], xs), lr, 0, err, [w1, w2, b1, b2, w3, b3])

    plot_gradient(xs)
    plot_sigmoid([0.5, 0.2, 0.3, -0.1, 0.1, -0.2])
    plot_sigmoid(xs)

    print("Exemplos:")
    for num in numeros:
        numbersClassification(num, xs)

    while True:
        x = int(input("\nDigite um número para classificar ou -1 para sair: "))
        if x == -1:
            break
        numbersClassification(x, xs)

if __name__ == '__main__':
    main()


# Digite o valor de w1: 0.05
# Digite o valor de w2: 0.07
# Digite o valor de w3: 0.03
# Digite o valor de b1: -0.02
# Digite o valor de b2: 0.01
# Digite o valor de b3: -0.03
# Digite o valor de lr (taxa de aprendizado): 0.1
# Digite o valor de err: 0.001

# Digite o valor de w1: 0
# Digite o valor de w2: 0
# Digite o valor de w3: 0
# Digite o valor de b1: 0
# Digite o valor de b2: 0
# Digite o valor de b3: 0
# Digite o valor de lr (taxa de aprendizado): 1.0
# Digite o valor de err: 10