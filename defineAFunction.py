import numpy as np
import random
import matplotlib.pyplot as plt

def sum_2_numbers(a, b):
    return a + b


def add_matrix(X, Y):
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]


def print_matrix(m):
    for e in m:
        print(e)


def print_index_num_array(arr, n):
    row = None
    column = None
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == n:
                row = i
                column = j
                break
        if row is not None:
            break

    print("Row index of value:", row)
    print("Column index of value:", column)


def get_column_array(arr, n):
    for i in range(len(arr)):
        print(arr[i][n])



if __name__ == "__main__":
    # print(sum_2_numbers(2, 5))

    M = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    v = np.array([1, 2, 3])
    #
    # print("Matrix M:")
    # print(M)
    # print("Rank of M:", np.linalg.matrix_rank(M))
    # print("Shape of M:", M.shape)
    #
    # print("\nVector v:")
    # print(v)
    # print("Rank of v:", np.linalg.matrix_rank(v))
    # print("Shape of v:", v.shape)

    # N = np.array([[2, 4, 6],
    #               [8, 10, 12],
    #               [14, 16, 18]])
    # L = np.array(M) + 2
    # print_matrix(add_matrix(M, N))
    # print_matrix(L)

    # 4
    # print("Transpose of Matrix M:")
    # print(M.T)
    # print("Transpose of Vector v:")
    # print(np.transpose(v))

    # 5
    x = np.array([2, 7])

    # Compute the norm of x
    # norm_x = np.linalg.norm(x)

    # Normalize the vector x
    # normalized_x = x / norm_x
    #
    # print("Norm of vector x:", norm_x)
    # print("Normalized vector x:", normalized_x)

    # 6
    a = np.array([10, 15])
    b = np.array([8, 2])
    c = np.array([1, 2, 3])

    # x1 = a + b
    # x2 = a - b
    # try:
    #     x3 = a - c
    # except ValueError as e:
    #     x3 = str(e)
    # print(x1, '\n', x2, '\n', x3)

    # 7
    # x4 = np.dot(a, b)
    # print(x4)

    # 8
    A = np.array([[2, 4, 9], [3, 6, 7]])
    # print("Rank of A:", np.linalg.matrix_rank(A))

    # print_index_num_array(A, 4)

    # row_index = None
    # column_index = None
    # for i in range(len(A)):
    #     for j in range(len(A[0])):
    #         if A[i][j] == 4:
    #             row_index = i
    #             column_index = j
    #             break
    #     if row_index is not None:
    #         break
    #
    # print("Row index of value:", row_index)
    # print("Column index of value:", column_index)

    # get_column_array(A, 1)

    # 9
    # np.random.seed(0) #Init matrix 1 time
    #
    # B = np.random.randint(-10, 10, size=(3, 3))
    # print(B)

    # 10
    # C = np.eye(4)

    # Print the identity matrix
    # print("3x3 Identity Matrix:")
    # print(C)

    # 11
    # D = np.random.randint(1, 10, size=(3, 3))
    # print(D)
    # way1
    # print("Trace of the random matrix:", np.trace(D))
    # way2
    # trace = 0
    # for i in range(len(D)):
    #     trace += D[i][i]
    # print("Trace of the random matrix:", trace)

    # 12
    # diagonal_value = [1, 2, 3]
    # M = np.diag(diagonal_value)
    # print("Matrix of 3x3 with diagonal:\n", M)

    # 13
    # A = np.array([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
    # print(np.linalg.det(A))

    # 14
    # a1 = np.array([1, -2, -5])
    # a2 = np.array([2, 5, 6])
    # a3 = np.column_stack((a1, a2))
    # print(a3)

    # 15
    # y_values = range(-5, 6)
    #
    # # Compute the square of each y value
    # y_squared = [y ** 2 for y in y_values]
    #
    # # Plot the square of y
    # plt.plot(y_values, y_squared)
    # plt.title('Square of y')
    # plt.xlabel('y')
    # plt.ylabel('y^2')
    # plt.grid(True)
    # plt.show()

    # 16
    # evenly_spaced_values = np.linspace(0, 32, 4)
    # print("Evenly spaced values:", evenly_spaced_values)

    # 17
    # # Generate 50 evenly spaced values for x from -5 to 5
    # x_values = np.linspace(-5, 5, 50)
    #
    # # Calculate y = x^2
    # y_values = x_values ** 2
    #
    # # Plot the results
    # plt.plot(x_values, y_values)
    # plt.title('Plot of y = x^2')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.grid(True)
    # plt.show()

    # 18
    # # Generate x values
    # x_values = np.linspace(-5, 5, 100)  # Generating 100 evenly spaced points from -5 to 5
    #
    # # Calculate y = exp(x)
    # y_values = np.exp(x_values)
    #
    # # Plot y = exp(x)
    # plt.plot(x_values, y_values, label='$y = e^x$')  # Using LaTeX syntax for the label
    # plt.xlabel('x')  # Label for the x-axis
    # plt.ylabel('y')  # Label for the y-axis
    # plt.title('Plot of $y = e^x$')  # Title of the plot
    # plt.legend()  # Show legend
    # plt.grid(True)  # Show grid
    # plt.show()

    # 19
    # # Generate x values (excluding 0)
    # x_values = np.linspace(0.001, 5, 100)  # Generating 100 evenly spaced points from 0.001 to 5
    #
    # # Calculate y = log(x)
    # y_values = np.log(x_values)
    #
    # # Plot y = log(x)
    # plt.plot(x_values, y_values, label='$y = \log(x)$')  # Using LaTeX syntax for the label
    # plt.xlabel('x')  # Label for the x-axis
    # plt.ylabel('y')  # Label for the y-axis
    # plt.title('Plot of $y = \log(x)$')  # Title of the plot
    # plt.legend()  # Show legend
    # plt.grid(True)  # Show grid
    # plt.show()

    # 20
    # # Generate x values
    # x_values = np.linspace(0.001, 5, 100)  # Generating 100 evenly spaced points from 0.001 to 5
    #
    # # Calculate y values for exp(x) and exp(2*x)
    # y_exp_x = np.exp(x_values)
    # y_exp_2x = np.exp(2 * x_values)
    #
    # # Calculate y values for log(x) and log(2*x)
    # y_log_x = np.log(x_values)
    # y_log_2x = np.log(2 * x_values)
    #
    # # Plotting
    # plt.figure(figsize=(12, 6))
    #
    # # Subplot 1: exp(x) and exp(2*x)
    # plt.subplot(1, 2, 1)
    # plt.plot(x_values, y_exp_x, label='$y = e^x$')
    # plt.plot(x_values, y_exp_2x, label='$y = e^{2x}$')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Exponential Functions')
    # plt.legend()
    # plt.grid(True)
    #
    # # Subplot 2: log(x) and log(2*x)
    # plt.subplot(1, 2, 2)
    # plt.plot(x_values, y_log_x, label='$y = \log(x)$')
    # plt.plot(x_values, y_log_2x, label='$y = \log(2x)$')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Logarithmic Functions')
    # plt.legend()
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()