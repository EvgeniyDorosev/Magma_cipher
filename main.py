import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools



def second_abs_max(arr):
    new_arr = np.sort(np.absolute(arr))
    res = np.array([el for el in new_arr if new_arr[-1] - el > 1e-7])
    return res[-1] if len(res) > 0 else None


def partition(str, block_size, r):
    lst = []
    if r == 0:
        r = 1
    for i in range(r):
        lst.append(str[block_size * i:block_size * i + block_size])
    return lst


def gener_pattern(M):
    lst = []
    temp = ''
    count = 0
    while True:
        temp = str(bin(count))[2:]
        count += 1
        if len(temp) <= M:
            lst.append(temp.zfill(M))
        else:
            break
    return lst


def transition_probability_matrix(x, y, block_size_x):
    lst = gener_pattern(block_size_x)

    # ???????????????????? ниже
    col1 = [el for el in x]
    col2 = [el for el in y]

    for comb in itertools.permutations(lst, 2):
        col1.append(comb[0])
        col2.append(comb[1])

    for el in lst:
        col1.append(el)
        col2.append(el)

    data = {'col1': col1, 'col2': col2, 'amount': 1}
    df = pd.DataFrame(data)
    # print(df.head())
    # print(df.tail(8))

    new_df = pd.DataFrame()
    new_df['amount'] = df.groupby(['col1', 'col2'])['amount'].sum() - 1
    counts = new_df['amount'].to_list()

    res_df = pd.DataFrame(columns=lst, index=lst).fillna(0)
    # print(res_df)

    k = 0
    for i in range(len(lst)):
        for j in range(len(lst)):
            res_df.iloc[i, j] = counts[k]
            k += 1
    return res_df


def eigenvaluies(matr):
    eigenvalues, v = np.linalg.eig(matr)
    print('Eigenvalues:')
    for el in eigenvalues:
        print(el)
    eigenvalues = [abs(el) for el in eigenvalues]
    print('Abs eigenvalues:')
    for el in eigenvalues:
        print(el)
    print('Second max eigenvalue:')
    print(second_abs_max(eigenvalues))


def xor_str(x, k_i, block_size, L):  # L - длина раунд ключа
    # res = ''
    # if block_size > L:
    #     for i in range(block_size - L):
    #         k_i += k_i[i]
    #
    # for i in range(block_size):
    #     res += str((int(x[i]) + int(k_i[i])) % 2)
    #
    # if L > block_size:
    #     new_res = ''  # т к разница максимум в один то пойдёт
    #     for i in range(block_size):
    #         new_res += str((int(res[i]) + int(k_i[-1])) % 2)
    #     return new_res.zfill(block_size)
    # return res.zfill(block_size)
    res = ''
    if len(x) > L:
        for i in range(len(x) - L):
            k_i += k_i[i]

    for i in range(len(x)):
        res += str((int(x[i]) + int(k_i[i])) % 2)

    if L > len(x):
        new_res = ''  # т к разница максимум в один то пойдёт
        for i in range(len(x)):
            new_res += str((int(res[i]) + int(k_i[-1])) % 2)
        return new_res.zfill(len(x))
    return res.zfill(len(x))


def y_xor_transformation(x, k, block_size_x, L, r):
    lst_y = []
    for j in range(len(x)):
        temp = x[j]
        for i in range(r):
            temp = xor_str(temp, k[i], block_size_x, L)
        lst_y.append(temp)
    return lst_y


def round_transform_xor(M, L, n, block_size_x=3):  # длина ключа, длина раундового ключа, длина сообщения
    # раундовое преобразование:
    # для каждого блока делаем xor со всеми раундовыми ключами поочереди

    lst_k = gener_pattern(M)  # всевозможные ключ
    lst_x = gener_pattern(n)  # всевозможные тексты для шифрования
    lst = gener_pattern(block_size_x)
    matr = pd.DataFrame(columns=lst, index=lst).fillna(0)

    r = M // L  # число раундов = число блоков k
    for K in lst_k:
        for X in lst_x:
            lst_k_i = partition(K, L, r)
            lst_x_i = partition(X, block_size_x, len(X) // block_size_x)
            lst_y_i = y_xor_transformation(lst_x_i, lst_k_i, block_size_x, L, r)
            matr += transition_probability_matrix(lst_x_i, lst_y_i, block_size_x)

    matr = matr.div(matr.sum(axis=1), axis=0).fillna(0)
    print('Perm matrix')
    print(matr)
    return matr


def shift_str(x, k_i, size):
    res = ""
    amount = k_i.count('1')
    for i in range(len(x)):
        res += x[(i - amount) % size]
    return res


def y_shift_transformation(x, k, block_size, r):
    lst_y = []
    for j in range(len(x)):
        temp = x[j]
        for i in range(r):
            temp = shift_str(temp, k[i], block_size)
        lst_y.append(temp)
    return lst_y


def round_transform_shift(M, L, n, block_size_x=3):  # длина ключа, длина раундового ключа, длина сообщения
    # раундовое преобразование:
    # для каждого блока делаем сдвиг влево на число единиц в k_i

    lst_k = gener_pattern(M)  # всевозможные ключ
    lst_x = gener_pattern(n)  # всевозможные тексты для шифрования
    lst = gener_pattern(block_size_x)
    matr = pd.DataFrame(columns=lst, index=lst).fillna(0)

    r = M // L  # число раундов = число блоков k

    for K in lst_k:
        for X in lst_x:
            lst_k_i = partition(K, L, r)
            lst_x_i = partition(X, block_size_x, len(X) // block_size_x)
            lst_y_i = y_shift_transformation(lst_x_i, lst_k_i, block_size_x, r)
            matr += transition_probability_matrix(lst_x_i, lst_y_i, block_size_x)

    matr = matr.div(matr.sum(axis=1), axis=0).fillna(0)
    print('Perm matrix')
    print(matr)
    return matr


def round_transform_shift_xor(M, L, n, block_size_x=3):  # длина ключа, длина раундового ключа
    # раундовое преобразование:
    # для каждого блока делаем перестановку, сдвиг влево на число единиц в k_i и снова перестановку

    lst_k = gener_pattern(M)  # всевозможные ключ
    lst_x = gener_pattern(n)  # всевозможные тексты для шифрования
    lst = gener_pattern(block_size_x)
    matr = pd.DataFrame(columns=lst, index=lst).fillna(0)

    r = M // L  # число раундов = число блоков k

    for K in lst_k:
        for X in lst_x:
            lst_k_i = partition(K, L, r)
            lst_x_i = partition(X, block_size_x, len(X) // block_size_x)
            lst_y_i = y_shift_transformation(lst_x_i, lst_k_i, block_size_x, r)
            lst_y_i_new = y_xor_transformation(lst_y_i, lst_k_i, block_size_x, L, r)
            matr += transition_probability_matrix(lst_x_i, lst_y_i_new, block_size_x)

    matr = matr.div(matr.sum(axis=1), axis=0).fillna(0)
    print('Perm matrix')
    print(matr)
    return matr


def delt(matr):
    lst = gener_pattern(3)
    P = pd.DataFrame(columns=lst, index=lst).fillna(1 / 8)
    delt = (matr - P).applymap(lambda x: abs(x)).to_numpy().sum()
    print("Delt=", delt / 2)


# не проходит случай, когда длина сообщения < длина блока х
# грамотно работает для длины блока Х=3

if __name__ == '__main__':
    # #1 нет связи равномерности с r
    # print('xor________________________________________________________________________________________')
    # matr = round_transform_xor(2, 2, 3, 3)  # длина ключа, длина раундового ключа, длина сообщения, длина блока х
    # eigenvaluies(matr)
    # delt(matr)

    # #2 почти нет связи равномерности с r
    # print('shift________________________________________________________________________________________')
    # matr = round_transform_shift(4, 2, 9, 3)
    # eigenvaluies(matr)
    # delt(matr)

    # 3  2,14 2,6875     2,34    2,14
    print('xor+shift________________________________________________________________________________________')
    matr = round_transform_shift_xor(2, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr)




