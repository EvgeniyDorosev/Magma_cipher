import numpy as np
import pandas as pd
import itertools
from itertools import permutations


def second_abs_max(arr):
    new_arr = np.sort(np.absolute(arr))
    res = np.array([new_arr[i] for i in range(len(new_arr) - 1, 0, -1) if fabs(1 - new_arr[i]) > 0.00001])
    return res[0]


def partition(str, block_size, r):
    lst = []
    if r == 0:
        r = 1
    for i in range(r):
        lst.append(str[block_size * i:block_size * i + block_size])
    return lst


def gener_pattern(M):
    lst = []
    count = 0
    while True:
        temp = str(bin(count))[2:]
        count += 1
        if len(temp) <= M:
            lst.append(temp.zfill(M))
        else:
            break
    return lst


def transition_matrix(x, y, block_size_x): 
    n_states = block_size_x
    matrix = np.zeros((n_states, n_states))
    X = [int(el, 2) for el in x]
    Y = [int(el, 2) for el in y]
    # print(max(Y))
    for i in range(len(x)):
        matrix[X[i]][Y[i]] += 1
    for i in range(n_states):
        if sum(matrix[i]) > 0:
            matrix[i] /= sum(matrix[i])
    return matrix


def eigenvaluies(matr):
    eigenvalues, v = np.linalg.eig(matr)
    # print('Eigenvalues:')
    # for el in eigenvalues:
    #     print(el)
    eigenvalues = [abs(el) for el in eigenvalues]
    # print('Abs eigenvalues:')
    # for el in eigenvalues:
    #     print(el)
    # print('Second max eigenvalue:')
    return second_abs_max(eigenvalues)


def xor_str(x, k_i, block_size, L):  # L - длина раунд ключа
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


def round_transform_shift_xor(M, L, n, block_size_x=3):  # длина ключа, длина раундового ключа
    # раундовое преобразование:
    # для каждого блока делаем перестановку, сдвиг влево на число единиц в k_i и снова перестановку

    lst_k = gener_pattern(L)  # всевозможные ключ
    lst_x = gener_pattern(n)  # всевозможные тексты для шифрования

    r = M // L  # число раундов = число блоков k
    lst_input = []
    lst_output = []
    r = 1
    for K in lst_k:
        for X in lst_x:
            lst_k_i = partition(K, L, r)
            lst_x_i = partition(X, block_size_x, len(X) // block_size_x)
            lst_y_i = y_shift_transformation(lst_x_i, lst_k_i, block_size_x, r)
            lst_y_i_new = y_xor_transformation(lst_y_i, lst_k_i, block_size_x, L, r)
            lst_input.append(lst_x_i[0])
            lst_output.append((lst_y_i_new[0]))

    matr = transition_matrix(lst_input, lst_output, 8)
    print(matr)
    return matr


def delt(matr, matr_size):
    examp = np.full((matr_size, matr_size), 1 / matr_size)
    print("Max element in perm matr = ", np.max(matr))
    print("Min element in perm matr = ", np.min(matr))
    delt = np.sum(np.abs(matr - examp))
    print("Delt=", delt / 2)
    return delt


# ____________________________________________________________________________________________

def xor(num1, num2, base=2):
    '''
    Исключающее сложение
    :param num1: число №1
    :param num2: число №2
    :param base: система счисления
    :return: результат сложения
    '''
    len1 = len(str(num1))
    num1 = int(num1, base)
    num2 = int(num2, base)

    num = str(bin(num1 ^ num2)[2:])

    num = fillZerosBeforeNumber(num, len1)  # чтобы 4 бита было

    return num


def fillZerosBeforeNumber(num, length):
    '''
    Заполняет строку нулями (до числа)
    :param num: число
    :param length: длина итоговой строки
    :return: строка с нулями в начале и числом в конце
    '''
    num = str(num)
    if len(str(num)) != length:
        for i in range(length - len(str(num))):
            num = '0' + num
    return num


def fillZerosAfterNumber(num, length):
    """
    Заполняет строку нулями (после числа)
    :param num: число
    :param length: длина итоговой строки
    :return: строка с нулями в конце и числом в начале
    """
    num = str(num)
    if len(str(num)) != length:
        for i in range(length - len(str(num))):
            num = num + '0'
    return num


def cycleLeftShift(str, shift):
    shift = -shift % len(str)
    return str[-shift:] + str[:-shift]


def tableTransformation(numIn, subst, newBlocksSize, shift):
    '''
    Преобразует число путем разбития на newBlocksSize-битные и делает замены по таблице transformaion_table
    :param numIn: входное число
    :param newBlocksSize: новая длина блока после разбиения
    :return: Преобразованное число
    '''
    numOut = ''
    for i in range(2):
        num1 = numIn[i * newBlocksSize: i * newBlocksSize + newBlocksSize]
        num2 = bin(subst[i][int(num1, 2)])[2:]
        num2 = fillZerosBeforeNumber(num2, newBlocksSize)
        numOut += num2

    numOut = cycleLeftShift(numOut, shift)

    return numOut


def tableTransformation3(numIn, subst, newBlocksSize, shift):
    numOut = ''

    num2 = bin(subst[int(numIn, 2)])[2:]
    num2 = fillZerosBeforeNumber(num2, newBlocksSize)
    numOut += num2

    numOut = cycleLeftShift(numOut, shift)

    return numOut


def transformation(numLeft, numRight, key, subst, shift):
    '''
    Исполняет одну итерацию шифрования блока
    :param numLeft: левая половина блока в 4 бита
    :param numRight: правая половина блока в 4 бита
    :param key: ключ в 4 бита, соответствующий данной итерации
    :return: кортеж с двумя новыми половинами блока
    '''
    numLeftOut = numRight
    numRightOut = xor(numRight, key, 2)  # вот эти ключи нужно перебрать!!!!
    numRightOut = tableTransformation(numRightOut, subst, len(numLeft) // 2, shift)
    numRightOut = xor(numRightOut, numLeft, 2)  # это просто по алгоритму
    return numLeftOut, numRightOut


def transformation3(block, key, subst, shift):
    '''
    Исполняет одну итерацию шифрования блока
    :param block: левая половина блока в 4 бита
    :param numRight: правая половина блока в 4 бита
    :param key: ключ в 4 бита, соответствующий данной итерации
    :return: кортеж с двумя новыми половинами блока
    '''
    block = xor(block, key, 2)  # вот эти ключи нужно перебрать!!!!
    block = tableTransformation3(block, subst, 3, shift)
    return block


def chainOfTransformations(numLeft, numRight, keys, subst, shift, move='straight'):
    '''
    Исполняет все итерации шифрования блока
    :param numLeft: левая половина блока
    :param numRight: правая половина блока
    :param keys: массив ключей для каждой итерации
    :param subst: подстановка
    :param shift: параметр для циклического сдвига
    :param move: направление кодирования (straight для кодирования, reverse для декодирования)
    :return: кортеж с двумя новыми половинами блока
    '''
    start = 0
    stop = len(keys) - 1
    step = 1
    if move == 'reverse':
        start = len(keys) - 1
        stop = 0
        step = -1
    for i in range(start, stop, step):  # 3 раза подряд подстановка Фейстеля
        numLeft, numRight = transformation(numLeft, numRight, keys[i], subst, shift)
    numRightLast = numRight
    numLeft, numRight = transformation(numLeft, numRight, keys[stop], subst, shift)
    return numRight + numRightLast


def chainOfTransformations2(numLeft, numRight, key, subst, shift, move='straight'):
    '''
        Исполняет все итерации шифрования блока
        :param numLeft: левая половина блока
        :param numRight: правая половина блока
        :param keys: массив ключей раундовых
        :param subst: подстановка
        :param shift: параметр для циклического сдвига
        :param move: направление кодирования (straight для кодирования, reverse для декодирования)
        :return: кортеж с двумя новыми половинами блока
    '''
    numLeft, numRight = transformation(numLeft, numRight, key, subst, shift)
    return numLeft + numRight


def chainOfTransformations3(block, key, subst, shift, move='straight'):
    '''
        Исполняет все итерации шифрования блока
        :param block: левая половина блока
        :param numRight: правая половина блока
        :param keys: массив ключей раундовых
        :param subst: подстановка
        :param shift: параметр для циклического сдвига
        :param move: направление кодирования (straight для кодирования, reverse для декодирования)
        :return: кортеж с двумя новыми половинами блока
    '''
    block = transformation3(block, key, subst, shift)
    return block


def keyToKeys(key, keyRoundLen):
    '''
    Из 8-битного ключа создает массив 4-битных
    :param key:8-битный ключ
           keyRoundLen - длина раундового ключа
    :return: массив keyRoundLen-битных ключей
    '''
    keys = []
    for j in range(2):  # разбиение ключа на два keyRoundLen-битных куска
        keys.append(key[j * keyRoundLen: j * keyRoundLen + keyRoundLen])
    # повтор первого ключа в прямом порядке, а второго в обратном
    keys.append(keys[0])
    temp = ""
    for i in range(3, -1, -1):  # последний в обратном порядке
        temp += keys[1][i]
    keys.append(temp)
    return keys


def encode(text, key, subst, shift, textBlockLen, keyRoundLen):
    '''
    Шифрует строку
    :param text: строка длины 8 бит (но вообще не важно)
    :param key: ключ длины 8 бит
    :param subst: подстановка
    :param shift: параметр для циклического сдвига
    :param textBlockLen: длина блока текста
    :param keyRoundLen: длина раундового ключа
    :return: зашифрованная строка
    '''

    # keys = keyToKeys(key, keyRoundLen)

    if len(text) % textBlockLen != 0:
        text = fillZerosBeforeNumber(text, textBlockLen)
    textBlocksArray = []
    textEncrypt = ''
    if (len(text) // textBlockLen * textBlockLen) != len(text):  # count-число блоков
        count = len(text) // textBlockLen + 1
    else:
        count = len(text) // textBlockLen

    for i in range(count):
        textForAppend = text[i * textBlockLen: i * textBlockLen + textBlockLen]  # по блокам производим шифр
        textForAppend = fillZerosAfterNumber(textForAppend, textBlockLen)  # для нецелых блоков
        textBlocksArray.append(textForAppend)

    # for i in range(len(textBlocksArray)):
    #     textEncrypt += chainOfTransformations2(textBlocksArray[i][:textBlockLen // 2],
    #                                            textBlocksArray[i][textBlockLen // 2:], key,
    #                                            subst, shift)  # разбили блок на две части
    for i in range(len(textBlocksArray)):
        textEncrypt += chainOfTransformations3(textBlocksArray[i], key, subst, shift)  # разбили блок на две части
    return textEncrypt


def decode(text, key, subst, shift):
    '''
    Дешифрует строку
    :param text: строка
    :param key: ключ
    :param subst: подстановка
    :param shift: длина циклического сдвига
    :return: дешифрованная строка
    '''
    if len(text) % 8 != 0:
        text = fillZerosBeforeNumber(text, (len(text) // 8) * 8 + 8)
    textArray = []
    textDecrypt = ''
    if (len(text) // 64 * 64) != len(text):
        count = len(text) // 64 + 1
    else:
        count = len(text) // 64
    for i in range(count):
        textForAppend = text[i * 64: i * 64 + 64]
        textForAppend = fillZerosAfterNumber(textForAppend, 64)
        textArray.append(textForAppend)
    for i in range(len(textArray)):
        textDecrypt += chainOfTransformations(textArray[i][:8 // 2],
                                              textArray[i][8 // 2:], key,
                                              subst, shift, move='reverse')
    return textDecrypt


def magmaResearch(M, L, n, block_size_x=8):  # (12, 3, 12, 3)
    '''
        Исследование матрицы переходных вероятностей для шифра магма
        :param M: длина ключа
        :param L: длина раундового ключа
        :param n: длина текста шифрования
        :param block_size_x: длина блока текста
        :return: матрица переходных вероятностей
    '''
    lst_k = gener_pattern(L)  # всевозможные раундовые ключи
    lst_x = gener_pattern(n)  # всевозможные тексты для шифрования

    shift = 4

    # subst = [7, 5, 2, 0, 1, 4, 6, 3]
    subst = [0, 1, 3, 2, 7, 4, 6, 5]

    # subst = [[0, 1, 2, 3],
    #          [0, 1, 2, 3]]

    lst_output = []
    lst_input = []

    for K in lst_k:
        for X in lst_x:
            Y = encode(X, K, subst, shift, block_size_x, L)  # text, key, subst, shift, textBlockLen, keyRoundLen
            lst_input.append(X)
            lst_output.append(Y)

    matr = transition_matrix(lst_input, lst_output, 4096)
    print('Perm matrix')
    print(matr)
    print(eigenvaluies(matr))
    print(delt(matr, len(matr[0])))


if __name__ == '__main__':
    print('xor+shift________________________________________________________________________________________')

    matr = round_transform_shift_xor(2, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(4, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(6, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(8, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(10, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))

    matr = round_transform_shift_xor(12, 2, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(12, 3, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))
    
    matr = round_transform_shift_xor(12, 4, 3, 3)
    eigenvaluies(matr)
    delt(matr, len(matr[0]))

    print('magma________________________________________________________________________________________')

    magmaResearch(8, 4, 8, 8)

    magmaResearch(12, 3, 12, 3)

