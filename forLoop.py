def print_name(name):
    for n in name:
        print(n)


def print_odd(start, end):
    for x in range(start - 1, end + 1):
        if x % 2 == 0:
            print(x)


def sum_devided_2(start, end):
    s = 0
    for x in range(start - 1, end + 1):
        if x % 2 == 0:
            s += x
    return s


def sum_range(start, end):
    s = 0
    for x in range(start - 1, end + 1):
        s += x
    return s


def print_dict(dict):
    print("Keys:", list(dict.keys()))
    print("Values:", list(dict.values()))
    print("Keys and values:", dict)


def merge_sequence_tuples(x, y):
    """
    :param x: tuple1
    :param y: tuple2
    :return: dict has keys tuple1 & values tuple2
    """
    dict = {x[i]: y[i] for i, _ in enumerate(x)}
    return dict


def count_consonants(s):
    return len([x for x in s if x not in 'ueoai'])


def range_10_devided(start, end):
    i = 0
    try:
        for x in range(start - 1, end + 1):
            i = x
            if 10 % x == 0:
                print(x)

    except:
        print("Can not devided by zero !")

    finally:
        for x in range(i + 1, end + 1):
            if 10 % x == 0:
                print(x)


if __name__ == '__main__':
    # print_name("Nguyen Hung")
    # print_odd(1, 10)
    # print(sum_devided_2(1, 10))
    # print(sum(1, 6))
    # print(sum_range(1, 6))
    # myDict = {
    #     "a": 1,
    #     "b": 2,
    #     "c": 3,
    #     "d": 4
    # }
    # print_dict(myDict)

    # WAY1
    # courses = [131, 141, 142, 212]
    # names = ["Maths", "Physics", "Chem", "Bio"]
    # myDict = dict(zip(courses, names))
    # print_dict(myDict)
    # WAY2
    # print(merge_sequence_tuples(courses, names))

    # print(count_consonants("jabbawocky"))

    # range_10_devided(-2, 3)

    ages = [23, 10, 80, 15]
    names = ["Hoa", "Lam", "Nam", "An"]
    myDict = list(zip(names, ages))
    print(myDict)
    myDict = sorted(myDict, key=lambda t: t[0]) #sort by first element in tuple
    myDict = sorted(myDict, key=lambda t: t[1]) #sort by second element in tuple
    print(myDict)

