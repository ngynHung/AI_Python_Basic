def write_file(fn, names):
    with open(fn, "w") as file:
        for name in names:
            file.write(name + "\n")


def print_line_file(f):
    with open(f, "r") as file:
        for line in file:
            print(line.strip())


if __name__ == "__main__":
    file_name = "firstname.txt"
    names = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Hannah", "Ian", "Julia"]

    # write_file(file_name, names)
    # print_line_file(file_name)

    # file1 = open(file_name, "r")
    # print(file1.read())
