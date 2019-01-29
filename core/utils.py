import os


def read_list(path):
    file_list = []
    for root, _, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] == '.png':
                file_list.append(os.path.join(root, file))
    return file_list
