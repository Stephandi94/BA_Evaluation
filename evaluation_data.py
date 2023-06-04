import os
import csv
import pandas as pd
import numpy as np


def nd_array_to_csv(data, path, filename, header=None, row_description=None):
    """
    writes an nd np.array to csv
    :param data: nd numpy array (col, rows) [corresponding data is in rows]
    :param path: directory to store
    :param filename: file name w/o extension
    :param header: list of header strings, has to fit data.shape[1]; default None
    :param row_description: list of row description strings, has to fit data.shape[0]; default None
    :return:
    """
    dest = os.path.join(path, "{}.csv".format(filename))
    add_row_description = False
    if row_description is not None:
        if len(row_description) != data.shape[0]:
            raise Exception("nd_array_to_csv ERROR: row_description size does not fit amount of data rows")
        add_row_description = True
    if header is not None:
        if len(header) != data.shape[1]:
            raise Exception("nd_array_to_csv ERROR: header size does not fit data column size!")
        if add_row_description:
            header.insert(0, "Type")
    with open(dest, 'w', newline='') as file:
        writer = csv.writer(file)
        if header is not None:
            writer.writerow(header)
        for i in range(data.shape[0]):
            if add_row_description:
                row = data[i, :].tolist()
                row.insert(0, row_description[i])
                writer.writerow(row)
            else:
                writer.writerow(data[i, :].tolist())
    print("{} is written".format(dest))

def stats_1x3_to_csv(data, path, filename, header=["Mean", "Min", "Max"]):
    """
    prints 1x3 stats array to csv
    :param data: numpy array (1, 3)
    :param path: directory to store
    :param filename: file name w/o extension
    :param header: 3 element list of header strings, if not intended set None; default ["Mean", "Min", "Max"]
    """
    nd_array_to_csv(data.reshape(1, -1), path, filename, header)

def stats_3x3_to_csv(data, path, filename, row_description=["Mean Values", "Min Values", "Max Values"], header=["Mean", "Min", "Max"]):
    """
    prints 3x3 stats array to csv
    :param data: numpy array (3, 3)
    :param path: directory to store
    :param filename: file name w/o extension
    :param header: 3 element list of header strings, if not intended set None; default ["Mean", "Min", "Max"]
    :param row_description: 3 element list of row description strings, if not intended set None, default ["Mean Values", "Min Values", "Max Values"]
    """
    nd_array_to_csv(data, path, filename, header, row_description)

def stats_to_csv(points, distances, intensities, path, filename):
    """
    brings stats about point numbers, distances and intensities together
    :param points: (1, 3) array with point stats
    :param distances: (3, 3) array with distance stats
    :param intensities: (3, 3) array with intensity stats
    :param path: destination directory
    :param filename: filename w/o extension
    """
    dest = os.path.join(path, r"{}.csv".format(filename))
    output = [
        ["Points Per PCL", points[0], points[1], points[2]],
        ["Distances Mean Values", distances[0, 0], distances[0, 1], distances[0, 2]],
        ["Distances Min Values", distances[1, 0], distances[1, 1], distances[1, 2]],
        ["Distances Max Values", distances[2, 0], distances[2, 1], distances[2, 2]],
        ["Intensities Mean Values", intensities[0, 0], intensities[0, 1], intensities[0, 2]],
        ["Intensities Min Values", intensities[1, 0], intensities[1, 1], intensities[1, 2]],
        ["Intensities Max Values", intensities[2, 0], intensities[2, 1], intensities[2, 2]]
    ]
    with open(dest, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Measurement", "Mean", "Min", "Max"])
        writer.writerows(output)


def read_csv_to_array(source, dummy=False, has_description=False):
    """
    TODO
    :param source:
    :param has_header:
    :param has_description:
    :return: np.array
    """
    df = pd.read_csv(source)
    if has_description:
        return df.values[:, 1:]
    return df.values
    # with open(source, 'r') as file:
    #     reader = csv.reader(file)
    #     data = list(reader)
    #
    # if has_header:
    #     data.pop(0)
    # if has_description:
    #     output = [l[1:] for l in data]
    #     data = output
    # return np.array(data, dtype=dtype)

