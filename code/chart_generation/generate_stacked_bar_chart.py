import os
import math
import numpy as np
import matplotlib.pyplot as plt


def read_data(file_path):
    iterations = []
    join_time = []
    union_time = []
    deduplication_time = []
    memory_clear_time = []
    with open(file_path) as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if line != "" and i > 3:
                line = line.replace(" ", "")[1:-1]
                values = line.split("|")
                iterations.append(int(values[0]))
                join_time.append(float(values[-4]))
                union_time.append(float(values[-3]))
                deduplication_time.append(float(values[-2]))
                memory_clear_time.append(float(values[-1]))
    return iterations, np.array(join_time), np.array(union_time), \
        np.array(deduplication_time), np.array(memory_clear_time)


def draw_stacked_bar(data, dataset_name):
    labels = data[0]
    width = 0.60
    fig, ax = plt.subplots()
    ax.bar(labels, data[1], width, label='Join')
    ax.bar(labels, data[2], width, bottom=data[1], label='Union')
    ax.bar(labels, data[3], width, bottom=data[1] + data[2], label='Deduplication')
    ax.bar(labels, data[4], width, bottom=data[1] + data[2] + data[3], label='Memory clear')
    ax.set_xlabel('Iteration', fontsize=16)
    ax.set_ylabel('Time(s)', fontsize=16)
    # ax.set_title(f'Time breakdown for {dataset_name}')
    ax.legend(fontsize=16)
    ax.xaxis.get_major_locator().set_params(integer=True)
    plt.xlim([0, data[0][-1] + 1])
    plt.gcf().set_size_inches(8, 4)
    plt.tight_layout()
    output_filename = os.path.join("output", f"{dataset_name}.png")
    plt.savefig(output_filename, bbox_inches='tight', dpi=600)
    print(f"Generated {output_filename}")
    plt.close()


if __name__ == "__main__":
    file_directory = "data"
    filenames = [filename for filename in os.listdir(file_directory) if filename.endswith(".txt")] #
    for filename in filenames:
        file_path = os.path.join(file_directory, filename)
        data = read_data(file_path)
        draw_stacked_bar(data, filename.split(".txt")[0])
