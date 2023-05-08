import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter


def draw_bar_chart(xtick_labels, datasets, dataset_titles,
                   x_label=None,
                   y_label=None,
                   title=None, figure_path=None, figure_width=10, figure_height=5):
    total_width = 0.9

    single_width = total_width / len(datasets)

    # sort the labels and data by first dataset ascending order
    sorted_index = np.argsort(np.array(datasets[0]))
    xtick_labels = [xtick_labels[i] for i in sorted_index]
    for i, dataset in enumerate(datasets):
        datasets[i] = [dataset[j] for j in sorted_index]
    label_locations = np.arange(len(xtick_labels))

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    for i in range(len(datasets)):
        mid = len(datasets) / 2
        if i < mid:
            rect = ax.bar(label_locations - (mid - i) * single_width + (single_width / 2),
                          datasets[i], single_width,
                          label=dataset_titles[i])
        else:
            rect = ax.bar(label_locations + (i - mid) * single_width + (single_width / 2),
                          datasets[i], single_width,
                          label=dataset_titles[i])
        ax.bar_label(rect, fmt='%.2f', fontsize=12)

    if x_label:
        ax.set_xlabel(x_label, fontsize=16)
    if y_label:
        ax.set_ylabel(y_label, fontsize=16)
    if title:
        ax.set_title(title)
    if len(datasets[0]) > 7:
        ax.set_xticks(label_locations, xtick_labels, rotation='vertical', fontsize=16)
    else:
        ax.set_xticks(label_locations, xtick_labels, fontsize=16)
    ax.legend(fontsize=16)

    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    fig.tight_layout()
    if figure_path == None:
        figure_path = 'output/bar.png'
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved in {figure_path}")


def show_cuda_pinned_unified():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge']
    pinned_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481]
    pinned_dataset_title = "Pinned memory"
    unified_dataset = [11.4198, 45.7082, 2.2018, 7.3043, 1.1011, 0.3776, 0.3453]
    unified_dataset_title = "Unified memory"
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04']
    pinned_dataset = [4.3180, 11.2749, 0.7202, 2.0920]
    pinned_dataset_title = "Pinned memory"
    unified_dataset = [11.4198, 45.7082, 2.2018, 7.3043]
    unified_dataset_title = "Unified memory"
    x_label = "Datasets"
    y_label = "Execution Time(s) (log scale)"
    output_filename = os.path.join("output", "pinned_vs_unified.png")
    draw_bar_chart(dataset_labels, [pinned_dataset, unified_dataset], [pinned_dataset_title, unified_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=10, figure_height=5)


def show_cuda_cudf_join():
    dataset_labels = ['R 1000000', 'R 2000000', 'R 3000000', 'R 4000000', 'R 5000000',
                      'S 1000000', 'S 2000000', 'S 3000000', 'S 4000000', 'S 5000000']
    cuda_dataset = [0.118800, 0.443422, 0.986914, 1.727381, 2.640382, 0.014831, 0.022604, 0.031009, 0.038893, 0.049386]
    cuda_dataset_title = "CUDA Single Hashjoin"
    cudf_dataset = [1.009139, 3.559351, 7.486542, 13.013476, 20.446236, 0.144650, 0.363000, 0.577207, 0.826158,
                    1.062329]
    cudf_dataset_title = "cuDF"
    x_label = "Datasets"
    y_label = "Execution Time(s) (log scale)"
    output_filename = os.path.join("output", "cuda_vs_cudf_join.png")
    draw_bar_chart(dataset_labels, [cuda_dataset, cudf_dataset], [cuda_dataset_title, cudf_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=10, figure_height=6)


def show_cuda_souffle():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge',
                      'luxembourg_osm', 'wiki-Vote', 'wing', 'ego-Facebook', 'delaunay_n16',
                      'fe_sphere', 'loc-Brightkite', 'fe_ocean', 'cti', 'fe_body', 'usroads']
    cuda_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481, 1.3222, 1.1372, 0.0857,
                    0.5442, 1.1374, 13.1590, 15.8805, 138.2379, 0.2953, 47.7587, 364.5549]
    cuda_dataset_title = "CUDA Hashjoin (our implementation)"
    souffle_dataset = [15.206, 17.073, 3.094, 7.537, 0.455, 0.219, 0.181, 2.548, 3.172, 0.193,
                       0.606, 1.612, 20.008, 29.184, 536.233, 1.496, 29.070, 222.761]
    souffle_dataset_title = "Souffle"
    x_label = "Datasets"
    y_label = "Execution Time(s) (log scale)"
    output_filename = os.path.join("output", "cuda_vs_souffle.png")
    draw_bar_chart(dataset_labels, [cuda_dataset, souffle_dataset],
                   [cuda_dataset_title, souffle_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=14, figure_height=6)


def show_cuda_cudf():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge',
                      'luxembourg_osm', 'wiki-Vote', 'wing', 'ego-Facebook', 'delaunay_n16',
                      'fe_sphere', 'loc-Brightkite', 'fe_ocean', 'cti', 'fe_body', 'usroads']
    cuda_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481, 1.3222, 1.1372, 0.0857,
                    0.5442, 1.1374, 13.1590, 15.8805, 138.2379, 0.2953, 47.7587, 364.5549]
    cuda_dataset_title = "CUDA Hashjoin (our implementation)"
    cudf_dataset = [26.115098, 64.417961, 3.906619, 14.005228, 2.756417, 0.857208, 0.523132,
                    8.194708, 6.841340, 0.905852, 3.719607, 5.596315, 80.077607, 0, 0, 3.181488, 0, 0]
    cudf_dataset_title = "cuDF"
    x_label = "Datasets"
    y_label = "Execution Time(s) (log scale)"
    output_filename = os.path.join("output", "cuda_vs_cudf.png")
    draw_bar_chart(dataset_labels, [cuda_dataset, cudf_dataset],
                   [cuda_dataset_title, cudf_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=14, figure_height=6)


def show_cuda_souffle_cudf():
    dataset_labels = ['CA-HepTh', 'SF.cedge', 'p2p-Gnutella09', 'p2p-Gnutella04', 'cal.cedge', 'TG.cedge', 'OL.cedge',
                      'luxembourg_osm', 'wiki-Vote', 'wing', 'ego-Facebook', 'delaunay_n16',
                      'fe_sphere', 'loc-Brightkite', 'fe_ocean', 'cti', 'fe_body', 'usroads']
    cuda_dataset = [4.3180, 11.2749, 0.7202, 2.0920, 0.4894, 0.1989, 0.1481, 1.3222, 1.1372, 0.0857,
                    0.5442, 1.1374, 13.1590, 15.8805, 138.2379, 0.2953, 47.7587, 364.5549]
    cuda_dataset_title = "CUDA Hashjoin (our implementation)"
    souffle_dataset = [15.206, 17.073, 3.094, 7.537, 0.455, 0.219, 0.181, 2.548, 3.172, 0.193,
                       0.606, 1.612, 20.008, 29.184, 536.233, 1.496, 29.070, 222.761]
    souffle_dataset_title = "Souffle"
    cudf_dataset = [26.115098, 64.417961, 3.906619, 14.005228, 2.756417, 0.857208, 0.523132,
                    8.194708, 6.841340, 0.905852, 3.719607, 5.596315, 80.077607, 0, 0, 3.181488, 0, 0]
    cudf_dataset_title = "cuDF"
    x_label = "Datasets"
    y_label = "Execution Time(s) (log scale)"
    output_filename = os.path.join("output", "cuda_vs_souffle_vs_cudf.png")
    draw_bar_chart(dataset_labels, [cuda_dataset, souffle_dataset, cudf_dataset],
                   [cuda_dataset_title, souffle_dataset_title, cudf_dataset_title],
                   x_label, y_label,
                   figure_path=output_filename, figure_width=16, figure_height=6)


if __name__ == "__main__":
    show_cuda_cudf_join()
    # show_cuda_pinned_unified()
    # show_cuda_souffle()
    # show_cuda_cudf()
    # show_cuda_souffle_cudf()
