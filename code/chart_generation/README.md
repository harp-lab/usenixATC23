## Chart generator
- Generate stacked bar chart on the iteration results (Join, Union, Deduplication, and memory clear time of each iteration).
- Generate bar chart comparing CUDA pinned memory with CUDA unified memory model.

### Run instruction
- Install the packages listed in [requirements.txt](requirements.txt) using Python 3 and pip
- Run the file [generate_stacked_bar_chart.py](generate_stacked_bar_chart.py) for iteration level stacked bar chart generation 
- Run the file [generate_bar_chart.py](generate_bar_chart.py) for memory model bar chart generation

### Output directory
- Stacked bar charts generated from the input [datafiles](data) are stored in [output](output) folder
- Bar charts generated comparing memory model: [output/pinned_vs_unified.png](output/pinned_vs_unified.png)

### Demo 
#### Comparison between CUDA and Souffle
![alt CUDA vs souffle](output/cuda_vs_souffle.png)

#### Comparison between CUDA and cuDF
![alt CUDA vs cuDF](output/cuda_vs_cudf.png)

#### Comparison between CUDA and Souffle and cuDF
![alt CUDA vs souffle vs cuDF](output/cuda_vs_souffle_vs_cudf.png)

#### Comparison between CUDA pinned memory and CUDA unified memory
![alt memory model comparison](output/pinned_vs_unified.png)

#### Comparison between isolated join operation on CUDA and cuDF
![alt JOIN comparison](output/cuda_vs_cudf_join.png)

#### Iteration level operation breakdown
- SF.cedge:
  ![alt sf.cedge](output/SF.cedge.png)
- CA-HepTh:
  ![alt CA-HepTh](output/CA-HepTh.png)
- fe_body:
  ![alt fe_body](output/fe_body.png)
- fe_sphere:
  ![alt fe_sphere](output/fe_sphere.png)

- For more iteration level operation breakdown check [output](output) folder.