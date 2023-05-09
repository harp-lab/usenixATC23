# Specify the base image
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        make \
        gcc \
        g++ \
        && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy the makefile and the source files to the container
COPY Makefile ./
COPY tc_cuda.cu ./
COPY hashjoin.cu ./
COPY common ./common/
COPY data ./data/

# Build the executable
RUN make build
# Set the entry point to run the executable
ENTRYPOINT ["/app/tc_cuda.out"]

