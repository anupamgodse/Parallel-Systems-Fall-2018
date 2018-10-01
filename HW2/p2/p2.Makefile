p2: p2.cu
	nvcc p2.cu -o p2 -O3 -lm -Wno-deprecated-gpu-targets
