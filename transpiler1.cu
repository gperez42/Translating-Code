#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void processLine(const char *line, FILE *outputFile) {
    if (strncmp(line, "int ", 4) == 0 || strncmp(line, "float ", 6) == 0 || strncmp(line, "double ", 7) == 0 || strncmp(line, "String", 7) == 0)
        fprintf(outputFile, "%s;\n", line);
    else if (strncmp(line, "System.out.println", 18) == 0 || strncmp(line, "System.out.print", 16) == 0) {
        const char *text = strchr(line, '(') + 1;
        char *end = (char*)strrchr(text, ')');
        if (end) *end = '\0';
        fprintf(outputFile, "printf(\"%%s\\n\", %s);\n", text);
    } else if (strchr(line, '=') != NULL)
        fprintf(outputFile, "%s;\n", line);
    else if (strncmp(line, "import", 6) == 0 || strcmp(line, "") == 0)
        return;
    else
        fprintf(stderr, "Unsupported line: %s\n", line);
}

__global__ void gpuCompute(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 2;
}




int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <output_file>\n", argv[0]);
        return 1;
    }
    FILE *inputFile = fopen(argv[1], "r");
    if (inputFile == NULL) {
        perror("Error opening input file");
        return 1;
    }
 	FILE *outputFile = fopen(argv[2], "w");
    if (outputFile == NULL) {
        perror("Error opening output file");
        fclose(inputFile);
        return 1;
    }
    fprintf(outputFile, "#include <stdio.h>\n\nint main() {\n");
    char line[256];
    while (fgets(line, sizeof(line), inputFile)) {
        line[strcspn(line, "\n")] = '\0';
        processLine(line, outputFile);
    }
 	 fprintf(outputFile, "return 0;\n}\n");
    fclose(inputFile);
    fclose(outputFile);

    int n = 10;
    int *data;
    cudaMallocManaged(&data, n * sizeof(int));
    for (int i = 0; i < n; i++) data[i] = i;
    gpuCompute<<<1, 128>>>(data, n);
    cudaDeviceSynchronize();
    for (int i = 0; i < n; i++) printf("data[%d] = %d\n", i, data[i]);
    cudaFree(data);
    return 0;
}













