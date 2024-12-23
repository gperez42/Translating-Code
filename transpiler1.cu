#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define MAX_LINES 1024
#define MAX_LINE_LENGTH 256

__device__ void device_strcpy(char *dest, const char *src) {
    int i = 0;
    while (src[i] != '\0') {
        dest[i] = src[i];
        i++;
    }
    dest[i] = '\0';  // Null-terminate the string
}

__device__ char* device_strchr(char *str, char c) {
    int i = 0;
    while (str[i] != '\0') {
        if (str[i] == c) {
            return &str[i];
        }
        i++;
    }
    return NULL;
}

__device__ char* device_strrchr(char *str, char c) {
    int i = 0;
    char *last_occurrence = NULL;
    while (str[i] != '\0') {
        if (str[i] == c) {
            last_occurrence = &str[i];
        }
        i++;
    }
    return last_occurrence;
}

__device__ int device_strlen(const char *str) {
    int len = 0;
    while (str[len] != '\0') {
        len++;
    }
    return len;
}

__device__ void device_strcat(char *dest, const char *src) {
    int dest_len = device_strlen(dest);
    int i = 0;
    while (src[i] != '\0') {
        dest[dest_len + i] = src[i];
        i++;
    }
    dest[dest_len + i] = '\0';  // Null-terminate the resulting string
}

__device__ void processPrintStatement(const char *line, char *output) {
    char *openParen = device_strchr((char *)line, '(');   // Use device_strchr
    char *closeParen = device_strrchr((char *)line, ')'); // Use device_strrchr

    if (openParen && closeParen) {
        openParen++;  // Move past the '('
        int length = closeParen - openParen;

        if (length > 0) {
           if (length > 0) {
            if (*openParen == '"') {  // String literal
                device_strcpy(output, "printf(\"%s\\n\", ");
            } else {  // Integer or variable
                device_strcpy(output, "printf(\"%d\\n\", ");
            }

              // Copy the content inside parentheses
            int offset = device_strlen(output);
            for (int i = 0; i < length; i++) {
                output[offset + i] = openParen[i];
            }
            output[offset + length] = '\0';

            
            // Add closing parenthesis and semicolon
            device_strcat(output, ");");
        }
     } else {
        // Handle invalid print statement
        device_strcpy(output, "/* Invalid print statement */\n");
       }
    }
}

// Kernel to process the lines on the GPU
__global__ void processLineKernel(char *d_lines, int *d_flags, char *d_output, int numLines) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numLines) {
    	// char *line = d_lines + idx * MAX_LINE_LENGTH;
        char temp[MAX_LINE_LENGTH];  // Temporary buffer for output
        // int out_idx = 0;  // Index for temp buffer

        // Process variable declarations or print statements based on flags
        if (d_flags[idx] == 1) {  // Variable declaration
            device_strcpy(d_output + idx * MAX_LINE_LENGTH, d_lines + idx * MAX_LINE_LENGTH);
        } else if (d_flags[idx] == 2) {  // Print statement           
        	processPrintStatement(d_lines + idx * MAX_LINE_LENGTH, temp);
        	device_strcpy(d_output + idx * MAX_LINE_LENGTH, temp);    
        } else if (d_flags[idx] == 3) {  // Assignment operation
            device_strcpy(d_output + idx * MAX_LINE_LENGTH, d_lines + idx * MAX_LINE_LENGTH);
 		}  else if (d_flags[idx] == 4) {  // Handle #include <stdio.h>
            device_strcpy(d_output + idx * MAX_LINE_LENGTH, d_lines + idx * MAX_LINE_LENGTH);
        } else if (d_flags[idx] == 5) {  // Unsupported line
            device_strcpy(d_output + idx * MAX_LINE_LENGTH, "Unsupported line\n");
        }
        
     }
}

// Host-side function to preprocess the lines
void processLine(const char *line, FILE *outputFile, int *flags, int idx) {
    if (strncmp(line, "int ", 4) == 0 || strncmp(line, "float ", 6) == 0 || 
        strncmp(line, "double ", 7) == 0 || strncmp(line, "String", 7) == 0) {
        flags[idx] = 1; // Mark this line as a variable declaration
    }
    else if (strncmp(line, "System.out.println", 18) == 0 || strncmp(line, "System.out.print", 16) == 0) {
        flags[idx] = 2; // Mark this line for print statement handling
    }
    else if (strchr(line, '=') != NULL) {
        flags[idx] = 3; // Mark for simple assignment handling
    }
    else if (strncmp(line, "import", 6) == 0 || strncmp(line, "", 0) == 0) {
        flags[idx] = 0; // Skip imports or empty lines
    }
    else if (strncmp(line, "#include <stdio.h>", 18) == 0) {
        flags[idx] = 4; // Mark this line as a special include line
    }
    else {
        flags[idx] = 5; // Unsupported line
    }
}

double get_clock() {
  struct timeval tv; 
  int ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { 
  	printf("gettimeofday error"); 
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// Main function
int main(int argc, char *argv[]) {

	double t0 = get_clock();

    char lines[MAX_LINES][MAX_LINE_LENGTH];
    int flags[MAX_LINES];
    FILE *inputFile = fopen(argv[1], "r");
    FILE *outputFile = fopen(argv[2], "w");

    if (!inputFile || !outputFile) {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int numLines = 0;
    while (fgets(lines[numLines], sizeof(lines[numLines]), inputFile)) {
        processLine(lines[numLines], outputFile, flags, numLines);
        numLines++;
    }

     // Write the #include <stdio.h> line first
    fprintf(outputFile, "#include <stdio.h>\n\nint main() {\n");

    fclose(inputFile);

    // Prepare the data for CUDA
    char *d_lines, *d_output;
    int *d_flags;

    cudaMalloc((void**)&d_lines, MAX_LINES * MAX_LINE_LENGTH * sizeof(char));
    cudaMalloc((void**)&d_flags, MAX_LINES * sizeof(int));
    cudaMalloc((void**)&d_output, MAX_LINES * MAX_LINE_LENGTH * sizeof(char));

    cudaMemcpy(d_lines, lines, MAX_LINES * MAX_LINE_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, flags, MAX_LINES * sizeof(int), cudaMemcpyHostToDevice);

     // Launch the kernel with numLines blocks and threads per block
    int threadsPerBlock = 256;
    int blocks = (numLines + threadsPerBlock - 1) / threadsPerBlock;
    processLineKernel<<<blocks, threadsPerBlock>>>(d_lines, d_flags, d_output, numLines);

    // Check for kernel errors
    cudaDeviceSynchronize();
    
    // Copy the processed output back to the host
    char output[MAX_LINES][MAX_LINE_LENGTH];
    cudaMemcpy(output, d_output, MAX_LINES * MAX_LINE_LENGTH * sizeof(char), cudaMemcpyDeviceToHost);

     // Write the processed lines to the output file
    for (int i = 0; i < numLines; i++) {
        if (strlen(output[i]) > 0) {
            fprintf(outputFile, "%s\n", output[i]);
        }
    }

	fprintf(outputFile, "return 0;\n}\n");

	double t1 = get_clock();
    printf("time per call: %f s\n", t1-t0);

    // Free memory
    cudaFree(d_lines);
    cudaFree(d_flags);
    cudaFree(d_output);

    fclose(outputFile);
    return 0;
}
