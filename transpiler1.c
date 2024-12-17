#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// clock code from 5.2 powerpoint
// double  get_clock() {
//      struct timeval tv;
//      int ok;
//      ok = gettimeofday(&tv, (void *) 0);
//      if (ok<0) {
//              printf(“gettimeofday error”);
//      }
//      return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
// }

// Function to process a line of Java code and generate C code
void processLine(char *line, FILE *outputFile) {
    // Translate variable declarations
    if (strncmp(line, "int ", 4) == 0 || strncmp(line, "float ", 6) == 0 || strncmp(line, "double ", 7) == 0 || strncmp(line, "String", 7) == 0) {
        fprintf(outputFile, "%s;\n", line);
    }
    // Translate print statements
    else if (strncmp(line, "System.out.println", 18) == 0 || strncmp(line, "System.out.print", 16) == 0) {
        char *text = strchr(line, '(') + 1;
        char *end = strrchr(text, ')');
        *end = '\0'; // Remove the closing parenthesis
        fprintf(outputFile, "printf(\"%%d\\n\", %s);\n", text);
    }
    // Translate basic arithmetic operations
    else if (strchr(line, '=') != NULL) {
        fprintf(outputFile, "%s;\n", line);
    }

//    else if (strncmp(line, "import", 6) == 0 || strncmp(line, "", 0) == 0){}
    // Handle unsupported lines
    else {
        fprintf(stderr, "Unsupported line: %s\n", line);
    }
}
int main(int argc, char *argv[]) {
//      float t0 = get_clock(); // start timer
        clock_t t0 = clock(); // start timer
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
        line[strcspn(line, "\n")] = '\0'; // Remove newline character
         processLine(line, outputFile);
    }

    fprintf(outputFile, "return 0;\n}\n");

    fclose(inputFile);
    fclose(outputFile);

        //float t1 = get_clock(); // end timer
        clock_t t1 = clock();
        printf("time per call: %f ns\n", (float)((t1-t0)) );

    return 0;
}
