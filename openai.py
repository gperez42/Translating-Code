import time
import openai
import os

# Set up OpenAI API credentials

openai.api_key = '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'

# Define the prompt
prompt = "Translate the following code from Java to C: public class Main { public static void main(String[] args){ int num = 4; System.out.println(num);}}"

# Start the timer
start_time = time.time()

# Call the API
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt,
  max_tokens=50
)

# End the timer
end_time = time.time()

# Calculate the elapsed time
total_time = end_time - start_time

# Print the elapsed time
print(f"Elapsed time: {Total_time} seconds")
