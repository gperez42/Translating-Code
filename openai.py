import time
import openai
import os

# Set up OpenAI API credentials
# enter your own API key
openai.api_key = 'API key' 

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
