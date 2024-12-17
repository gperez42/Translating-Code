import time
import openai
import os



# Set up OpenAI API credentials

openai.api_key = '' # enter your API key here

messages = [ {"role": "system", "content":
              "You are a intelligent assistant."} ]
while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        chat = openai.Completion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})

# def time_llm_response(prompt):
#
#     start_time = time.time()
#
#     response = openai.ChatCompletion.create(prompt=prompt)
#
#     end_time = time.time()
#
#     response_time = end_time - start_time
#
#     return response, response_time
#
#
#
# # Example usage
#
# prompt = "What is the capital of France?"
#
# start_time = time.time();
#
# response = openai.Completion.create(
#     engine ="text-davinci-003",
#     prompt=prompt,
#     max_takens=50
# )
#
# end_time = time.time();
#
# total_time = end_time - start_time
#
# # response, response_time = time_llm_response(prompt)
#
# print(f"Answer: {response.choices[0].text}, Response time: {total_time} seconds")

# import time
# import openai
#
# # Define your API key
# openai.api_key = 'your_api_key_here'
#
# # Define the prompt
# prompt = "Your prompt here"
#
# # Start the timer
# start_time = time.time()
#
# # Call the API
# response = openai.Completion.create(
#   engine="text-davinci-003",
#   prompt=prompt,
#   max_tokens=50
# )
#
# # End the timer
# end_time = time.time()
#
# # Calculate the elapsed time
# elapsed_time = end_time - start_time
#
# # Print the elapsed time
# print(f"Elapsed time: {elapsed_time} seconds")
