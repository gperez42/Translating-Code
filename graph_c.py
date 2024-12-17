import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# import csv
# Open the file in write mode
# with open("graph_c.csv", "w") as csv_file:
#     # Create a writer object
#     csv_writer = csv.writer(csv_file)
#     # Write the data to the file
#     csv_writer.writerow(["Test #", "Time (nanoseconds)"])
#     csv_writer.writerow([1, 194.000000])
#     csv_writer.writerow([2, 157.000000])
#     csv_writer.writerow([3, 189.000000])
#     csv_writer.writerow([4, 171.000000])
#     csv_writer.writerow([5, 168.000000])
#     csv_writer.writerow([6, 180.000000])
#     csv_writer.writerow([7, 190.000000])
#     csv_writer.writerow([8, 167.000000])
#     csv_writer.writerow([9, 178.000000])
#     csv_writer.writerow([10,170.000000])
# Apply the default theme sns.set_theme()
# Load an example dataset tips = sns.load_dataset("tips")
# create a dataframe of the test number and its results
# initialize list of lists
#data = [[1, 194.000000], [2, 157.000000], [3, 189.000000], [4, 171.000000], [5, 168.000000], [6,180.000000], [7,190.000000],[8,167.000000],[9,178.000000], [10, 170.000000]]
data = {'Test #': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'Time (nanoseconds)': [194.000000, 157.000000, 189.000000, 171.000000, 168.000000, 180.000000, 190.000000, 167.000000, 178.000000, 170.000000]}

# Create the pandas DataFrame
df = pd.DataFrame(data)

# Create a visualization
#sns.relplot( data=data, x="total_bill", y="tip", col="time", ) hue="smoker",style="smoker", size="size",
sns.scatterplot(x='Test #', y='Time (nanoseconds)', data=df)
# Display the plot
plt.show()
