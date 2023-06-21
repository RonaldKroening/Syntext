import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")

using CSV, DataFrames

# Read the CSV file with appropriate options
csv_file = CSV.File("dataset.csv", header=true, delim=",")

# Convert the CSV.File to a DataFrame
data = DataFrame(csv_file)

# Print the head of the DataFrame
first(data, 5)
