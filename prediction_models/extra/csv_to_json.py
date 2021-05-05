import csv
import json
from collections import defaultdict
# Function to convert a CSV to JSON
# Takes the file paths as arguments
def make_json(csvFilePath, jsonFilePath):
    columns = defaultdict(list)
    # create a dictionary
    data = []

    # Open a csv reader called DictReader
    with open(csvFilePath, encoding='utf-8-sig') as csvf:
        csvReader = csv.DictReader(csvf)
        # Convert each row into a dictionary
        # and add it to data
        for rows in csvReader:  # read a row as {column1: value1, column2: value2,...}
            # Assuming a column named 'No' to
            # be the primary key
            # key = rows['no']
            data.append(rows['review'])
            # Open a json writer, and use the json.dumps()
    # function to dump data
    with open(jsonFilePath, 'w', encoding='utf-8-sig') as jsonf:
        jsonf.write(json.dumps(data, indent=4, ensure_ascii=False))


# Driver Code

# Decide the two file paths according to your
# computer system
csvFilePath = r'dataset.csv'
jsonFilePath = r'dataset.json'

# Call the make_json function
make_json(csvFilePath, jsonFilePath)
