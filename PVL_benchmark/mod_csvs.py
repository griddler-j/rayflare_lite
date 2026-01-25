import pandas as pd
import os

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
# Get a list of all files in the current directory
files_in_directory = os.listdir()
# Filter out only the CSV files
csv_files = [file for file in files_in_directory if file.endswith('.csv')]
for file in csv_files:
    # Step 1: Read the CSV file
    df = pd.read_csv(file)
    header_names = df.columns
    
    num_columns = df.shape[1]
    if num_columns > 4 and is_number(header_names[0])==False:
        # Step 2: Drop columns 5 and onwards
        df = df.iloc[:, :4]  # This selects all rows and first 5 columns
        # Step 3: Save the modified DataFrame back to a CSV file
        df.to_csv(file, index=False)
        print("Modified ", file)