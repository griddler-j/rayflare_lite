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
csv_files = [file for file in files_in_directory if file=='Si_Crystalline, 300 K [Gre08].csv']
for file in csv_files:
    # Step 1: Read the CSV file
    df = pd.read_csv(file)
    header_names = df.columns
    
    num_columns = df.shape[1]
    # Step 2: Drop columns 5 and onwards
    df = df.iloc[:, :4]  # This selects all rows and first 5 columns
    intrinsic_k = df['k']

    for doping in [1e19, 3e19, 1e20, 3e20]:
        alpha_FCA = 1.68e-6*doping*(df['λ,k (nm)'].values*1e-7)**2.88
        # α = 4π k / λ
        k_FCA = alpha_FCA / (4*3.14159) * df['λ,k (nm)'].values*1e-7
        df['k'] = intrinsic_k + k_FCA

        # Step 3: Save the modified DataFrame back to a CSV file
        df.to_csv('n_type_c_Si_' + str(doping) + '.csv', index=False)