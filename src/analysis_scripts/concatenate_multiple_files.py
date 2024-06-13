import os
import glob
import pandas as pd

"""
2023-06-23
concatenate multiple csv or xlsx files with manual scoring
this was originally used to compare performances of two human observers

*backbone script provided by Chat-GPT
"""


def get_EV_data(filename_in):
    '''
    Get Ethovision file
    >>> Number of header rows must be 37! --> see skiprows
    '''
    ethovision_data_out = pd.read_excel(filename_in, skiprows=37)
    ethovision_data_out.columns = ['Trial time','Recording time','X center','Y center','Area','Areachange','Elongation','grooming','light','Result 1']
    return(ethovision_data_out.drop(['Result 1'], axis=1))


# Root directory path
ROOT = r"..\..\results\raw_data_output"

# Get a list of CSV and XLSX files in the directory
csv_files = glob.glob(os.path.join(ROOT, '*.csv'))
xlsx_files = glob.glob(os.path.join(ROOT, '*.xlsx'))
files = csv_files + xlsx_files


# List to store individual DataFrames
dataframes = []

# Iterate over the files
for file in files:
    print(file)
    if file.endswith(".csv"):
        # Open CSV file using pandas
        df = pd.read_csv(file)
    elif file.endswith(".xlsx"):
        # Open XLSX file using pandas
        df = get_EV_data(file)
    else:
        # Skip unsupported file types
        print('Skipped file: {}'.format(file))
        continue

    # Add a column with the filename information
    filename = os.path.basename(file)
    df['Filename'] = filename

    # Append DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated DataFrame as a CSV file
output_file = os.path.join(ROOT, 'arnau_after.csv')
concatenated_df.to_csv(output_file, index=False)
print("\nDone! >>>>>>>>>>>>>>\n")