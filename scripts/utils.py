import json
import numpy as np
from uncertainties import ufloat
'''
def filter_rows(group):
    # Function to avoid duplication of events due to multicandidates
    return group[group['entry']==group['entry'].unique()[0]]

def remove_multicandidates(df):
    """
    Function for removing multicandidates (different candidates with same RUNNUMBER, EVENTNUMBER).
    It can slow quite much the code execution.
    The function replaces 'RUNNUMBER', 'EVENTNUMBER', 'entry' with a new unique event identifier 'even_entry'.
    The entry alone doesn't unqiuely identify different events as it started again from 0 when reading a new ROOT file.
    """
    print("Removing multicandidates")
    df_grouped = df.groupby(['RUNNUMBER', 'EVENTNUMBER'])
    df = df_grouped.apply(filter_rows).drop(columns = ['RUNNUMBER', 'EVENTNUMBER']) #Drop multicandidates
    df.reset_index(inplace=True)
    df['event_entry'] = df.groupby(['RUNNUMBER', 'EVENTNUMBER']).ngroup() # in the concatenation the entries are the same among different files, needed to look at evt and run number to identify them
    df.drop(columns=['entry', 'level_2'], inplace=True)
    return df
'''
def remove_multicandidates(test_df):
    """
    Function for removing multicandidates (different candidates with same RUNNUMBER, EVENTNUMBER but different entry).
    It can slow quite much the code execution.
    The function replaces 'RUNNUMBER', 'EVENTNUMBER', 'entry' with a new unique event identifier 'even_entry'.
    The entry alone doesn't unqiuely identify different events as it started again from 0 when reading a new ROOT file.
    """
    print("Removing multicandidates")
    # entry_list = test_df.drop_duplicates(subset=['RUNNUMBER', 'EVENTNUMBER'], keep='first').index.tolist()
    # filtered_df = test_df[test_df.index.isin(entry_list)]
    # filtered_df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
    # filtered_df.rename(columns={'entry': 'event_entry'}, inplace=True)
    test_df = test_df.drop_duplicates(subset=['RUNNUMBER', 'EVENTNUMBER'], keep='first')
    test_df.reset_index(inplace=True)
    test_df['event_entry'] = test_df.index
    return test_df

def format_and_propagate(values):
    values = np.array(values) * 100  # Multiply all values by 100

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return ufloat(np.nan, np.nan)

    if len(values) > 2:  # For cases with multiple errors
        combined_error = np.sqrt(np.sum(np.square(values[1:])))
        if np.isnan(combined_error) or np.isinf(combined_error):
            return ufloat(np.nan, np.nan)
        return ufloat(values[0], combined_error)
    else:
        return ufloat(values[0], values[1])

def load_and_process_json(json_file): 

    with open(json_file, 'r') as f:
        data = json.load(f)
    processed_data = {}
    for key, values in data.items():
        # Ensure all values are numeric and convert to float
        numeric_values = [float(x) for x in values if isinstance(x, (int, float))]
        if len(numeric_values) != len(values):
            print(f"Non-numeric data found in {key}. Replacing with NaN.")
            processed_data[key] = ufloat(np.nan, np.nan)
            continue # skip to the next iteration in the loop

        if len(numeric_values) < 2:
            print(f"Not enough data to create ufloat for {key}.")
            processed_data[key] = ufloat(np.nan, np.nan)
            continue
        
        processed_data[key] = format_and_propagate(numeric_values)
    
    return processed_data

# Custom JSON encoder for ufloat objects
class UFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ufloat):
            return {
                "nominal_value": obj.nominal_value,
                "std_dev": obj.std_dev
            }
        return super(UFloatEncoder, self).default(obj)