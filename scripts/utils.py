
'''
def filter_rows(group):
   
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
    entry_list = test_df.drop_duplicates(subset=['RUNNUMBER', 'EVENTNUMBER'], keep='first')['entry'].tolist()
    filtered_df = test_df[test_df['entry'].isin(entry_list)]
    filtered_df.drop(columns=['RUNNUMBER', 'EVENTNUMBER'], inplace=True)
    filtered_df.rename(columns={'entry': 'event_entry'}, inplace=True)
    test_df = filtered_df
    return test_df

import json
import numpy as np
from uncertainties import ufloat

def format_and_propagate(values, is_percentage=True):
    if is_percentage:
        values = np.array(values) * 100  # Multiply all percentage values by 100

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return ufloat(np.nan, np.nan)

    if len(values) > 2:  # For cases with multiple errors
        combined_error = np.sqrt(np.sum(np.square(values[1:])))
        if np.isnan(combined_error) or np.isinf(combined_error):
            return ufloat(np.nan, np.nan)
        return ufloat(values[0], combined_error)
    else:
        return ufloat(values[0], values[1])

def extract_taggingInfo(json_file, taggerName):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Navigate to the "calibrated" -> "selected" section
    try:
        selected_data = data[f'{taggerName}']['calibrated']['selected']
    except KeyError as e:
        print(f"Key error: {e}. Could not find the required data.")
        return None

    processed_data = {}

    # Fields we are interested in
    fields_of_interest = ['tag_efficiency', 'mistag_rate', 'effective_mistag', 'tagging_power']

    for field in fields_of_interest:
        values = selected_data.get(field)
        if not values:
            print(f"{field} not found or empty.")
            processed_data[field] = ufloat(np.nan, np.nan)
            continue

        # Ensure all values are numeric and convert to float
        numeric_values = [float(x) for x in values if isinstance(x, (int, float))]
        if len(numeric_values) != len(values):
            print(f"Non-numeric data found in {field}. Replacing with NaN.")
            processed_data[field] = ufloat(np.nan, np.nan)
            continue  # skip to the next iteration in the loop

        if len(numeric_values) < 2:
            print(f"Not enough data to create ufloat for {field}.")
            processed_data[field] = ufloat(np.nan, np.nan)
            continue

        # Apply the format and propagate function
        formatted_value = "{:.1u}".format(format_and_propagate(numeric_values))
        processed_data[field] = formatted_value
        print(f'{field}: {formatted_value}')
    return processed_data

def find_tree_name(decay):
    if decay == 'Bs2JpsiPhi':
        return 'BsToJpsiPhi_Detached/DecayTree'
    if decay == 'Bs2DsPi':
        return 'Hlt2B2OC_BdToDsmPi_DsmToKpKmPim/DecayTree' # For file of type root://eoslhcb.cern.ch//eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00229398/0000/00229398_00000001_1.mc.root
    if decay == 'Bu2JpsiK':
        return 'BuToJpsiKplus_JpsiToMuMu_Detached/DecayTree'
    if decay == 'Bd2JpsiKst':
        return 'BdToJpsiKstar_JpsiToMuMu_Detached'
    else:
        return 'Tuple/DecayTree'


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
        
        processed_data[key] = format_and_propagate(numeric_values, 'Fitpar_' not in key)  
    
    return processed_data
'''
# Custom JSON encoder for ufloat objects
class UFloatEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ufloat):
            return {
                "nominal_value": obj.nominal_value,
                "std_dev": obj.std_dev
            }
        return super(UFloatEncoder, self).default(obj)
'''