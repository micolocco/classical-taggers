import numpy as np
# Function to propagate and round the errors and values
def propagate_and_round(values):
    values = np.array(values) * 100  # Multiply all values by 100

    if np.any(np.isnan(values)) or np.any(np.isinf(values)):
        return [np.nan, np.nan]

    if len(values) > 2:  # For TaggingPower_Cali and EffectiveMistag_Cali
        combined_error = np.sqrt(np.sum(np.square(values[1:])))
        if np.isnan(combined_error) or np.isinf(combined_error):
            return [np.nan, np.nan]
        
        rounded_error = round(combined_error, -int(np.floor(np.log10(combined_error))))
        significant_digit = int(np.floor(np.log10(rounded_error)))
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value, rounded_error]
    else:  # For other data
        max_error = max(values[1:])
        if np.isnan(max_error) or np.isinf(max_error):
            return [np.nan, np.nan]
        
        if max_error == 0:
            # If the maximum error is zero, no need to round it further
            significant_digit = 0
        else:
            significant_digit = int(np.floor(np.log10(max_error)))
        
        rounded_errors = [round(err, -significant_digit) if err != 0 else 0 for err in values[1:]]
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value] + rounded_errors