import struct
import csv

def read_and_convert_bin_to_csv(bin_file_path, csv_file_path):
    try:
        # Open the binary file for reading
        with open(bin_file_path, 'rb') as bin_file:
            # Assuming the data is stored as floats (4 bytes each)
            # Adjust the format ('f') based on the actual data type in the file
            bin_data = bin_file.read()
            num_floats = len(bin_data) // 4  # Each float is 4 bytes
            
            # Unpack binary data into a tuple of floats
            float_data = struct.unpack(f'{num_floats}f', bin_data)

        # Write the unpacked data to a CSV file
        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Writing the float values as rows
            for value in float_data:
                csv_writer.writerow([value])
        
        print(f"Data from {bin_file_path} has been successfully converted to {csv_file_path}.")
    
    except FileNotFoundError:
        print(f"The file {bin_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with the path to your binary and CSV file
bin_file_path = 'model_params.bin'
csv_file_path = 'model_params.csv'

read_and_convert_bin_to_csv(bin_file_path, csv_file_path)
