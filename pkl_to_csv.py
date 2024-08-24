import pickle
import csv

def convert_pkl_to_csv(pkl_file_path, csv_file_path):
    try:
        with open(pkl_file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        
        if isinstance(data, (list, tuple)):
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                if all(isinstance(i, (list, tuple)) for i in data):
                    csv_writer.writerows(data)
                else:
                    for item in data:
                        csv_writer.writerow([item])
        
        elif isinstance(data, dict):
            with open(csv_file_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                for key, value in data.items():
                    csv_writer.writerow([key, value])
        
        else:
            print("Unsupported data format for CSV conversion.")
        
        print(f"Data from {pkl_file_path} has been successfully converted to {csv_file_path}.")
    
    except FileNotFoundError:
        print(f"The file {pkl_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

pkl_file_path = 'model_params.pkl'
csv_file_path = 'model_params.csv'

convert_pkl_to_csv(pkl_file_path, csv_file_path)
