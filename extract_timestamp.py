import re
import csv
import os
import fnmatch
import time
import chardet


def generate_timestamp_csv(this_dir):
    file_pattern = "*.log"
    # this_dir = "C:/src/fictrac/bin/Release"
    file_match = fnmatch.filter(os.listdir(this_dir), file_pattern)
    file_path = os.path.join(this_dir, file_match[0])
    file_name, file_extension = os.path.splitext(file_path)
    output_file_name = file_name + "_timestamp.csv"
    # Create lists to store the extracted data
    frame_number_list = []
    float_values_list = []

    frame_pattern = r"Trackball::process \[INF\] Frame (\d+)"
    float_pattern = r"BaslerSource::grab \[DBG\] Frame captured \d+x\d+x\d+ @ \d+\.\d+ \((\d+\.\d+)\)"

    # Detect the file's encoding
    with open(file_path, "rb") as rawdata:
        result = chardet.detect(rawdata.read(10000))

    detected_encoding = result["encoding"]

    # Open and read the log file
    with open(file_path, "r", encoding=detected_encoding) as log_file:
        log_content = log_file.read()

    # Use regular expressions to extract frame numbers and float values
    frame_numbers = [int(match) for match in re.findall(frame_pattern, log_content)]
    float_values = [float(value) for value in re.findall(float_pattern, log_content)]

    # Combine the data into a 2D array
    if len(frame_numbers) == len(float_values):
        data = list(zip(frame_numbers, float_values))
    else:
        frame_numbers.pop()
        data = list(zip(frame_numbers, float_values))

    # Save the extracted data to a new CSV file
    with open(output_file_name, "w", newline="") as output_csv:
        csv_writer = csv.writer(output_csv)
        csv_writer.writerow(["Frame Number", "Float Value"])
        csv_writer.writerows(data)

    print(f"Data extracted and saved to {output_file_name}")


if __name__ == "__main__":
    this_dir = "C:/Users/neuroLaptop/Documents/L23038/230824/dots"
    tic = time.perf_counter()
    generate_timestamp_csv(this_dir)
    toc = time.perf_counter()
    print(f"it takes {toc-tic:0.4f} seconds to extract time stamp")
