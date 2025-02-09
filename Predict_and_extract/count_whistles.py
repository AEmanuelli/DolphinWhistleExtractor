import csv
import os

def howmanywhistlesinthisfile(csv_file_path):
    """
    Counts the total number of whistle groups in a CSV file, applying the consecutive whistle grouping rules.

    Args:
        csv_file_path (str): The path to the CSV file.

    Returns:
        int: The total number of whistle groups. Returns -1 if there's an error.
    """
    try:
        with open(csv_file_path, 'r', newline='') as file:  # added newline='' to handle potential empty lines
            reader = csv.reader(file)
            next(reader, None)  # Safely skip header row if it exists

            consecutive_whistles = []
            last_end_time = 0
            total_whistle_groups = 0

            for row in reader:
                try:
                    start_time = float(row[1])
                    end_time = float(row[2])

                    is_consecutive = start_time <= last_end_time

                    if is_consecutive:
                        consecutive_whistles.append((start_time, end_time))
                    else:
                        if consecutive_whistles:
                            total_whistle_groups += count_whistle_groups(consecutive_whistles)
                        consecutive_whistles = [(start_time, end_time)]

                    last_end_time = end_time
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping row due to error: {e} - Row data: {row}")


            if consecutive_whistles:  # Process the last group
                total_whistle_groups += count_whistle_groups(consecutive_whistles)
            return total_whistle_groups

    except FileNotFoundError:
        print(f"Error: File not found at {csv_file_path}")
        return -1
    except Exception as e:
        print(f"An error occurred: {e}")
        return -1



def count_whistle_groups(consecutive_whistles):
    """Counts groups of consecutive whistles based on the specified rules."""
    num_whistles = len(consecutive_whistles)
    if num_whistles <= 3:
        return 1
    elif num_whistles == 4:
        return 2
    elif num_whistles == 7:
        return 3
    elif num_whistles == 10:
        return 4
    else:
        # More robust extrapolation for larger groups:
        return (num_whistles + 2) // 3  # Adjust formula as needed


