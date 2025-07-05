import os
import shutil
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def organize_folders():
    # Putanja do početnog direktorija
    base_path = r"C:\\BagOfLies\\BagOfLies\\Finalised"
    # Putanja gdje će svi folderi biti kopirani
    destination_path = r"C:\\OrganizedFolders"

    # Kreiranje ciljnog direktorija ako ne postoji
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    # Dobijanje liste svih mapa u baznom direktoriju (sortirano prirodnim redoslijedom)
    main_folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))], key=natural_sort_key)

    print("Redoslijed glavnih mapa:")

    for i, folder in enumerate(main_folders):
        print(f"{i}: {folder}")

    for main_index, folder in enumerate(main_folders):
        folder_path = os.path.join(base_path, folder)
        # Podmape sortirane prirodnim redoslijedom
        sub_folders = sorted([sf for sf in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, sf))], key=natural_sort_key)

        for j, sub_folder in enumerate(sub_folders):
            print(f"    {j}: {sub_folder}")

        for sub_index, sub_folder in enumerate(sub_folders):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            new_folder_name = f"User{main_index}Run{sub_index}"
            new_folder_path = os.path.join(destination_path, new_folder_name)

            # Kopiranje foldera na novu lokaciju
            shutil.copytree(sub_folder_path, new_folder_path)

    print("Kopiranje foldera završeno.")

if __name__ == "__main__":
    organize_folders()
