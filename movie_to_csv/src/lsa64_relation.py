import os
import shutil

lsa64_key_value = {
    "Opaque": 1,
    "Red": 2,
    "Green": 3,
    "Yellow": 4,
    "Bright": 5,
    "Light-blue": 6,
    "Colors": 7,
    "Pink": 8,
    "Women": 9,
    "Enemy": 10,
    "Son": 11,
    "Man": 12,
    "Away": 13,
    "Drawer": 14,
    "Born": 15,
    "Learn": 16,
    "Call": 17,
    "Skimmer": 18,
    "Bitter": 19,
    "Sweet_milk": 20,
    "Milk": 21,
    "Water": 22,
    "Food": 23,
    "Argentina": 24,
    "Uruguay": 25,
    "Country": 26,
    "Last_name": 27,
    "Where": 28,
    "Mock": 29,
    "Birthday": 30,
    "Breakfast": 31,
    "Photo": 32,
    "Hungry": 33,
    "Map": 34,
    "Coin": 35,
    "Music": 36,
    "Ship": 37,
    "None": 38,
    "Name": 39,
    "Patience": 40,
    "Perfume": 41,
    "Deaf": 42,
    "Trap": 43,
    "Rice": 44,
    "Barbecue": 45,
    "Candy": 46,
    "Chewing-gum": 47,
    "Spaghetti": 48,
    "Yogurt": 49,
    "Accept": 50,
    "Thanks": 51,
    "Shut_down": 52,
    "Appear": 53,
    "To_land": 54,
    "Catch": 55,
    "Help": 56,
    "Dance": 57,
    "Bathe": 58,
    "Buy": 59,
    "Copy": 60,
    "Run": 61,
    "Realize": 62,
    "Give": 63,
    "Find": 64,
}

directory_path = "../../data/lsa64/all"
restore_directory_path = "../../data/lsa64_split/"
files = os.listdir(directory_path)
files = sorted(files)
count = 0
for file in files:
    prefix = file[:3]
    key = [k for k, v in lsa64_key_value.items() if str(v).zfill(3) == str(prefix)]
    if len(key) != 1:
        raise ValueError
    target_dir_path = os.path.join(restore_directory_path, key[0])
    if not os.path.exists(target_dir_path):
        os.makedirs(target_dir_path)
    
    file_path = os.path.join(directory_path, file)
    target_file_path = os.path.join(target_dir_path, file)
    shutil.move(file_path, target_file_path)
    count += 1

print(count)
   


