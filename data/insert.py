import json
import os

# Function to interactively get tag, patterns, and a single response from the user
def get_intent_from_user():
    tag = input("Enter a tag: ").strip()
    patterns = input("Enter patterns (comma-separated): ").split(',')
    response = input("Enter a response: ").strip()
    return {"tag": tag, "patterns": [pattern.strip() for pattern in patterns], "response": response}

# Function to append a new intent to the JSON file
def append_to_json_file(data, filename='intents.json'):
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        with open(filename, 'w') as file:
            json.dump({"intents": []}, file, indent=4)

    try:
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    except json.JSONDecodeError:
        existing_data = {"intents": []}

    # Add an "id" to the data
    data["id"] = len(existing_data["intents"]) + 1

    existing_data["intents"].append(data)

    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

# Main script
while True:
    intent_data = get_intent_from_user()

    append_to_json_file(intent_data)
    print(f"Intent added to the dataset with id {intent_data['id']}.\n")

    continue_input = input("Do you want to add another intent? (yes/no): ").strip().lower()
    if continue_input != 'yes':
        print("Exiting.")
        break

print("Script completed.")
