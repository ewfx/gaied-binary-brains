import json

# Load categories from JSON file
def load_categories_file(json_file):
    with open(json_file, "r") as file:
        return json.load(file)
    
# Function to read categories from JSON file
def load_categories():
    categories_data = load_categories_file("../../config/categories.json")
    formatted_categories = "\n".join([f"- {cat}: {', '.join(subs)}" for cat, subs in categories_data.items()])
    return formatted_categories

categories=load_categories()
print(categories)