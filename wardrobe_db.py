# wardrobe_db.py
import json
from datetime import datetime

# Path to the persistent storage file
WARDROBE_FILE = 'mira_wardrobe.json'

def load_wardrobe():
    """Loads the entire virtual wardrobe from the JSON file."""
    try:
        with open(WARDROBE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Wardrobe file not found: {WARDROBE_FILE}. Starting fresh.")
        return []
    except json.JSONDecodeError:
        print("Wardrobe file is corrupted. Starting fresh.")
        return []

def save_wardrobe(wardrobe_list):
    """Saves the current state of the virtual wardrobe to the JSON file."""
    with open(WARDROBE_FILE, 'w') as f:
        json.dump(wardrobe_list, f, indent=4)
    print(f"Wardrobe saved with {len(wardrobe_list)} items.")

def add_item_to_wardrobe(item_data):
    """
    Adds a new item to the wardrobe.
    item_data should be a dict: {'label': 't-shirt', 'confidence': 0.95, 'color': 'red', ...}
    """
    wardrobe = load_wardrobe()
    # Add a timestamp for easy tracking/management
    item_data['added_on'] = datetime.now().isoformat()
    wardrobe.append(item_data)
    save_wardrobe(wardrobe)
    
def get_wardrobe_summary():
    """Returns a simple text summary of the current wardrobe for the AI."""
    wardrobe = load_wardrobe()
    if not wardrobe:
        return "The virtual wardrobe is currently empty."
        
    summary = "Current virtual wardrobe items:\n"
    # Group items by label for a cleaner summary
    item_counts = {}
    for item in wardrobe:
        label = item.get('label', 'unknown item')
        item_counts[label] = item_counts.get(label, 0) + 1
        
    for label, count in item_counts.items():
        summary += f"- {count} x {label}\n"
        
    return summary.strip()