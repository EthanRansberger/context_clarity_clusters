import json
import os
import sys

def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Error: {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
    return {}

def save_json(data, file_path):
    """Save data to a JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

def sanitize_filename(filename):
    """Sanitize a string to be used as a valid filename."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()

def extract_conversations_by_title(data, output_dir='conversations'):
    """Extract conversations from JSON data and save them by title."""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        conversations = []

        if isinstance(data, list):  # Handle list of conversations
            conversations = data
        elif isinstance(data, dict):  # Handle single conversation
            conversations.append(data)
        
        for conversation in conversations:
            title = sanitize_filename(conversation.get('title', 'untitled'))
            file_path = os.path.join(output_dir, f"{title}.json")
            save_json(conversation, file_path)
    except Exception as e:
        print(f"Error extracting conversations by title: {str(e)}")

def extract_user_authored_content(data):
    """Extract user-authored content from JSON data."""
    try:
        user_content = []
        if isinstance(data, list):  # If the data is a list of conversations
            for conversation in data:
                user_content.extend(_extract_user_content_from_mapping(conversation))
        elif isinstance(data, dict):  # If the data is a single conversation
            user_content.extend(_extract_user_content_from_mapping(data))

        return user_content
    except Exception as e:
        print(f"Error extracting user-authored content: {str(e)}")
        return []

def _extract_user_content_from_mapping(conversation):
    """Helper function to extract user-authored content from conversation mapping."""
    user_content = []
    if 'mapping' in conversation:
        for message_id, message_data in conversation['mapping'].items():
            if 'message' in message_data and 'author' in message_data['message']:
                if message_data['message']['author']['role'] == 'user':
                    parts = message_data['message']['content'].get('parts', [])
                    user_content.extend(parts)
    return user_content

def save_user_generated_content(folder_path, output_file):
    """Extract and save user-generated content from all JSON files in a folder."""
    all_user_messages = []
    file_count = 0
    message_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            data = load_json(file_path)
            if data:
                user_messages = extract_user_authored_content(data)
                all_user_messages.extend(user_messages)
                file_count += 1
                message_count += len(user_messages)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as output:
            for content in all_user_messages:
                output.write(content + '\n')
        print(f"Processed {file_count} JSON files.")
        print(f"Extracted {message_count} user messages.")
    except Exception as e:
        print(f"Error writing to output file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python json_utils.py <folder_path> <output_file>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2]
    save_user_generated_content(folder_path, output_file)
