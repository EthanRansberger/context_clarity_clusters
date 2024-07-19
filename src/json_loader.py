import json
import os
import re

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def sanitize_filename(filename):
    """
    Sanitize a string to be used as a filename by removing or replacing invalid characters.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: The sanitized filename.
    """
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def extract_conversations_by_title(data, output_dir='conversations'):
    """
    Extract individual conversations by title and save them as separate JSON files.

    Args:
        data (list): The JSON data.
        output_dir (str): The directory to save individual conversations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    title_counts = {}
    for record in data:
        title = record.get('title', 'untitled_conversation')
        sanitized_title = sanitize_filename(title)
        if sanitized_title in title_counts:
            title_counts[sanitized_title] += 1
            sanitized_title = f"{sanitized_title}_{title_counts[sanitized_title]}"
        else:
            title_counts[sanitized_title] = 0
        
        file_name = f"{sanitized_title}.json"
        file_path = os.path.join(output_dir, file_name)
        save_json(record, file_path)
        print(f"Extracted conversation: {title} -> {file_path}")

def extract_user_authored_content(data):
    """
    Extract user-authored content from JSON data.

    Args:
        data (list): The JSON data.

    Returns:
        list: A list of user-authored content strings.
    """
    user_authored_content = []
    if not isinstance(data, list):
        return user_authored_content

    for record in data:
        mapping = record.get('mapping', {})
        if not isinstance(mapping, dict):
            continue

        for key, value in mapping.items():
            if not isinstance(value, dict):
                continue
            message = value.get('message')
            if not isinstance(message, dict):
                continue
            author = message.get('author')
            if not isinstance(author, dict) or author.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, dict):
                continue
            parts = content.get('parts')
            if isinstance(parts, list) and parts and isinstance(parts[0], str):
                user_authored_content.append(parts[0])
    return user_authored_content
