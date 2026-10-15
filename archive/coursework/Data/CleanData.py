import os

def rename_images(base_path):
    if not os.path.exists(base_path):
        print(f"Error: Base path '{base_path}' does not exist")
        return

    for set_type in ['test', 'train']:
        set_path = os.path.join(base_path, set_type)
        if not os.path.exists(set_path):
            print(f"Warning: {set_type} directory not found at {set_path}")
            continue

        for emotion in os.listdir(set_path):
            emotion_path = os.path.join(set_path, emotion)
            if not os.path.isdir(emotion_path):
                continue

            try:
                files = sorted(os.listdir(emotion_path))
                for index, filename in enumerate(files):
                    # Skip files that are already in the correct format
                    if filename.startswith(f"{set_type.capitalize()}_{emotion}_"):
                        continue

                    ext = os.path.splitext(filename)[-1].lower()
                    new_name = f"{set_type.capitalize()}_{emotion}_{index}{ext}"
                    src = os.path.join(emotion_path, filename)
                    dst = os.path.join(emotion_path, new_name)

                    try:
                        if os.path.exists(dst):
                            print(f"Warning: Skipping {filename} as {new_name} already exists")
                            continue
                        os.rename(src, dst)
                    except OSError as e:
                        print(f"Error renaming {filename}: {str(e)}")

            except OSError as e:
                print(f"Error processing directory {emotion_path}: {str(e)}")

    print("✅ All images have been renamed.")

# 🛠 Example usage
rename_images("C:\\Users\\user\\Sourcetree\\Github\\cs471Final\\CK+")