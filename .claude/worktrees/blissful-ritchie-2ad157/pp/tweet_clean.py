import os
import json
import re
import pandas as pd

# Function to clean tweet text (remove URLs, $, @mentions, and #hashtags)
def clean_text(text):
    text = re.sub(r"http\S+|https\S+", "", text)  # Remove links
    text = re.sub(r"@\w+", "", text)  # Remove @mentions
    text = re.sub(r"#\w+", "", text)  # Remove #hashtags
    text = text.replace("$", "")  # Remove dollar signs
    return text.strip()

# Function to process a single file
def process_file(file_path, date_str):
    cleaned_data = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                tweet = json.loads(line.strip())  # Parse JSON
                cleaned_text = clean_text(tweet["text"])  # Clean tweet text
                cleaned_data.append([date_str, cleaned_text])  # Append as row
            except json.JSONDecodeError:
                print(f"Skipping file due to JSON error: {file_path}")
            except KeyError:
                print(f"Skipping file due to missing keys: {file_path}")
    
    return cleaned_data

# Main function to process all tweet files
def process_tweet_folder(root_folder, output_folder="processed_tweets"):
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    for company in os.listdir(root_folder):
        company_path = os.path.join(root_folder, company)
        
        if os.path.isdir(company_path):  # Check if it's a folder
            all_data = []
            
            for date_file in os.listdir(company_path):
                file_path = os.path.join(company_path, date_file)
                
                if os.path.isfile(file_path):  # Ensure it's a file
                    date_str = date_file  # Extract date from filename (YYYY-MM-DD)
                    all_data.extend(process_file(file_path, date_str))  # Process file and collect data

            if all_data:  # Only save if there's data
                df = pd.DataFrame(all_data, columns=["Date", "Tweet"])
                output_csv = os.path.join(output_folder, f"{company}.csv")
                df.to_csv(output_csv, index=False, encoding="utf-8")
                print(f"Saved: {output_csv}")

# Run script
process_tweet_folder("stocknet-dataset-master/stocknet-dataset-master/tweet/raw")
