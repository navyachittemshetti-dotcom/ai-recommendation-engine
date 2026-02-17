import pandas as pd
import numpy as np

def clean_data():

    # Read CSV file
    data = pd.read_csv("clean_data.csv")

    # Replace invalid IDs with NaN
    data["ProdID"] = data["ProdID"].replace("2147483648", np.nan)
    data["User's ID"] = data["User's ID"].replace("2147483648", np.nan)

    # Drop rows where User ID or ProdID is missing
    data = data.dropna(subset=["User's ID", "ProdID"])

    # Convert to integer
    data["User's ID"] = data["User's ID"].astype("int64")
    data["ProdID"] = data["ProdID"].astype("int64")

    # Convert Review Count to integer
    data["Review Count"] = data["Review Count"].fillna(0)
    data["Review Count"] = data["Review Count"].astype("int64")

    # Fill missing text columns with empty string
    data["Category"] = data["Category"].fillna("")
    data["Brand"] = data["Brand"].fillna("")
    data["Description"] = data["Description"].fillna("")
    data["Tags"] = data["Tags"].fillna("")

    # Clean Image column (take only first URL before '|')
    if "Image" in data.columns:
        data["Image"] = data["Image"].astype(str).apply(lambda x: x.split("|")[0])

    return data


# Call function
cleaned_data = clean_data()

# Save cleaned file (optional)
cleaned_data.to_csv("cleaned_data.csv", index=False)

print("Data cleaned successfully ✅")