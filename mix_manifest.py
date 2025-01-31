import pandas as pd

# List of CSV file names to combine
csv_files = [
    "results/S2S/conmambamamba_S_S2S/7775/dev_clean_2_SD_Ctrl.csv",  # Replace with your actual file names
    "results/S2S/conmambamamba_S_S2S/7775/dev_clean_2_SD_W.csv",
    "results/S2S/conmambamamba_S_S2S/7775/dev_clean_2_SI_Ctrl.csv",
    "results/S2S/conmambamamba_S_S2S/7775/dev_clean_2_SI_W.csv",
    "results/S2S/conmambamamba_S_S2S/7775/dev_clean_2_W_CHAINS.csv"
]

# Combine all specified CSV files into a single DataFrame
combined_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save the combined DataFrame to a new CSV file
output_file = "results/S2S/conmambamamba_S_S2S/7775/dev-clean.csv"
combined_df.to_csv(output_file, index=False)

print(f"All specified files have been combined into {output_file}")

