import pandas as pd

# Read original dataset
df = pd.read_csv("data/creditcard.csv")

# Create a random sample
sample = df.sample(n=20000, random_state=42)

# Save sample
sample.to_csv("data/creditcard_sample.csv", index=False)

print("Done!")
print(sample.shape)