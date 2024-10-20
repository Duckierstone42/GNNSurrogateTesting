import pandas as pd
from codec import Codec

data_path = "my_test_folder/compiled_data_best_epoch.csv"

df = pd.read_csv(data_path)

valid_genomes = []
invalid_genomes = []
invalid_genome_indices = []

num_classes = 7
num_loss_components = 7
codec = Codec(num_classes=num_classes)
counter = 0
for idx, row in df.iterrows():
    genome = row['genome']
    genome_hash = row['individual_hash']
    print("Checking: " + genome_hash)
    counter += 1
    print("Counter: " + str(counter))

    try:
        model_dict = codec.decode_genome(genome, num_loss_components)
        model = model_dict['model']

        valid_genomes.append((genome_hash, genome))
    except Exception as e:
        print(f"Invalid genome: {genome_hash}, Error: {e}")
        invalid_genomes.append((genome_hash, genome))
        invalid_genome_indices.append(idx)

print(f"Number of valid genomes: {len(valid_genomes)}")
print(f"Number of invalid genomes: {len(invalid_genomes)}")

pd.DataFrame(valid_genomes, columns=['hash', 'genome']).to_csv("my_test_folder/valid_genomes.csv", index=False)
pd.DataFrame(invalid_genomes, columns=['hash', 'genome']).to_csv("my_test_folder/invalid_genomes.csv", index=False)

df_valid = df.drop(invalid_genome_indices)

pd.DataFrame(valid_genomes, columns=['hash', 'genome']).to_csv("my_test_folder/valid_genomes.csv", index=False)
df.iloc[invalid_genome_indices].to_csv("my_test_folder/invalid_genomes.csv", index=False)

df_valid.to_csv("my_test_folder/compiled_data_valid_only.csv", index=False)