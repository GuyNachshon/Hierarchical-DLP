from openai import OpenAI

client = OpenAI(api_key="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")

results = []

# List batches, optionally with pagination
batches = client.batches.list(limit=100)  # Adjust limit as needed
for batch in batches.data:
    print(f"Batch ID: {batch.id}, Status: {batch.status}")

    # Further processing to get output file ID
    # Assuming 'batch' is a completed batch object from the listing step
    # check if batch is expired
    if batch.status == 'expired':
        print(f"Batch {batch.id} has expired. Skipping retrieval of output file.")
        continue
    if batch.status == 'completed' and batch.output_file_id:
        try:
            output_file_content = client.files.content(batch.output_file_id).text
        except Exception as e:
            print(f"Error retrieving output file for batch {batch.id}: {e}")
            continue
        print(f"Content of batch {batch.id}:")
        print(output_file_content) # Decode if it's bytes
        results.append(output_file_content)


print(f"Retrieved {len(results)} batch outputs.")
with open('batch_outputs.txt', 'w+') as f:
    for output in results:
        f.write(output + "\n")
