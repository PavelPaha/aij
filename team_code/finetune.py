import torch
from utils import process_data_sample

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, data_samples, processor):
        self.data_samples = data_samples
        self.processor = processor

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        return process_data_sample(self.data_samples[idx], self.processor)


def fine_tune_model(model, tokenizer, dataset, processor, device="cuda", num_epochs=3, learning_rate=5e-5):
    """
    Function for fine-tuning a multimodal model on a specific dataset.
    Args:
        model: The pre-trained multimodal model.
        tokenizer: The tokenizer for text processing.
        dataset: A dataset to fine-tune the model on.
        processor: A processor function for handling the input modalities (video, image, etc.).
        device: The device to train the model on (e.g., "cuda").
        num_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
    """

    # Prepare the model for training
    model.train()
    model.to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Prepare data loader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0

        for batch in data_loader:
            # Process the batch (video, image, or text inputs)
            inputs = [process_data_sample(sample, processor) for sample in batch]

            # Create input ids and attention masks for the text
            input_ids = torch.cat(
                [tokenizer(sample['instruction'], return_tensors="pt").input_ids for sample in inputs])
            attention_mask = torch.cat(
                [tokenizer(sample['instruction'], return_tensors="pt").attention_mask for sample in inputs])

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, images=None)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Loss after epoch {epoch + 1}: {total_loss / len(data_loader)}")

    torch.save(model.state_dict(), 'model_weights.pth')
    # model.load_state_dict(torch.load('model_weights.pth'))
    print("Fine-tuning complete.")

dataset = VideoDataset(your_data_samples, processor)
model, processor, tokenizer = setup_model_and_tokenizer(device="cuda")
fine_tune_model(model, tokenizer, dataset, processor, device="cuda", num_epochs=3, learning_rate=5e-5)
