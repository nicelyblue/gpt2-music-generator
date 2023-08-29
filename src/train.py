import os
import json
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm
from midi_tokenizer import MidiTokenizer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Music Generation with Transformers')
    parser.add_argument('model', type=str, help='Path to the model configuration')
    parser.add_argument('--data', default='dataset', type=str, help='Path to MIDI dataset folder')
    parser.add_argument('--save_directory', default='model', type=str, help='Directory to save the model')
    parser.add_argument('--num_epochs', default=2000, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--learn_rate', default=0.01, type=float, help='Initial learning rate')
    args = parser.parse_args()
    return args

def load_model_config(config_path):
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    return config_data

def save_model_config(config, save_path):
    with open(os.path.join(save_path,'model.json'), "w") as json_file:
        json.dump(config, json_file)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_arguments()
    config = load_model_config(args.model)

    tokenizer = MidiTokenizer(args.data, config['sequence_length'])
    tokenizer.process_midi_files()
    tokens = tokenizer.tokens
    vocab_size = len(tokenizer.vocab)

    train_size = int(0.9 * len(tokens))
    train_tokens = tokens[:train_size]
    valid_tokens = tokens[train_size:]

    train_tokens = train_tokens[:len(train_tokens) // config['sequence_length'] * config['sequence_length']]
    valid_tokens = valid_tokens[:len(valid_tokens) // config['sequence_length'] * config['sequence_length']]

    train_dataset = TensorDataset(torch.tensor(train_tokens).view(-1, config['sequence_length']))
    valid_dataset = TensorDataset(torch.tensor(valid_tokens).view(-1, config['sequence_length']))

    train_dataset = TensorDataset(torch.tensor(train_tokens).view(-1, config['sequence_length']))
    valid_dataset = TensorDataset(torch.tensor(valid_tokens).view(-1, config['sequence_length']))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    config = GPT2Config(
        vocab_size=vocab_size,
        **config
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=1e-5)

    patience = 5
    early_stopping_counter = 0
    best_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader)):
            inputs = batch[0][:, :-1].to(device)
            targets = batch[0][:, 1:].to(device)
            outputs = model(inputs).logits
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                inputs = batch[0][:, :-1].to(device)
                targets = batch[0][:, 1:].to(device)
                outputs = model(inputs).logits
                loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
                total_valid_loss += loss.item()

        avg_valid_loss = total_valid_loss / len(valid_loader)
        print(f"Validation Loss: {avg_valid_loss:.4f}")

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            if not os.path.exists(args.save_directory):
                os.mkdir(args.save_directory)
            torch.save(model.state_dict(), os.path.join(args.save_directory))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

        scheduler.step(avg_valid_loss)

    model.eval()
    tokenizer.save(os.path.join(args.save_directory, 'tokenizer.pkl'))

if __name__ == '__main__':
    main()