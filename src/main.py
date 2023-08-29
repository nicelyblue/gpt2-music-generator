import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm

# Parameters
sequence_length = 20
embedding_size = 4
num_heads = 1
num_layers = 1
num_epochs = 2000
batch_size = 4
lr = 0.01

folder_path = "MIDI dataset"
training = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_names = [f for f in os.listdir(folder_path) if f.endswith(".mid")]

vocab = {}
counter = 0
tokens = []

def find_nearest_known_token(oov_token):
    return min(vocab.values(), key=lambda k: abs(k - oov_token))

def handle_oov_token(oov_token):
    return find_nearest_known_token(oov_token)

def get_token(note, velocity, duration):
    global counter
    token_key = (note, velocity, duration)
    if token_key not in vocab:
        vocab[token_key] = counter
        counter += 1
    return vocab[token_key]

tokens = []

for file_name in file_names:
    midi_path = os.path.join(folder_path, file_name)
    midi = MidiFile(midi_path)

    for track in midi.tracks:
        current_time = 0
        note_on_time = {}
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note_on_time[msg.note] = (current_time, msg.velocity)
            elif (msg.type == "note_on" and msg.velocity == 0) and msg.note in note_on_time:
                note, velocity = msg.note, note_on_time[msg.note][1]
                duration = current_time - note_on_time[msg.note][0]
                tokens.append(get_token(note, velocity, duration))
                del note_on_time[msg.note]

num_full_sequences = len(tokens) // sequence_length
trimmed_length = num_full_sequences * sequence_length
tokens = tokens[:trimmed_length]

vocab_size = len(vocab)

save_directory = "model"
if os.path.exists(save_directory):
    model = GPT2LMHeadModel.from_pretrained(save_directory).to(device)
    training = False

# Early stopping parameters
patience = 5  # Number of epochs with no improvement after which training will be stopped
early_stopping_counter = 0
best_loss = float('inf')

if training:
    # Split tokens into training and validation sets (e.g., 90% train, 10% validation)
    train_size = int(0.9 * len(tokens))
    train_tokens = tokens[:train_size]
    valid_tokens = tokens[train_size:]

    train_tokens = train_tokens[:len(train_tokens) // sequence_length * sequence_length]
    valid_tokens = valid_tokens[:len(valid_tokens) // sequence_length * sequence_length]

    train_dataset = TensorDataset(torch.tensor(train_tokens).view(-1, sequence_length))
    valid_dataset = TensorDataset(torch.tensor(valid_tokens).view(-1, sequence_length))

    train_dataset = TensorDataset(torch.tensor(train_tokens).view(-1, sequence_length))
    valid_dataset = TensorDataset(torch.tensor(valid_tokens).view(-1, sequence_length))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=sequence_length,
        n_embd=embedding_size,
        n_layer=num_layers,
        n_head=num_heads,
        resid_pdrop=0.2,
        embd_pdrop=0.2
    )

    model = GPT2LMHeadModel(config)
    model.to(device)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
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

        # Validation loop
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

        # Early stopping logic
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            if not os.path.exists(save_directory):
                os.mkdir(save_directory)
            torch.save(model.state_dict(), os.path.join(save_directory))
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping")
                break

        # Update learning rate based on validation loss
        scheduler.step(avg_valid_loss)

    model.eval()

# Load the best model for generation
model.load_state_dict(torch.load(save_directory))
model.eval()

generated = [torch.randint(0, vocab_size, (1,)).item()]
temperature = 0.9

for i in range(sequence_length):
    inputs = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(device)
    outputs = model(inputs).logits
    probs = F.softmax(outputs / temperature, dim=-1)
    next_token = torch.multinomial(probs[-1], 1)[0].item()

    if next_token not in vocab.values():
        next_token = handle_oov_token(next_token)

    generated.append(next_token)

midi = MidiFile()
track = MidiTrack()
midi.tracks.append(track)

for token in generated:
    note, velocity, duration = [k for k, v in vocab.items() if v == token][0]
    track.append(Message("note_on", note=note, velocity=velocity, time=0))
    track.append(Message("note_on", note=note, velocity=0, time=duration))

midi.save('generated_midi.mid')