import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel
from mido import MidiFile, MidiTrack, Message
from tqdm import tqdm

# Parameters
note_vocab_size = 128
relative_onset_vocab_size = 100
duration_vocab_size = 100
vocab_size = note_vocab_size * relative_onset_vocab_size * duration_vocab_size
sequence_length = 100
embedding_size = 128
num_heads = 16
num_layers = 4
num_epochs = 100
batch_size = 64
lr = 0.001

folder_path = "C:\\Users\\Marko Pap\\Downloads\\MIDI dataset"
file_names = [f for f in os.listdir(folder_path) if f.endswith(".mid")]
midi_data = []

for file_name in file_names:
    midi_path = os.path.join(folder_path, file_name)
    midi = MidiFile(midi_path)

    events = []
    for track in midi.tracks:
        current_time = 0
        note_on_time = {}  
        for msg in track:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                note = msg.note
                relative_onset = min(current_time, relative_onset_vocab_size - 1)
                note_on_time[note] = current_time
                current_time = 0
            elif (msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)) and msg.note in note_on_time:
                note = msg.note
                duration = min(current_time - note_on_time[note], duration_vocab_size - 1)
                events.append(note + relative_onset * note_vocab_size + duration * note_vocab_size * relative_onset_vocab_size)
                del note_on_time[note]

    midi_data.extend(events[:sequence_length * (len(events) // sequence_length)])

midi_data = torch.tensor(midi_data).view(-1, sequence_length)
dataset = TensorDataset(midi_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

config = GPT2Config(
    vocab_size=vocab_size,
    n_positions=sequence_length,
    n_embd=embedding_size,
    n_layer=num_layers,
    n_head=num_heads,
)

model = GPT2LMHeadModel(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss = 0
    total_batches = len(data_loader)

    for batch in tqdm(data_loader):
        inputs = batch[0][:, :-1].to(device)
        targets = batch[0][:, 1:].to(device)
        outputs = model(inputs).logits
        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / total_batches
    print(f"Training Loss: {avg_loss:.4f}")

model.eval()
generated = [torch.randint(0, vocab_size, (1,))]
temperature = 0.7

for i in range(sequence_length - 1):
    inputs = torch.tensor(generated, dtype=torch.long).to(device)
    outputs = model(inputs).logits
    probs = F.softmax(outputs / temperature, dim=-1)
    next_token = torch.multinomial(probs[-1], 1)
    generated.append(next_token)

midi = MidiFile()
track = MidiTrack()
midi.tracks.append(track)

for token in generated:
    if isinstance(token, torch.Tensor):
        token = token.item()
        note = token % note_vocab_size
        relative_onset = (token // note_vocab_size) % relative_onset_vocab_size
        duration = token // (note_vocab_size * relative_onset_vocab_size)
        track.append(Message("note_on", note=note, velocity=64, time=relative_onset))
        track.append(Message("note_off", note=note, velocity=64, time=duration))

midi.save("generated_midi.mid")
