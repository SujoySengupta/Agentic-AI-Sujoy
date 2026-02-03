import os
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    TextDataset
)
# import data.txt

# ==========================================
# 1. SETUP & HARDWARE
# ==========================================
# Your RTX 4060 is perfect for this.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 2. TRAIN A TINY TOKENIZER
# ==========================================
# CRITICAL STEP: Standard tokenizers have vocab_size=50k.
# 50,000 * 64 hidden_dim = 3.2 Million params (FAIL).
# We must limit vocab to ~2,000 to stay under 2M total.

script_dir = os.path.dirname(os.path.abspath(__file__))
files = os.path.join(script_dir, "data.txt")

# Sanity Check
if not os.path.exists(files):
    print(f"❌ ERROR: Cannot find file at: {files}")
    print("Make sure the file is named 'data.txt' and is in the same folder as this script.")
    exit()
else:
    print(f"✅ Found data file at: {files}")
# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=files, vocab_size=2048, min_frequency=2, special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>"
])

# Save the tokenizer to load it with Transformers later
os.makedirs("nano-tokenizer", exist_ok=True)
tokenizer.save_model("nano-tokenizer")

# Load it back as a Transformers-compatible tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("nano-tokenizer")
tokenizer.add_special_tokens({'eos_token': '</s>', 'bos_token': '<s>', 'unk_token': '<unk>', 'pad_token': '<pad>'})

# ==========================================
# 3. DEFINE THE NANO-MODEL (< 2M Params)
# ==========================================
config = GPT2Config(
    vocab_size=2048,   # Must match our tiny tokenizer
    n_positions=512,   # Context window (keep small for efficiency)
    n_embd=128,        # Embedding dimension (Tiny!)
    n_layer=4,         # Only 4 transformer blocks
    n_head=4,          # 4 attention heads
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
model.to(device)

# Count parameters to prove it to your professor
num_params = sum(p.numel() for p in model.parameters())
print(f"🔥 TOTAL PARAMETERS: {num_params:,}")

if num_params > 2000000:
    print("WARNING: Model is too big! Reduce n_embd or vocab_size.")
else:
    print("SUCCESS: Model is under 2 Million parameters.")

# ==========================================
# 4. PREPARE DATASET
# ==========================================
# This effectively tokenizes your text file for training
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=files,
    block_size=128,  # Max sequence length
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ==========================================
# 5. TRAIN
# ==========================================
training_args = TrainingArguments(
    output_dir="./nano-gpt-output",
    overwrite_output_dir=True,
    num_train_epochs=50,       # It's tiny, so we need many epochs to learn
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=1e-3,        # Higher LR for smaller models often helps
    logging_steps=10,
    report_to="none",          # Disable wandb/mlflow for simplicity
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

print("Starting training...")
trainer.train()

# ==========================================
# 6. SAVE & TEST
# ==========================================
trainer.save_model("./final-nano-model")
tokenizer.save_pretrained("./final-nano-model")

# Inference Test
input_text = "The agent decided to"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

output = model.generate(
    **inputs, 
    max_length=50, 
    num_return_sequences=1, 
    temperature=0.7,
    do_sample=True
)

print("\nGenerated Text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))