from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
print(f"Loading {model_name}...")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with optimizations for 3B model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",          # Automatic device placement
    trust_remote_code=True
)

print(f"Model loaded successfully!")
print(f"Device: {next(model.parameters()).device}")


test_prompt="""
Please Analyze the following position and provide your best move.

Position (FEN): 8/1p4pk/3R1b1p/7r/8/5KP1/4N3/8 w - -
Turn: White
Legal moves to chosse from: [Ke3, Ke4, Kf2, Kf4, Kg2, Kg4, Nc1, Nc3, Nd4, Nf4, Ng1, Ra6, Rb6, Rc6, Rd1, Rd2, Rd3, Rd4, Rd5, Rd7, Rd8, Re6, Rxf6, g4]

CURRENT BOARD:
. . . . . . . .
. p . . . . p k
. . . R . b . p
. . . . . . . r
. . . . . . . .
. . . . . K P .
. . . . N . . .
. . . . . . . .

Your response MUST follow this EXACT format, without any extra text, PGN, or game history. 

Example Format of your response:

Best move: [The single best move in Standard Algebraic Notation, e.g., Ra7]
<reasoning>
[Explain the strategic and tactical reasons for your move. Address the opponent's threats and your own opportunities.]
</reasoning>
END OF RESPONSE


Your response for the given position:
"""


# FIX: Move inputs to the same device as model
inputs = tokenizer(test_prompt, return_tensors="pt")
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}


input_token_count = inputs['input_ids'].shape[1]

print(f"Inputs moved to device: {device}")

with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=200,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )


generated_tokens = outputs[0][input_token_count:]
generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print("\\nGenerated Response:")
print(generated_response)



# Show memory usage
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\\nGPU Memory: {memory_used:.1f}GB / {memory_total:.1f}GB used")
