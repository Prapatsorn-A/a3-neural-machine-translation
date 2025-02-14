from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from model import Seq2SeqTransformer, Encoder, Decoder

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'models/Seq2SeqTransformer-general.pt'

# Load vocab_transform and token_transform
vocab_transform = torch.load('vocab_transform.pt')
token_transform = torch.load('token_transform.pt')

SRC_LANGUAGE = 'input_text'
TRG_LANGUAGE = 'translated_text'
SRC_PAD_IDX = vocab_transform[SRC_LANGUAGE]['<pad>']
TRG_PAD_IDX = vocab_transform[TRG_LANGUAGE]['<pad>']

input_dim = len(vocab_transform[SRC_LANGUAGE])
output_dim = len(vocab_transform[TRG_LANGUAGE])
hid_dim = 256
enc_layers = 3
dec_layers = 3
enc_heads = 8
dec_heads = 8
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.1
dec_dropout = 0.1

enc = Encoder(input_dim, hid_dim, enc_layers, enc_heads, enc_pf_dim, enc_dropout, device)
dec = Decoder(output_dim, hid_dim, dec_layers, dec_heads, dec_pf_dim, enc_dropout, device)
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define a function for translation
def translate(input_text, top_k=10):
    try:
        print(f"Translating input: {input_text}")
        
        # Tokenize and numericalize input text
        tokenized_input = token_transform[SRC_LANGUAGE](input_text)
        print(f"Tokenized input: {tokenized_input}")
        numericalized_input = [vocab_transform[SRC_LANGUAGE][word] for word in tokenized_input]
        tensor_input = torch.tensor(numericalized_input).unsqueeze(0).to(device)
        
        # Initialize the decoder input (start-of-sequence token, usually <sos>)
        start_token = vocab_transform[SRC_LANGUAGE].get_stoi().get('<sos>', None)
        if start_token is None:
            start_token = 0  # Handle case where no <sos> token is available
        decoder_input = torch.tensor([start_token]).unsqueeze(0).to(device)
        
        # Generate the translated output sequence
        output_sequence = []
        max_output_length = 50  # Set a maximum length for the output (avoid infinite loops)

        for _ in range(max_output_length):  # Max length of output sequence
            output = model(tensor_input, decoder_input)
            
            # Check if the output is a tuple and get the first element (output tensor)
            if isinstance(output, tuple):
                output = output[0]  # Extract the output tensor (assuming it's the first element)
            
            # Apply top-k sampling to avoid repeated tokens
            logits = output[0, -1]  # Get the logits for the last token
            top_k_values, top_k_indices = torch.topk(logits, top_k)
            
            # Convert to probabilities
            probabilities = F.softmax(top_k_values, dim=-1)
            
            # Sample from the top-k tokens
            next_token_idx = torch.multinomial(probabilities, 1).item()  # Randomly sample one token
            
            # Get the actual token ID
            next_token = top_k_indices[next_token_idx].item()

            # If we hit the <eos> token, stop the translation process
            if next_token == vocab_transform[TRG_LANGUAGE].get_stoi().get('<eos>', None):
                break
            
            # Append to the output sequence
            output_sequence.append(next_token)
            
            # Update decoder input with the predicted token
            decoder_input = torch.cat([decoder_input, torch.tensor([[next_token]]).to(device)], dim=1)
        
        # Convert numericalized tokens back to text
        translated_text = ' '.join([vocab_transform[TRG_LANGUAGE].get_itos()[word] for word in output_sequence])
        print(f"Translated text: {translated_text}")
        return translated_text
    except Exception as e:
        print(f"Error in translation function: {e}")
        return None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        input_text = request.form['text']
        if not input_text:
            raise ValueError("No input text provided.")
        
        translated_text = translate(input_text)
        
        if translated_text:
            return jsonify({'translation': translated_text})
        else:
            return jsonify({'error': 'Translation failed'}), 500
    except KeyError:
        return jsonify({'error': 'Missing input text'}), 400
    except Exception as e:
        print(f"Error in /translate route: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
