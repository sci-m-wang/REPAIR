from easyeditor import BaseEditor, GraceHyperParams
from transformers import AutoTokenizer
import torch

def main():
    # Path to the config
    hparams_path = 'hparams/GRACE/demo.yaml'

    print(f"Loading hparams from {hparams_path}")
    # Load hparams
    hparams = GraceHyperParams.from_hparams(hparams_path)

    print("Initializing editor...")
    # Initialize editor
    editor = BaseEditor.from_hparams(hparams)

    # Edit request
    prompts = ['The Eiffel Tower is in']
    target_new = ['Rome']
    subject = ['Eiffel Tower']

    print(f"Editing: '{prompts[0]}' -> '{target_new[0]}'")
    
    # Execute edit
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True,
        test_generation=True
    )

    print("Edit metrics:", metrics)

    # Test generation manually
    print("\nVerifying edit manually:")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    device = f"cuda:{hparams.device}" if torch.cuda.is_available() else "cpu"
    
    input_text = "The Eiffel Tower is in"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    print(f"Generating from: '{input_text}'")
    # Use the edited model for generation
    # Note: BaseEditor.edit returns the edited model
    out = edited_model.generate(**inputs, max_new_tokens=10)
    print("Output:", tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
