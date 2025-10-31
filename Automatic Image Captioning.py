import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Load model and tokenizer
model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Fix pad_token if missing
if model.config.pad_token_id is None:
    model.config.pad_token_id = model.config.eos_token_id

def generate_caption(image_path, max_length=16, num_beams=4):
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    # Add attention mask (all ones)
    attention_mask = torch.ones(pixel_values.shape[:-1], dtype=torch.long).to(device)

    # Generate caption
    output_ids = model.generate(
        pixel_values,
        attention_mask=attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        pad_token_id=model.config.pad_token_id
    )
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

# Single image path
image_path = "134206.jpg"   
caption = generate_caption(image_path)
print("Generated Caption:", caption)



