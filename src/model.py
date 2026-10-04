from transformers import SegformerForSemanticSegmentation

ID2LABEL = {0: "obstacle", 1: "water", 2: "sky"}


def build_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=len(ID2LABEL),                                  
        id2label=ID2LABEL,
        label2id={v: k for k, v in ID2LABEL.items()}, 
        use_safetensors=True,                                   
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_model()
    x = torch.randn(2, 3, 384, 512)                     
    out = model(pixel_values=x)
    print(out.logits.shape)