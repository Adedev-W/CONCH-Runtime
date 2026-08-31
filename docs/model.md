# Model and Capabilities

This guide describes the lower-level CONCH model API and its limitations. The HTTP surface currently exposes only candidate-query ranking through `/rank`.

## What CONCH provides

The bundled `conch_ViT-B-16` configuration is a pathology-focused vision-language model with:

- A ViT-B/16 image encoder using a default image size of `448 x 448`.
- A normalized 512-dimensional contrastive image embedding.
- A normalized text embedding in the shared image-text space.
- An image-conditioned multimodal decoder for autoregressive caption generation.
- Attention-pooled visual tokens used by the captioning path.

## Image feature extraction

The public model API returns a global embedding:

```python
with torch.inference_mode():
    image_features = model.encode_image(
        image_tensor,
        normalize=True,
        proj_contrast=True,
    )
```

This embedding is useful for similarity search, retrieval, clustering, nearest-neighbor analysis, and downstream classifiers. The current HTTP service does not expose raw embeddings; it uses them internally for query ranking.

## Manual query ranking

The runtime's `/rank` endpoint uses manually supplied text candidates. It encodes the image and each query, computes their similarity, and sorts the candidates. The returned softmax value is relative to the submitted candidates, not a calibrated confidence score.

Good query sets should be comparable and use clear pathology terminology, for example:

```json
[
  "malignant tumor cells",
  "benign tissue",
  "normal tissue"
]
```

## Auto-captioning

The lower-level `model.generate()` method can generate a caption without a manual query. When `text=None`, generation begins with the start-of-text token and predicts the next token autoregressively using image-conditioned visual tokens. A text tensor can also be provided as a prefix.

The current implementation supports `top_k` and `top_p` sampling. Use an explicit supported generation type:

```python
with torch.inference_mode():
    output_ids = model.generate(
        image_tensor,
        seq_len=30,
        generation_type="top_k",
        top_k=5,
        temperature=0.7,
    )

caption = tokenizer.batch_decode(
    output_ids,
    skip_special_tokens=True,
)[0]
```

Captioning requires the `transformers` dependency and is currently available only through the Python model API, not through `/rank`.

## Localization limitations

CONCH is not trained or exposed here as a dedicated object detector. It does not directly return bounding boxes, class confidence per box, or segmentation masks.

Weak localization can be built as a separate workflow by scoring image crops or patch-level features against a text prompt and converting high-scoring regions into a heatmap or candidate boxes. Reliable detection requires a detector or segmenter trained with suitable box or mask annotations.

## Intended use and limitations

CONCH is a foundation model for computational pathology research. Performance depends on staining, magnification, tissue preparation, image quality, prompt wording, and domain shift. Outputs require expert review and must not be interpreted as a medical diagnosis.
