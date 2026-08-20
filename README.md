# vlm-from-scratch

A transformer, a Vision Transformer, and a PaliGemma-style vision-language model built entirely from scratch — no `torch.nn.Transformer`, no pretrained backbones. Every component implemented, trained and verified.

**Verified, not just shape-checked.** The ViT trains to 0.95 on real MNIST in 8 epochs. The VLM generates autoregressively — its own predictions fed back in — not only under teacher forcing. Image-ablation tests confirm the vision tower actually drives the output rather than a language-modelling shortcut, and exposure bias is documented as a known limitation rather than hidden.
