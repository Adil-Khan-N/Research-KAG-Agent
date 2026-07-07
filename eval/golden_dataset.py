"""
Golden evaluation dataset — 40 questions with ground truth answers.

Deliberately spans 3 types:
- Simple factual (15): single paper, direct answer
- Comparison (10): across 2+ papers
- Multi-hop (15): requires traversing relationships

This split matters for RAGAS — the graph should help most on multi-hop.
"""

GOLDEN_DATASET = [

    # ── SIMPLE FACTUAL (15) ───────────────────────────────────
    {
        "question": "What is the key innovation of Vision Transformer (ViT)?",
        "ground_truth": "ViT applies a pure Transformer architecture directly to sequences of image patches for image recognition, demonstrating that reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform well on image classification tasks when pre-trained on large datasets.",
        "type": "factual",
        "arxiv_ids": ["2010.11929"],
    },
    {
        "question": "What image size does ViT use as its default patch size?",
        "ground_truth": "ViT uses 16x16 pixel patches as its default patch size, splitting images into a sequence of fixed-size non-overlapping patches.",
        "type": "factual",
        "arxiv_ids": ["2010.11929"],
    },
    {
        "question": "What is the shifted window mechanism in Swin Transformer?",
        "ground_truth": "The shifted window mechanism in Swin Transformer shifts the partitioning of windows between consecutive transformer layers, allowing cross-window connections while maintaining efficient non-overlapping window-based self-attention computation.",
        "type": "factual",
        "arxiv_ids": ["2103.14030"],
    },
    {
        "question": "What dataset does Swin Transformer primarily evaluate on?",
        "ground_truth": "Swin Transformer primarily evaluates on ImageNet-1K for image classification and also on COCO for object detection and ADE20K for semantic segmentation.",
        "type": "factual",
        "arxiv_ids": ["2103.14030"],
    },
    {
        "question": "What is the distillation token used in DeiT?",
        "ground_truth": "DeiT introduces a distillation token that interacts with the class token and patch tokens through self-attention, allowing the student transformer model to learn from a teacher network (typically a CNN) through attention-based distillation.",
        "type": "factual",
        "arxiv_ids": ["2012.12877"],
    },
    {
        "question": "What is masked autoencoder pretraining in MAE?",
        "ground_truth": "MAE randomly masks a high proportion (75%) of image patches and trains an asymmetric encoder-decoder architecture to reconstruct the missing pixels, learning rich visual representations through this self-supervised pretraining approach.",
        "type": "factual",
        "arxiv_ids": ["2111.06377"],
    },
    {
        "question": "What does DETR stand for and what is its main contribution?",
        "ground_truth": "DETR stands for Detection Transformer. Its main contribution is formulating object detection as a direct set prediction problem using a transformer encoder-decoder architecture with bipartite matching loss, eliminating the need for hand-crafted components like anchor generation and non-maximum suppression.",
        "type": "factual",
        "arxiv_ids": ["2005.12872"],
    },
    {
        "question": "What is the hierarchical feature representation in Swin Transformer?",
        "ground_truth": "Swin Transformer constructs hierarchical feature maps by merging image patches in deeper layers, starting from small patch sizes and increasing the receptive field progressively, similar to feature pyramid networks in CNNs, making it suitable for dense prediction tasks.",
        "type": "factual",
        "arxiv_ids": ["2103.14030"],
    },
    {
        "question": "What training dataset does ViT require to achieve good performance?",
        "ground_truth": "ViT requires large-scale pretraining datasets like JFT-300M or ImageNet-21k to achieve competitive performance, as it lacks the inductive biases of CNNs and needs more data to learn equivalent visual features.",
        "type": "factual",
        "arxiv_ids": ["2010.11929"],
    },
    {
        "question": "What is BEiT's pretraining objective?",
        "ground_truth": "BEiT pretrains vision transformers using a masked image modeling objective inspired by BERT, where image patches are masked and the model predicts the visual tokens of masked patches obtained from a discrete VAE tokenizer.",
        "type": "factual",
        "arxiv_ids": ["2106.08254"],
    },
    {
        "question": "What is ConViT's approach to combining convolutions and attention?",
        "ground_truth": "ConViT introduces gated positional self-attention that can be initialized as a convolutional layer through soft convolutional inductive biases, allowing the model to smoothly transition from convolutional to attentive behavior during training.",
        "type": "factual",
        "arxiv_ids": ["2103.10697"],
    },
    {
        "question": "What is the main challenge addressed by MobileViT?",
        "ground_truth": "MobileViT addresses the challenge of designing lightweight vision transformers for mobile devices by combining the global processing of transformers with the local processing of convolutions in a parameter-efficient architecture.",
        "type": "factual",
        "arxiv_ids": ["2110.02178"],
    },
    {
        "question": "What is the Pyramid Vision Transformer's key design feature?",
        "ground_truth": "PVT introduces a pyramid structure into the Vision Transformer by progressively shrinking the feature map through patch embedding layers between stages, enabling it to generate multi-scale feature maps suitable for dense prediction tasks without convolutions.",
        "type": "factual",
        "arxiv_ids": ["2104.01136"],
    },
    {
        "question": "What attention mechanism does Swin Transformer V2 introduce?",
        "ground_truth": "Swin Transformer V2 introduces scaled cosine attention, which computes attention logits using a scaled cosine function instead of dot products, and log-spaced continuous position bias for handling varying image resolutions.",
        "type": "factual",
        "arxiv_ids": ["2111.09883"],
    },
    {
        "question": "How does MAE differ from BERT in its masking strategy?",
        "ground_truth": "MAE masks a much higher proportion of patches (75%) compared to BERT's 15% token masking, because image patches contain significant spatial redundancy and predicting missing pixels requires a higher masking ratio to create a challenging pretraining task.",
        "type": "factual",
        "arxiv_ids": ["2111.06377"],
    },

    # ── COMPARISON (10) ──────────────────────────────────────
    {
        "question": "What are the main architectural differences between ViT and Swin Transformer?",
        "ground_truth": "ViT computes self-attention globally across all image patches with quadratic complexity, while Swin Transformer computes attention locally within non-overlapping windows with linear complexity. Swin also uses a hierarchical architecture with patch merging, while ViT maintains the same resolution throughout.",
        "type": "comparison",
        "arxiv_ids": ["2010.11929", "2103.14030"],
    },
    {
        "question": "How do DeiT and ViT differ in their training approach?",
        "ground_truth": "ViT requires large-scale pretraining on JFT-300M or ImageNet-21k, while DeiT can train effectively on ImageNet alone using knowledge distillation from a CNN teacher, data augmentation, and regularization techniques, making it more data-efficient.",
        "type": "comparison",
        "arxiv_ids": ["2010.11929", "2012.12877"],
    },
    {
        "question": "Compare the pretraining objectives of MAE and BEiT.",
        "ground_truth": "Both MAE and BEiT use masked image modeling but differ in their reconstruction targets. MAE reconstructs raw pixel values of masked patches, while BEiT predicts discrete visual tokens from a pretrained tokenizer (dVAE), making BEiT's targets more semantic.",
        "type": "comparison",
        "arxiv_ids": ["2111.06377", "2106.08254"],
    },
    {
        "question": "How does Swin Transformer V2 improve over Swin Transformer V1?",
        "ground_truth": "Swin Transformer V2 addresses instability in training larger models through scaled cosine attention, uses log-spaced continuous position bias for better resolution transfer, and employs a post-normalization architecture, enabling scaling to 3 billion parameters.",
        "type": "comparison",
        "arxiv_ids": ["2103.14030", "2111.09883"],
    },
    {
        "question": "What distinguishes DETR from traditional object detection methods?",
        "ground_truth": "DETR eliminates traditional detection pipeline components like anchor boxes, non-maximum suppression, and hand-crafted features by treating detection as a direct set prediction problem with a transformer encoder-decoder and bipartite matching loss, enabling end-to-end training.",
        "type": "comparison",
        "arxiv_ids": ["2005.12872"],
    },
    {
        "question": "Compare ViT and CNN approaches to positional information encoding.",
        "ground_truth": "ViT adds learnable 1D position embeddings to patch tokens to encode positional information, since transformers have no inherent notion of position. CNNs inherently encode position through the locality of convolutional filters and the spatial structure of feature maps.",
        "type": "comparison",
        "arxiv_ids": ["2010.11929"],
    },
    {
        "question": "How does ConViT compare to DeiT in training efficiency?",
        "ground_truth": "ConViT uses soft convolutional inductive biases through gated positional self-attention to improve training efficiency compared to standard ViT approaches. DeiT uses knowledge distillation from CNN teachers. Both aim to reduce data requirements but use different strategies.",
        "type": "comparison",
        "arxiv_ids": ["2103.10697", "2012.12877"],
    },
    {
        "question": "What is the difference between local and global attention in vision transformers?",
        "ground_truth": "Global attention computes interactions between all patch pairs with quadratic complexity as in ViT, while local attention restricts computation to local windows as in Swin Transformer, achieving linear complexity while using shifted windows to enable cross-region information exchange.",
        "type": "comparison",
        "arxiv_ids": ["2010.11929", "2103.14030"],
    },
    {
        "question": "Compare the computational complexity of ViT and Swin Transformer.",
        "ground_truth": "ViT has quadratic computational complexity with respect to image size due to global self-attention over all patches. Swin Transformer achieves linear complexity by computing self-attention within fixed-size local windows, making it more efficient for high-resolution inputs.",
        "type": "comparison",
        "arxiv_ids": ["2010.11929", "2103.14030"],
    },
    {
        "question": "How do PVT and Swin Transformer both address dense prediction tasks?",
        "ground_truth": "Both PVT and Swin Transformer use hierarchical architectures to generate multi-scale feature maps suitable for dense prediction. PVT uses progressive patch embedding shrinkage between stages, while Swin uses patch merging layers and window-based attention with linear complexity.",
        "type": "comparison",
        "arxiv_ids": ["2104.01136", "2103.14030"],
    },

    # ── MULTI-HOP (15) ────────────────────────────────────────
    {
        "question": "What datasets do papers that extend ViT use for evaluation?",
        "ground_truth": "Papers extending ViT use ImageNet-1K for image classification (DeiT, Swin), COCO for object detection (DETR, Swin), ADE20K for segmentation (Swin, PVT), and Tiny-ImageNet for small dataset experiments (Vision Transformer for Small-Size Datasets).",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929", "2012.12877", "2103.14030", "2005.12872"],
    },
    {
        "question": "What methods do papers using ImageNet-21k for pretraining introduce?",
        "ground_truth": "Papers using ImageNet-21k pretraining include ViT which introduces patch embedding and position encoding, and scaling experiments that demonstrate the importance of large-scale pretraining for vision transformers to match CNN performance.",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929"],
    },
    {
        "question": "Which papers build on the self-attention mechanism from Attention Is All You Need for vision tasks?",
        "ground_truth": "Multiple papers build on the self-attention mechanism for vision: ViT applies it to image patches directly, DETR uses encoder-decoder self-attention for detection, Swin Transformer uses windowed self-attention for efficiency, and DeiT adapts it with distillation.",
        "type": "multi_hop",
        "arxiv_ids": ["1706.03762", "2010.11929", "2005.12872", "2103.14030"],
    },
    {
        "question": "What self-supervised pretraining methods are used by papers that evaluate on ImageNet?",
        "ground_truth": "Papers evaluating on ImageNet that use self-supervised pretraining include MAE with masked pixel reconstruction, BEiT with masked visual token prediction, and DINO with self-distillation. These methods enable effective pretraining without labeled data.",
        "type": "multi_hop",
        "arxiv_ids": ["2111.06377", "2106.08254", "2203.23743"],
    },
    {
        "question": "What architectural improvements do descendants of ViT make to handle high-resolution images?",
        "ground_truth": "ViT descendants address high-resolution images through hierarchical designs: Swin Transformer uses window attention with linear complexity, PVT uses progressive patch shrinkage, T2T-ViT tokenizes images iteratively, and Multiscale Vision Transformers use multiple resolution scales.",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929", "2103.14030", "2104.01136", "2101.11605"],
    },
    {
        "question": "How do masked autoencoder papers relate to BERT pretraining strategies?",
        "ground_truth": "Masked autoencoder papers adapt BERT's masked language modeling to vision: BEiT directly applies BERT-style masked prediction to visual tokens, MAE extends it to pixel reconstruction with higher masking ratios, and both demonstrate that masked pretraining transfers effectively from NLP to vision.",
        "type": "multi_hop",
        "arxiv_ids": ["2111.06377", "2106.08254"],
    },
    {
        "question": "What position encoding methods are used across ViT-based architectures?",
        "ground_truth": "ViT uses learnable 1D absolute position embeddings, Swin Transformer uses relative position bias within windows, Swin V2 uses log-spaced continuous position bias for resolution transfer, and ConViT encodes position through gated positional self-attention.",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929", "2103.14030", "2111.09883", "2103.10697"],
    },
    {
        "question": "Which vision transformer papers address the problem of training with limited data?",
        "ground_truth": "DeiT addresses limited data through knowledge distillation and augmentation, ConViT through convolutional inductive biases, Vision Transformer for Small-Size Datasets through locality self-attention and shifted patch tokenization, and A Light Recipe for training robust ViTs through augmentation recipes.",
        "type": "multi_hop",
        "arxiv_ids": ["2012.12877", "2103.10697", "2112.13492"],
    },
    {
        "question": "What common evaluation benchmarks connect the ViT and Swin Transformer paper families?",
        "ground_truth": "ImageNet-1K connects both families as the primary image classification benchmark. COCO connects them for object detection and instance segmentation. ADE20K connects them for semantic segmentation. These shared benchmarks enable direct performance comparisons.",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929", "2103.14030", "2012.12877"],
    },
    {
        "question": "How does the concept of inductive bias relate to the design choices in ViT and its successors?",
        "ground_truth": "ViT deliberately removes CNN inductive biases (locality, translation equivariance) to rely purely on attention, requiring large datasets. Successors reintroduce biases: ConViT adds soft convolutional bias, Swin adds locality via windows, and MobileViT combines local convolutions with global attention.",
        "type": "multi_hop",
        "arxiv_ids": ["2010.11929", "2103.10697", "2103.14030", "2110.02178"],
    },
    {
        "question": "What methods do papers that use COCO dataset introduce for object detection?",
        "ground_truth": "Papers using COCO for detection include DETR which introduces end-to-end detection with bipartite matching, and Swin Transformer which serves as a hierarchical backbone for detection frameworks, both achieving strong COCO performance through transformer-based approaches.",
        "type": "multi_hop",
        "arxiv_ids": ["2005.12872", "2103.14030"],
    },
    {
        "question": "Trace the evolution from Attention Is All You Need to Masked Autoencoders.",
        "ground_truth": "The evolution goes: Attention Is All You Need (2017) introduces transformers → ViT (2021) applies transformers to image patches → BEiT (2021) adapts BERT masked prediction for images → MAE (2021) simplifies to pixel reconstruction with high masking ratio, each building on the previous work.",
        "type": "multi_hop",
        "arxiv_ids": ["1706.03762", "2010.11929", "2106.08254", "2111.06377"],
    },
    {
        "question": "What role does knowledge distillation play across vision transformer papers?",
        "ground_truth": "Knowledge distillation appears in DeiT through a dedicated distillation token learning from CNN teachers, in BEiT through discrete VAE token targets acting as soft labels, and in survey papers on masked autoencoders discussing the relationship between masked prediction and distillation-based learning.",
        "type": "multi_hop",
        "arxiv_ids": ["2012.12877", "2106.08254"],
    },
    {
        "question": "Which papers extend ViT and also evaluate on COCO?",
        "ground_truth": "Papers that extend ViT and evaluate on COCO include Swin Transformer which extends ViT with windowed attention and evaluates on COCO for detection and segmentation, and DETR which extends the transformer paradigm for end-to-end detection on COCO.",
        "type": "multi_hop",
        "arxiv_ids": ["2103.14030", "2005.12872"],
    },
    {
        "question": "What efficiency improvements do mobile-focused vision transformer papers introduce compared to standard ViT?",
        "ground_truth": "Mobile-focused papers like MobileViT reduce computation through hybrid convolution-transformer designs, LeViT uses non-uniform channel sizes and attention biases for fast inference, and both avoid the quadratic complexity of global attention while maintaining reasonable accuracy on ImageNet.",
        "type": "multi_hop",
        "arxiv_ids": ["2110.02178", "2103.10697"],
    },
]