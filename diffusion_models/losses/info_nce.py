import torch
import torch.nn.functional as F


def info_nce(embeddings, attrs, temperature=0.07, similarity_threshold=0.8):
    """
    Custom InfoNCE loss using cosine similarity for embeddings, guided by attribute similarity.
    This loss is used to train the attribute embedder.
    Given a batch attributes, the positive pairs are the attributes with similarity greater than the similarity_threshold.
    The negative pairs are all other attributes in the batch.

    Args:
        embeddings (torch.Tensor): Shape [B, 512], embeddings from the attribute embedder.
        attrs (torch.Tensor): Shape [B, 40], multi-hot attribute vectors.
        temperature (float): Temperature parameter for scaling similarity, default 0.07.
        similarity_threshold (float): Threshold for defining positive pairs based on attr similarity.

    Returns:
        torch.Tensor: Scalar InfoNCE loss.
    """
    # Ensure types and device
    embeddings = embeddings.float()
    attrs = attrs.float()
    
    # Compute cosine similarity between embeddings: [B, B]
    emb_sim_matrix = F.cosine_similarity(
        embeddings.unsqueeze(1),  # [B, 1, 512]
        embeddings.unsqueeze(0),  # [1, B, 512]
        dim=-1
    )
    
    # Compute similarity between attribute vectors: [B, B]
    # Use cosine similarity for multi-hot vectors (binary)
    attr_sim_matrix = F.cosine_similarity(
        attrs.unsqueeze(1),  # [B, 1, 40]
        attrs.unsqueeze(0),  # [1, B, 40]
        dim=-1
    )
    
    # Define positive mask based on attribute similarity: [B, B]
    # 1 for positive pairs (similar attrs), 0 otherwise
    pos_mask = (attr_sim_matrix >= similarity_threshold).float()
    
    # Scale embedding similarities by temperature
    emb_sim_matrix = emb_sim_matrix / temperature  # [B, B]
    exp_sim = torch.exp(emb_sim_matrix)  # [B, B]
    
    # Positive similarities: sum over pairs marked as positive
    pos_sim = (exp_sim * pos_mask).sum(dim=1)  # [B], sum of exp(sim) for positive pairs
    
    # Total sum of similarities (positive + negative)
    sum_sim = exp_sim.sum(dim=1)  # [B]
    
    # Avoid division by zero
    pos_sim = torch.clamp(pos_sim, min=1e-8)
    sum_sim = torch.clamp(sum_sim, min=1e-8)
    
    # InfoNCE loss per sample: -log(pos / total)
    loss_per_sample = -torch.log(pos_sim / sum_sim)  # [B]
    
    # Mean over batch
    loss = loss_per_sample.mean()
    
    return loss
