from rewards import PickScoreScorer, HPSv2, AestheticScorer, ClipScorer, ImageRewardScorer, HPSv3Scorer

# scorer = ImageRewardScorer(device="cpu")
# scorer = PickScoreScorer(device="cpu")
scorer = AestheticScorer(device="cpu")

total = sum(p.numel() for p in scorer.parameters())

print(f"total: {total/1e6:.2f}M")
