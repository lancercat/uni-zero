# ECCV: Beyond Script Family Boundaries: Towards Unified Open-Set Scene Text Recognition
This repo inculdes 
- the 50 samples (sampled_images) that we benchmarked ChatGPT 5.2 upon
- the ground truth files (gt)
- our's output (ours-uni)
- ChatGPT output (gpt)
- and the script to produce chatgpt results (oai.py), in case you want to adjust the prompts/parameters and test yourself
## Note 
This dataset is NOT the full synthetic dataset. 

It is a subset of that to reduce token cost of using VLM APIs. 

We believe we don't need all 5000 sample to illustrate a 140% vs ~50% CER capability gap. 
