import torch
import torch.nn as nn
from torch.nn import functional as F
import mini_gpt


model=mini_gpt.BigramLanguageModel(mini_gpt.V).to(mini_gpt.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
#entrainement du modèle
for k in range(5000):
    x,y=mini_gpt.get_batch("train")
    logits,loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if k%100==0:
        print(loss)

#Test du bigramm model entrainé
context = torch.zeros((1, 1), dtype=torch.long, device=mini_gpt.device)
generated = model.generate(context, max_new_tokens=300)[0].tolist()
print(mini_gpt.decode(generated))

