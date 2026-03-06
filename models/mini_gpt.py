import torch
import torch.nn as nn
from torch.nn import functional as F

#Lis un corpus de texte (le Tome 1 du Comte de Monte-Cristo)
with open("/home/ajyad/Documents/Code/VSCode/MicroGPT/data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

#print("Longueur du corpus :", len(text))
#print(text[:500])
# text est le corpus de text (c'est un chaine de caractère)
chars = sorted(set(text))
V=len(chars)

# stoi est un dictionnaire qui a chaque caractère associe un entier
stoi = {}
for i in range(V):
    stoi[chars[i]]=i
#itos est un dictionnaire qui a chaque entier associe un caractère
itos = {}
for i in range(V):
    itos[i]=chars[i]
#encode(s) est une fonction qui a une chaine de caractère renvoie la liste d'entiers correspondante
def encode(s):
    S=list(s)
    I=[]
    for i in range(len(S)):
        I.append(stoi[S[i]])
    return I
#decode(I) est l'opération réciproque de encode(s)
def decode(I):
    S=[]
    for i in range(len(I)):
        S.append(itos[I[i]])
    s="".join(S)
    return s
#data est un tenseur ou tout les caractères du texte du corpus est représenté sous forme d'indice (l'ordre est conservé) 
data=torch.tensor(encode(text),dtype=torch.long)
n=len(data)
#k correspond à l'indice correspondant à 90% de la taille de data
k=int(0.9*n)
train_data=data[:k]
val_data=data[k:]

#batch_size:le nombre de séquence renvoyé, block_size:la taille d'une séquence
batch_size = 32
block_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_batch(split):
    data_source = train_data if split == "train" else val_data
    T=torch.randint(0,len(data_source)-block_size-1,(batch_size,))
    x=[]
    y=[]
    for ix in T:
        x.append(data_source[ix:ix+block_size])
        y.append(data_source[ix+1:ix+block_size+1])
    x=torch.stack(x).to(device)
    y=torch.stack(y).to(device)
    return x,y

# Création d'un BIgramm Language Model dans un premier temps pour bien comprendre comment faire le miniGPT aprés
class BigramLanguageModel(nn.Module):
    #self.scores a la distribution de probabilité (dans les faits ce sont des scores) du prochain caractère associé à chaque caractères
    def __init__(self, vocab_size):
        super().__init__()
        self.scores=nn.Embedding(vocab_size,vocab_size)
    def forward(self,idx,target=None):
        #idx est de dimension: B, T où B est la taille du batch(le nombre de séquence) et T(la taille d'une séquence)
        #logits lui est donc de dimension B, T et vocab_size, cela représente les ditribution de probabilité pour chaque caractères des B séquences
        logits=self.scores(idx)
        loss=None
        if target is not None:
            #Il faut redimensionner target et logits pour la fonction cross entropy
            # target est de dimension B,T (Les B*T index des bon caractères suivants associé à idx)
            B,T,C=logits.shape
            logits_plat=logits.view(B*T,C)
            target_plat=target.view(B*T)
            loss = F.cross_entropy(logits_plat,target_plat)    
        return logits,loss
    def generate(self,idx,max_new_tokens):
        #final représente les B sequences finalisé qui sont donc de longueur T+max_new_tokens
        final=idx
        for i in range(max_new_tokens):
            logits,_=self(final)
            #Un bigramm model ne prévoit qu'à partir d'un caractère (il n'y a pas de notion de sequence), pour chaque séquence, il faut donc séléctionner le dernier caractères pour avoir celui d'aprés
            #logits va donc devenir de dimension B, vocab_size
            logits=logits[:,-1,:]
            #On transforme logits en "proba"
            probs=F.softmax(logits, dim=-1)
            #On choisit le caractère suivant (On ne prend pas toujours le meilleurs, mais à 90% le meilleur)
            idx_next = torch.multinomial(probs, num_samples=1)
            final=torch.cat((final,idx_next),dim=1)
        return final
            
