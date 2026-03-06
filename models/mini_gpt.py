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

#Passons à un vrai model GPT (on considère toujours que 1 token = 1 caractère)
# n_embd: la taille de nos vecteurs d'information lié a un caractère
# block_size:la taille du contexte (le nombre de caractère d'une séquence)
n_embd = 128 
block_size = 64 
dropout = 0.2
#On commence par coder la self attention causale
class Head(nn.Module):
    #head_size: la taille du vecteur ligne associé à une clé, une valeur ou une requête
    def __init__(self,head_size):
        super().__init__()
        #Permet d'avoir les matrice Q,K,V par lesquels on fait Qx, Kx, Vx; il n'y a pas de biais
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        #.reguster_buffer permet de dire que ce n'est pas un tenseur avec des paramètres à entrainer
        #self.tril donne le "masque" il y a des 1 ou on peut voir le contexte (le passé et le présent) et 0 pour le futur
        # les 1 sont en bas à gauche de la matrice carré 
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        #la couche de dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #Attention(Q,K,V) = {softmax}(scores)*V.
        #x a pour dimension B: le nb de séquences,T:la taille d'une séquence,C=n_embd
        B,T,C=x.shape
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        kT=k.transpose(1,2)
        wei=q@kT
        wei=wei*(k.shape[-1]**-0.5)
        #On crée le masque adapté à softmax, celui avec des -l'infini et 0
        #Ce sont ces valeurs qu'on prend car exp(-infini)=0 et exp(0)=1
        wei=wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))
        #rappel pour plus tard: c'est sur les colonnes de la matrice de chaque sequence qu'on veut une distribution de proba (d'ou le -1)
        wei=F.softmax(wei,-1)
        wei=self.dropout(wei)
        #Le résultat est de dimension B,T,head_size
        return wei@v

#Mise en place du Multi-Head
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        #Avoir une liste des 'num_heads' head(nn.ModuleList est important pour que les poids soit bien mémorisé pour la backpropagation)
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        #La projection pour s'assure de revenir dans les bonnes dimension même si en théorie num_heads*head_size=n_embd
        self.proj=nn.Linear(num_heads*head_size,n_embd)
        #le dropout final
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        H=torch.cat([head(x) for head in self.heads],dim=-1)
        H=self.proj(H)
        H=self.dropout(H)
        return H
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        #Création de mon réseaux, l'empilement de couche 
        self.reseaux=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.reseaux(x)
    
#Il est temps de créer le block qui assemble le tout
class Block(nn.Module):
    def __init__(self,num_head):
        super().__init__()
        head_size=n_embd//num_head
        self.multi_head=MultiHeadAttention(num_head,head_size)
        self.feedforward=FeedForward()
        #Il y a 2 couche de normalisation distincte
        self.layer_norm1=nn.LayerNorm(n_embd)
        self.layer_norm2=nn.LayerNorm(n_embd)
    def forward(self,x):
        #On fait de la pre-norm: hpre_norm=h+f(LN(h))
        H=self.layer_norm1(x)
        H1=self.multi_head(H)+x
        H2=self.layer_norm2(H1)
        H3=self.feedforward(H2)+H1
        return H3
#Le dernier gros bloc!! Il est temps de faire le GPTmodel
class GPTLanguageModel(nn.Module):
    def __init__(self,n_layer,num_head):
        super().__init__()
        self.token_embeding=nn.Embedding(V,n_embd)
        #Rappel:block_size correspond à la taille max de la sequence entrante
        self.pos_embeding=nn.Embedding(block_size,n_embd)
        self.block=nn.Sequential(*[Block(num_head) for _ in range(n_layer)])
        self.LN=nn.LayerNorm(n_embd)
        self.pred=nn.Linear(n_embd,V)
    def forward(self,x,target=None):
        #x est de dimension B,T
        B,T=x.shape
        #on génére un tenseur ordonée de 0 à T-1
        pos = torch.arange(T, device=x.device)
        P=self.pos_embeding(pos)
        x=P+self.token_embeding(x)
        x=self.block(x)
        x=self.LN(x)
        logits=self.pred(x)
        loss=None
        if target is not None:
            #Il faut redimensionner target et logits pour la fonction cross entropy
            # target est de dimension B,T (Les B*T index des bon caractères suivants associé à idx)
            B,T,C=logits.shape
            logits_plat=logits.view(B*T,C)
            target_plat=target.view(B*T)
            loss = F.cross_entropy(logits_plat,target_plat)    
        return logits,loss

        


        
