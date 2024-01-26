import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET

from utils import (
    load_checkpoint, # carica uno stato salvato di un modello di rete neurale. In pytorch, un "checkpoint" può includere i pesi del modello, i parametri dell'ottimizzatore e altri metadati. E' una funzione utile per riprendere il training da un punto specifico o per valutare un modello già addestrato.
    save_checkpoint, #questo comando salva lo stato attuale di un modello durante l'addestramento. E' utile per preservare il modello ad intervalli regolari durante il training permettendo di ripristinare lo stato del modello in caso di interruzioni o per altre analisi.
    get_loaders, #gestisce la divisione di dati in set di training, validazione e test. 
    check_accuracy, # valutazione delle prestazioni di un modello su un set di dati di validazione o test.
    save_predictions_as_imgs, #Funzione utilizzata per salvare le predizioni del modello sotto forma di immagini. Nel contesto della segmentazione può salvare le maschere di segmentazione predette dal modello come file immagine per un'analisi visiva.
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4 #imposta il tasso di apprendimento, determina la velocità con cui i pesi della rete si aggiornano durante la backpropagation. Un tasso di apprendimento troppo alto può causare un apprendimento instaile, mentre uno tropppo basso può rallentare il processo di training.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1 # in ogni iterazione di training verranno elaborati 16 esempi.
NUM_EPOCHS = 30 # Un'epoca è un ciclo completo attraverso l'intero set di training. Questo valore determina quante volte il modello vedrà l'intero set di dati durante il training.
NUM_WORKERS =8 # numero di workers =2 per il caricamento dei dati. Questo aiuta a velocizzare il processo di caricamento dei dati, utilizzando più processi in parallelo. In pytorch quando si utilizza un "DataLoader" è possibile accelerare il caricamento dei dati utilizzando più worker. Ogni "worker" è un processo che carica una parte del dataset in parallelo con gli altri. avere più worker significa che mentre il modello sta elaborando il batch corrente, i worker possono già caricare i dati per il prossimo batch, riducendo i tempi di attesa e aumentando l'efficienza del training.
IMAGE_HEIGHT = 512 
IMAGE_WIDTH = 512 
PIN_MEMORY = True #consente di bloccare la memoria durante il caricamento dei dati migliorando le prestazioni durante il trasferimento dei dati dalla CPU alla GPU. E' sempre un opzione nel "dataLoader" di pytorch che, se impostata su True, consente di preallocare la memoria per il prossimo batch di dati. 
LOAD_MODEL = False # indica che si vuole caricare un modello non pre addestrato. Utile per continuare il training da un checkpoint o per valutare un modello già addestrato
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"
CHECKPOINT_PATH = "my_checkpoint.pth.tar"
SAVED_IMAGES_PATH = "saved_images"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop): #data sono i dati mentre targets sono le verità di base del batch corrente
        data = data.to(device = DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE) #comversione dei target in numeri floating point, viene aggiunta una dimensione extra all'inizio ( se necessario, per far corrispondere la dimensione prevista dal modello) e li sposta sul dispositivo di addestramento.

        #forward
        with torch.cuda.amp.autocast(): #viene sfruttata la precisione mista per accellerare l'addestramento. la precisione mista utilizza una combinazione di float32 (precisione standard) e float16 (precisione ridotta) per ottimizzare le prestazioni di calcolo, specialmente su GPU compatibili, questo può portare ad un addestramento più veloce e a un minor utilizzo della memoria.
            predictions = model(data) #i dati vengono passati attraverso il modello per ottenere le previsioni. in quedta parte la rete fa le sue impotesi basate sui dai attuali 
            loss = loss_fn(predictions,targets) # calcola la perdita (o errore) utilizzando la funzione di perdita definita (loss_fn). Questa perdita misura quanto le previsioni del modello si discostano dai target reali.

            #I gradienti giocano un ruolo cruciale durante l'algoritmo di backpropagation, che è utilizzato per aggiornare i pesi della rete durante l'addestramento.

        #backward
        optimizer.zero_grad() # prima di iniziare il passaggio all'indietro azzera tutti i gradienti del modello perchè per impostazione predefinita i gradienti si accumulano in pytorch.
        scaler.scale(loss).backward() #calcola i gradienti della perdita rispetto ai prarametri della rete.  scaler.scale() è usato per scalare la perdita, mantenendo la stabilità numerica durante l'addestrmento in precisione mista.
        scaler.step(optimizer) #eseguie un passo di ottimizzazione, aggiornando i pesi della rete. scaler è utilizzato per garantire la precisione durante l'aggiornamento dei pesi.

        # In particolare l'ottimizzatore aggiorna i pesi del modello. Tuttavia, poichè la perdita è stata scalata, anche i gradienti sono scalati. Il "GradScaler gestisce questo aspetto, assicurando che i gradienti vengano descalati (riportati alla loro dimensione originale) prima di applicare l'aggiornamento.
        # "
        scaler.update() #aggiorna lo scaler per il prossimo barch, preparandolo per il prossimo ciclo di addestramento.
        #Dopo ogni passo di addestramento, GradScaler deve decidere se il fattore di scala usato era appropriato o se deve essere modificato.

        #Se i gradienti non hanno problemi di overflow, GradScaler può provare ad usare un fattore di scala più grande, permettendo di sfruttare ancora di più la precisione ridotta senza incorrere in problemi numerici

        #Se ci sono stati problemi di overflow, GradScaler riduce il fattore di scaka per evitare questi provblemi in futuro

        #Questo equilibrio consente di sfruttare i benefici della precisione mista riducendo al minimo i lrischio di problemi numerici.

        # update tqdm loop
        loop.set_postfix(loss=loss.item()) #aggiorna la barra di progresso tqdm mostrando il valore corrente della perdita. loss.item() converte il valroe della perdita da un tensore a un numero float standard, rendendolo leggibile per la visualizzazione. Fa parte di un meccaniscmo che permette di utilizzare la precisiokne mista in modo sicuro, evitando i problemi associati all'uso di float16 per il calcolo dei gradienti, mantenendo al contempo i benefici in termini di prestazioni e utilizzo della memoria.


def main(): # di seguito il codice utilizza la libreria Albumentation per l'augmentation delle immagini
    train_transform = A.Compose(
        [
            A.Resize(height= IMAGE_HEIGHT, width = IMAGE_WIDTH), #modifica le dimensioni delle immagini per farle corrispondere a una dimensuione specifica, questo è importante per garantire che tutte le immagini abbiano la stessa dimensione prima di essere inserite nella rete.
            A.Rotate(limit=15, p=0.5), # Ruota l'immagine di un angolo casuale all'intenro del range specificato. il parametro p indica la probabilità che la trasformazione venga applicata.
            A.HorizontalFlip(p=0.5), #riflette l'immagine orizzontalmente con una probabilità del 50%. Questo aiuta il modello ad essere meno sensibile all'orientamento orizzontale delle immagini.
            A.ElasticTransform(alpha=1, sigma =50, p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(blur_limit=(3,5), p=0.1),
            A.Normalize( #di seguito vengono normalizzati i valori dei pizel dell'immagine. I valori "mean" e "std" sono usati per normalizzare ogni canale dell'immagine (RGB in questo caso). La normalizzazione aiuta a stabilizzare l'addestramento e la convergenza della rete.
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(), #qui viene convertita l'immagine da una matrice NumPy o una immmagine PIL in un tensore Pytorch. E' un passabggio necessario per processare le immagini con pytorch.

            #Converte l'immagine in un tensore PyTorch e la trasla dal formato [altezza, larghezza, canali] al formato [canali, altezza, larghezza], che è il formato richiesto da PyTorch per le immagini
        ]
    )

    val_transforms = A.Compose( #le trasformazioni di validazione sono simili ma in genere più leggere di quelle di addestramento poichè l'obiettivo è valutare il modello su dati non alterati o minimamente alterati. In questa fase di validazione si desidera valutare il modello su immagin il più possibile simili a quelle reali
        
    #la fase di validazione è un passaggio in cui si valutano le prestazioni del modello su un dataset che non è ancora stato utilizzato per l'addestramento, noto come set di validazione.
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH), #stesse operazioni di ridimensionamento e normalizzazione 
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ]
    )

    model =UNET(in_channels=3, out_channels = 1).to(DEVICE) # si crea un istanza della rete U-net specificando che la rete accetta immagini con 3 canali (RGB) e restituisce un output a canale singolo (spesso usato per le maschere di segmentazione). Il modello viene poi spostato sul dispositivo specificato (tipicamente una GPU) per un addestramento più efficiente.

    loss_fn = nn.BCEWithLogitsLoss() #viene utilizzata la funzione di perdita "binary cross entrompy with logits", è una funzione comune per i problemi di classificazione binaria e segmentazione.

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # si sceglie l'ottimizzatore Adam per l'aggiornamento dei pesi del modello, con un tasso di apprendimento specificato("LEARNING_RATE").
    #L'ottimizzatore adam è uno degli algoritmi di ottimizzazioee più poplari nel campo dell'apprendimento profondo. E' noto per la sua efficacia nell'addestrare reti neurali in una vasta gamma di applicazioni. Adam combina le idee di due altri metodi di ottimizzazione: Momentum e RMSprop. Momentum aiuta l'ottimizatore a mantenere la direzione in cui i gradienti sono stati grandi in passato. Questo aiuta a velocizzare l'addestramento e ridurre il rischio di rimanere bloccati in minimi locali poco profondi.RMSprop, un altro componente di Adam, modifica il tasso di apprendimento per ciascun parametro. Fa ciò tenendo traccia della media mobile del quadrato dei gradienti. questa informazione viene usata per normalizzare l'aggiornamento dei pesi, il che è utile in scenari con gradienti molto variabili.

    # In Adam, ogni parametro del modello ha il proprio tasso di apprendimento, che si adatta nel tempo. L'algoritmo mantiene due vettori di momenti chiamati m (media del gradiente) e v (media non centrata del quadrato dei gradienti). Questi momenti sono versioni smussati del gradiente e del suo quadrato rispettivamente. L'aggiornamento dei pesi in ogni passaggio di addestramrnto è basato su quste stime smussate, che prendono in considerazione sia la magnitudine (tramite "v") sia la direzione (tramite "m") dei gradienti passati.



    train_loader, val_loader = get_loaders(  #Questa funzione prepara i DataLoader per i set di addestramento e validazione. I DataLoader sono utilizzati in Pytorch per caricare i dati in batch, facilitando l'addestramento efficiente su grandi data set.
        
    #I DataLoader permettono di caricare e processaren i dati in batch più piccioli (invece di utilizzare tutto il dataset). I DataLoader permettono anche di mescolare i dati ad ogni epoca prevendendo l'overfiting e assicurando che il modello non impari semplicemente l'ordine dei dati. inoltre possono caricare i dati in parallelo utilizzando più processi.
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"),model)

    check_accuracy(val_loader, model, device = DEVICE)


    scaler = torch.cuda.amp.GradScaler() #inizializza un oggetto Gradacaler per l'addestramento con precisione mista. Questo aiuta a ottimizzare le prestazioni quando si utilizzano GPU.

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler) #in ogni epoca la funzione train_fn viene chiamata per addestrare il modello sul set di addestramento. questa funzione eseguie il passaggio in avanti, il calcolo della perdita, il passaggio indietro e l'aggiornamento dei pesi del modello.
        
        #save model
        checkpoint = {
            "state_dict":model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint,CHECKPOINT_PATH)

        #check accuracy
        check_accuracy(val_loader, model, device = DEVICE)

        #print some examples to a folder
        save_predictions_as_imgs(
            val_loader,model,folder=SAVED_IMAGES_PATH, device=DEVICE
        )

if __name__ == "__main__":
    main()
