#CUDA è una piattaforma di calcolo parallelo e un'API modello creata da
#NVIDIA in quanto le GPU sono più efficienti dei processori tradizionali (CPU)


import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

#Nella U-Net in figura ci sono 2 3x3 convoluzioni quindi si può creare una classe DoubleConv che prende canali interni ed esterni

#nn.Module è la classe base per tutti i modelli di rete neurale, ogni nuova rete neurale o componente di una rete dovrebbe ereditare questa classe; in questo modo è possibile spostare tutti i parametri della rete su GPU, di salvare e caricare lo stato del modello, di applicare funzioni come train() e eval(). 
#Pytorch utilizza in automatico il metodo forward() quando si invocano istanze della classe su dati in input.

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels): #quando si crea l'istanza DoubleConv questo metodo è chiamato automaticamente. Con in_channels e out_channels si indicano il numero di canali in entrata e in uscita per le operazioni di convoluzione
        super(DoubleConv,self).__init__() #in questo modo stiamo chiamando il costruttore della clase base nn.Module per l'ereditarietà
        self.conv= nn.Sequential( #nn sequential mette insieme una sequenza di moduli in modo ordinato, è un contenitore di pytorch
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False), #in_channels e out_channels sono i canali di ingresso e uscita, 3 è la dimensione del kernel, 1 è il passo, 1 è il padding e bias =False indica che non viene utilizzato il termine di bias

            #La dimensione del kernel stabilisce l'area dell'input che viene considerata, se prendo 3x3 considero un'area di 3x3 pixel alla volta

            #padding  è l'aggiunta di pixel (solitamente di valore zero) intorno ai bordi dell'immagine in input. Ciò viene fatto per consentire al kernel di convoluzione di essere applicato anche ai pixel sui bordi dell'immagine. Viene utilizzato anche per controllare la dimensione dell'output.
        

            nn.BatchNorm2d(out_channels), #nn.BatchNorm2d è un layer di normalizzazione batch che stabilizza l'apprendimento e normalizza la distribuzione degli input al layer successivo.

            nn.ReLU(inplace=True), # è una funzione di attivazione ReLu, con inplace=True si indica che modificherà direttamente i dati in input risparmiando memoria.


            nn.Conv2d(out_channels, out_channels,3,1,1,bias=False), #qui i canali sono entrambi di uscita, ciò significa che il numero di canali non cambia tra l'ingresso e l'uscita di questa seconda convoluzione.

            nn.BatchNorm2d(out_channels), # il batch viene applicato due volte in quando è comune applicare la normalizzazione batch dopo ogni convoluzione. ad ogni passo durante l'addestramento calcola la media e la varianza per ciascun canale delle mappe delle caratteristiche e usa questi valori per normalizzare i dati. Successivamente applica due parametri apprendibili gamma e beta per scalare e traslare i dati normalizzati 

            nn.ReLU(inplace=True), #Questa funzione è definita come f(x) 0 max(0,x), ossia sostituisce tutti i valori negativi nella mappa delle caratteristiche con zero. Usando ReLU si aiuta a introdurre non linearità nel modello permettendo alla rete di apprendere relazioni più complesse nei dati.

            # Con la funzione inplace stiamo modificando i dati direttamente nello spazio di memoria sena creare una copia.
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64,128,256,512],
    ):
        super(UNET,self).__init__() # è una chiamata a un metodo della classe genitore (nn.Module) con super() stiamo chiamando il costruttore della classe base nn.Module; in pratica ci stiamo assicurando che tutte le inizializzazioni della classe base vengano eseguite permettendo alla classe UNET di usufruire delle funzionali tà della classe nn.Module

        self.ups = nn.ModuleList() #moduli di upsampling, stiamo creando una istanza di nn.modulelist che è un contenitore pytorch simile a una lsita python ma per moduli nn.Module. Con upsampling aumentiamo progressivamente la dimensione spaziale delle mappe delle caratteristiche , dopo aver raggiunto il fondo dell'architettura U-net. 

        self.downs = nn.ModuleList() # qui andiamo a creare un nn.ModuleList per contenere i moduli di downsampling.
        #Nel percorso di downsampling di una U-net i dati passano attraverso layer che riducono progressivamente le loro dimensioni spaziali aumentando al contempo il numero di canali.

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #qui creiamo un layer di max pooling, componente standard nel downsampling di reti neurali convoluzionali.
        #con il max pooling riduciamo le dimensioni spaziali (altezza e larghezza) delle mappe delle caratteristiche, selezionando il valore massimo in una finestra (kernel) di dimensioni specificate (2x2). Con stride di 2 si indica che la finestra di pool si sposta di due pixel sia in orizzontale che in verticale riducendo la dimensione della pappa delle caratteristiche di un fattore di 2 in entrambe le direzioni. In questo modo si evitano le sovrapposizioni: ogni blocco di 2x2 pixel è indipendente dagli altri.

        # Down part of UNET
        #In questa parte per ogni valore di feature nella lista features creo un blocco DoubleConv, ossia si crea un blocco che esegue due opeazioni convoluzionali sequenziali.
        #Si aggiunge ogni istanza di DoubleConv alla lista dei moduli self.downs usando self.downs.append(..). in questo modo creiamo il percorso di riduzine della U-Net dove ogni livello ha più canali del livello precedente.

        for feature in features:# features è un elenco che deinisce il numero di filtri (o canali ) ce ogni DoubleConv dovrebbe avere Esempio: [64, 128, 256, 512].
            self.downs.append(DoubleConv(in_channels, feature))
            #in ogni iterazione viene aggiunto un nuovo blocco "DoubleConv" alal lista "self.downs". Ogni blocco "DoubleConv" si aspetta un certo numero di canali di ingresso (in_channels) e produce un certo numero di canali di uscita (feature).
            in_channels = feature #il numero di canali di output del blocco attuale diventerà il numero di canali di input per il blocco successivo nel percorso di riduzione.

            #Per filtro, o anche kernel, si intende una matrice piccola ma profonda che scorre sull'immagine in input (o sulla mappa) per produrre una mappa delle caratteristiche
            #In fase di addestramento i filtri sono otimizzati per rilevare caratteristiche utili per il compito a cui la rete è addestrata. 
            #Avendo per esempio 64 filtri in un layer convoluzionale, ciascuno di questi filtri produrrà una mappa delle caratteristiche separata.

        # Up part of UNET
        for feature in reversed(features): #stiamo iterando sulla lista features al contrario ( partiamo dall'elemento con il maggior numero di caratteristiche verso il minor numero).
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2,stride=2,
                ) 
                #Questa linea si riferisce a un'operazione che raddoppia la dimensione spaziale delle mappe di caratteristiche (ad esempio, da 32x32 a 64x64) e riduce il numero di canali della metà (ad esempio, da 512 a 256 se feature fosse 256).

                #layer di convoluzione trasposta chiamata deconvoluzione. Serve per aumentare la dimensione delle mappe delle caratteristiche. In questo caso raddoppia la larghezza e l'altezza delle mappe delle caratteristiche (stride=2) e riduce il numero di canali da feature*2 a feature.
                #feature*2 è il numero di canali in input mentre "feature" è il numero di canali in output con kernel_size=2,stride=2 il layer posiziona il kernel su ogni valore di inpout, generando un output 2x2 per ogni singolo valore. In questo modo si disimballa l'input raddoppiando le dimensioni

                #il motivo per cui l'input ha il numero di canali "feature*2" è che tipicamente in una U-net, le mappe di caratteristiche provenienti dal percorso di riduzione vengono concatenate con quelle del percorso di espansione. Questa concatenzazione raddoppia il numero di canali perchè combina le informazioni di alto livello (profonde) con le informazion di basso livello (spaziali) attraverso connessioni di skip.

                #kernel size è la dimensione del kernel che determina la dimensione dell'area che ogni peso nel filtro guarda per calcolare un singolo valoren nell'output.
            )
            self.ups.append(DoubleConv(feature*2,feature))  #Questo blocco DoubleConv è aggiunto dopo la convoluzione trasposta. A questo punto, si assume che le mappe di caratteristiche siano state già concatenate con quelle dal percorso di downsampling. Quindi, se il layer precedente nel percorso di upsampling produce feature canali, e poi concatena con un altro set di feature canali dal percorso di downsampling, ora si hanno feature*2 canali.

            #DoubleCon lavorerà su questi feature*2 canali in input per rifinire ulteriorrmente le mappe delle di caratteristiche, riducendoli al numero originale "feature".

            self.bottleneck = DoubleConv(features[-1],features[-1]*2) #questa parte riceve l'output della parte di downsampling che è la mappa delle caratteristiche con il maggior numero di canali, stiamo raddoppiando l'ultimo numero di canali generati dalla parte di downsampling.

            self.final_conv = nn.Conv2d(features[0], out_channels,kernel_size=1) #stiamo creando un layer convoluzionale che sarà l'ultimo layer della rete.
            #Con features[0] indichiamo il numero di canalidi di input per questo layer, che corrisponde al pirmo valore della lista features.
            #out_cannels è il numero di canali di output desiderato per la rete, che è tipicamente il numero di classi nella segmentazion eper immagini. Facendo una segmentazione binaria avremo un solo canale di output.

    def forward(self,x):
        skip_connections = [] #lista che salverà le mappe delle caratteristiche da ciascun livello della parte di riduzione che saranno poi usate nella parte di espansione.

        for down in self.downs:
            x = down(x) #applico il livello corrente di downsampling all'input x.
            skip_connections.append(x) #dopo ogni livello di downsampling, salva l'output corrente in skip_connections per un uso successivo.
            x = self.pool(x) #applica l'operazione di pooling all'outpu del livello di downsampling per ridurre ulteriormente le sue dimensioni. Il pooling consiste nel concentrarsi su caratteristiche più ampie e meno dettagliate.
        
        x = self.bottleneck(x) #parte della rete con risoluzione più bassa
        skip_connections = skip_connections[::-1] # in questo modo l'ultimo livello di downsampling è il primo elemento in skip_connections, il penultimo il secondo elemento e così via.

        #Stiamo invertendo l'ordine delle connessioni di skip per allinearle con i passaggi di upsampling corrispondenti 
        
        for idx in range(0, len(self.ups),2): #itera attraverso i moduli di upsampling in self.ups, stiamo applicando il livello di convoluzione trasposta ( o deconvoluzione) per iniziare l'upsampling dell'input x. idx è un indice che aumenta di 2 ad ogni iterazione. Questo perchè in self.ups si hanno sia i layer di convoluzione trasposta sia i blocchi doubleconv che processano ulteriormente i dati. 

        #Quindi ogni due elementi in self.ups corrispondono a un singolo passo di upsampling.

            x = self.ups[idx](x) #layer di convoluzione traposta che aumenta le dimensioni spaziali della mappa delle caratteristiche
            skip_connection = skip_connections[idx//2] # recuperiamo la mappa delle caratteristiche corrispondente al downsampling. In questo modo combiniamo le informazioni di dettglio perse durante il downsampling con le mappe delle caratteristiche che vengono ricostruite durante l'upsampling.
            #Queste mappe sono in ordine inverso rispetto a come devno essere utilzizate nell'upsampling.
            #si fa idx//2 per ottenere l'indice completo di upsampling (che comprende due elementi in "self.ups") c'è solo una connessione saltata corrispondente.
            #in questo modo vengono sincronizzati correttamente gli elementi nella lista self.ups con le relative connessioni saltate in "skip_connections", garantendo che ogni passaggio di upsamplig utilizzi la mappa delle caratteristiche corretta dal downsamplig.

            if x.shape != skip_connection.shape:
                x = tf.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x),dim=1)
            #torch.cat è usato per concatenare la mappa delle caratteristiche dall'upsampling ('x') con la corrispondente mappa delle caratteristiche salvata durante il downsampling ('skip_connection'). dim=1 indica che la concatenazione avviene lungo l'asse dei canali. Ossia le informazioni dettagliate salvate durante il downsampling vengono combinate con le informazioni più ampie che vengono ricostruite durante l'upsampling. In questo modo reintroduciamo dettagli locali importanti per una segmentazione precisa.
            x = self.ups[idx+1](concat_skip) # applica il prossimo livello di doubleconv sul risultato della concatenzazione per elaborare ulteriormente la mappa delle caratteristiche combinata. fondiamo le informazioni in modo che la mappa delle caratteristiche risultante sia informativa e dettagliata.

        return self.final_conv(x)
    
def test():
    x = torch.randn((3,1,161,161)) # 3 rappresenta la dimensione del batch, significa che si sta generando un batch di 3 immagini ognuna con 1 canale (scala di grigi) e dimensione di 160x160 pixel.
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
    
# In un'architettura di rete neurale come la U-Net, ogni dimensione del tensore di input rappresenta:

#-Dimensione del batch: numero di immagini o dati che si stanno passando attraverso la rete in una sola volta La dimensione del batch è la prima dimensione del tensore di inout quando lo si definisce inPytorch. 
#-canali: numero di canali in ogni immagine di input (per esempio, 3 per immagini RGB o 1 per immagini in scala di grigi).
#-altezza e larghezza: le dimensioni spaziali dell'immagine in pixel.








