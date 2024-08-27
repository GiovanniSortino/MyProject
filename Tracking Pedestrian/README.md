Questo progetto implementa un sistema per il tracciamento di persone in video utilizzando tecniche di deep learning e computer vision. Il codice sfrutta modelli pre-addestrati per rilevare oggetti nelle immagini, estrarre caratteristiche visive, e assegnare identità univoche alle persone rilevate nel video, anche in presenza di entrate e uscite dal frame.

Funzionalità principali

1. Rilevamento e tracciamento oggetti:
Utilizza il modello pre-addestrato DETR (Detection Transformer) per rilevare le persone nei frame video.
Le bounding box rilevate vengono normalizzate e ridimensionate in base alla dimensione dell'immagine per garantire una corretta visualizzazione.

2. Estrazione delle caratteristiche:
Il modello VGG16, pre-addestrato su ImageNet, viene utilizzato per estrarre le feature delle immagini delle persone rilevate.
Le feature estratte sono utilizzate per calcolare la similarità coseno tra le persone rilevate in frame consecutivi.

3. Algoritmo di assegnamento:
Viene utilizzato l'algoritmo di assegnamento ungherese per associare le persone rilevate in frame consecutivi in base alla similarità delle feature.
Il sistema gestisce entrate e uscite delle persone dal frame, assegnando nuove identità a persone non precedentemente tracciate e rimuovendo quelle uscite dal video.

4. Visualizzazione dei risultati:
Le bounding box delle persone rilevate vengono visualizzate su ogni frame con un'etichetta identificativa. Viene utilizzata la libreria matplotlib per la visualizzazione.

Il progetto può sicuramente essere migliorato evitando molti id switch ed evitare l'assegnamento degli'id cosi alti

----------------------------------------------------------------------------------------------------------------------------------------------------------
This project implements a system for tracking people in videos using deep learning and computer vision techniques. The code uses pre-trained models to detect objects in images, extract visual features, and assign unique identities to people detected in the video, even if they enter and exit the frame.

Main features 
1. Object detection and tracking: It uses the pre-trained DETR (Detection Transformer) model to detect people in video frames.
The detected bounding boxes are normalized and scaled based on the image size to ensure correct display.

2. Feature extraction: The VGG16 model, pre-trained on ImageNet, is used to extract image features of the detected people.
The extracted features are used to calculate the cosine similarity between the detected people in consecutive frames.

3. Assignment algorithm: The Hungarian assignment algorithm is used to associate the detected people in consecutive frames based on the similarity of the features.
The system manages people entering and leaving the frame, assigning new identities to people not previously tracked and removing those who have left the video.

4. Visualization of results: The bounding boxes of the detected people are protected on each frame with an identifying label. The matplotlib library is used for visualization.

The project can definitely be improved by avoiding many id switches and avoiding assigning such high id's.
