Questo progetto implementa un rilevatore di pedoni basato su un modello SVM (Support Vector Machine) addestrato utilizzando descrittori HOG (Histogram of Oriented Gradients). Il codice carica un dataset di immagini positive e negative, effettua il pre-processing delle immagini e calcola i descrittori HOG per il training del modello. Di seguito sono descritti i principali componenti e funzionalità del codice:

Funzionalità Principali

1. Caricamento del Positive Set (loadPositiveSet):
Carica il set di immagini positive (pedoni) e applica il pre-processing. Utilizza le annotazioni presenti nel dataset per ritagliare e ridimensionare le regioni di interesse, calcolando infine i descrittori HOG.

2. Caricamento del Negative Set (loadNegativeSet):
Carica il set di immagini negative (non pedoni) e genera in modo casuale delle bounding box per ritagliare le regioni non pedonali. Anche in questo caso, vengono calcolati i descrittori HOG per l'addestramento.

3. Validazione del Modello (loadValidationSet):
Applica il modello SVM addestrato su un set di validazione, eseguendo una ricerca su diverse scale per rilevare pedoni di varie dimensioni. Utilizza la Non-Maxima Suppression (NMS) per rimuovere le bounding box ridondanti.

4. Addestramento del Modello (train):
Addestra un modello SVM lineare utilizzando un approccio Grid Search per trovare il miglior iperparametro C. Il modello addestrato viene salvato per utilizzi futuri.

5. Calcolo dell'IoU (Intersection over Union) (calculate_iou):
Calcola la sovrapposizione tra due bounding box per determinare l'efficacia del rilevatore nel confronto con le annotazioni.

6. Non-Maxima Suppression (nms):
Riduce le false rilevazioni eliminando le bounding box sovrapposte che rappresentano la stessa entità.

7. Calcolo delle Metriche di Performance (calculateTP, calculateF1Score):
Confronta le bounding box rilevate con quelle annotate per calcolare True Positives (TP), False Positives (FP), e False Negatives (FN), determinando infine il punteggio F1 per valutare la precisione del modello.

8. Visualizzazione dei Risultati:
Confronta e visualizza le bounding box individuate dal modello SVM, dall'algoritmo di rilevamento di persone predefinito di OpenCV, e le annotazioni originali su un'immagine di test.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project implements a pedestrian detector based on an SVM (Support Vector Machine) model trained using HOG (Histogram of Oriented Gradients) descriptors. The code loads a dataset of positive and negative images, preprocesses the images, and calculates the HOG descriptors for training the model. Below is a description of the main components and functionalities of the code:

Main Features

1. Loading the Positive Set (loadPositiveSet):
Loads the set of positive images (pedestrians) and applies preprocessing. It uses the annotations in the dataset to crop and resize the regions of interest, then calculates the HOG descriptors.

2. Loading the Negative Set (loadNegativeSet):
Loads the set of negative images (non-pedestrians) and randomly generates bounding boxes to crop non-pedestrian regions. HOG descriptors are also calculated for training in this case.

3. Model Validation (loadValidationSet):
Applies the trained SVM model on a validation set, performing a multi-scale search to detect pedestrians of various sizes. Non-Maxima Suppression (NMS) is used to remove redundant bounding boxes.

4. Model Training (train):
Trains a linear SVM model using a Grid Search approach to find the best hyperparameter C. The trained model is saved for future use.

5. Calculating IoU (Intersection over Union) (calculate_iou):
Calculates the overlap between two bounding boxes to determine the detector's effectiveness compared to the annotations.

6. Non-Maxima Suppression (nms):
Reduces false detections by eliminating overlapping bounding boxes that represent the same entity.

7. Performance Metrics Calculation (calculateTP, calculateF1Score):
Compares the detected bounding boxes with the annotated ones to calculate True Positives (TP), False Positives (FP), and False Negatives (FN), ultimately determining the F1 score to evaluate the model's precision.

8. Results Visualization:
Compares and visualizes the bounding boxes detected by the SVM model, the default OpenCV people detection algorithm, and the original annotations on a test image.
