import cv2
import pandas as pd
import random
import os
from sklearn import svm
import numpy as np
import joblib
from sklearn.model_selection import GridSearchCV

NUM_BB = 200    #definiamo il numero di bounding box per immagine negativa
hog = cv2.HOGDescriptor()

#carichiamo il positive set, effettuando il preprocessing alle immagine e calcolando gli hog descriptor 
#sulla base di quanto annotato nel file splitPositive
def loadPositiveSet():
    filesTrain = pd.read_csv("./split_positive/train_assignment.txt", dtype = str, header=None)

    hogPositiveSet = []
    label = []
    for row in range(filesTrain.shape[0]): #itero i file di train 
        imageName = filesTrain.iat[row,0]+ ".jpg"
        image = cv2.imread(f"./WiderPerson/Images/{imageName}")

        if image is not None:
            boundyBox = pd.read_csv(f"./WiderPerson/Annotations/{imageName}.txt")
            image = image_preprocessing(image)
            for indexBoundyBox in range(boundyBox.shape[0]): #itero le bb che ho per quell'immagine
                rowbb = boundyBox.iat[indexBoundyBox, 0]
                if rowbb.split()[0] == "1": #se come etichetta trovo 1 (pedestrian) prendo la riga
                    items = rowbb[2:].split()

                    items = [int(item) for item in items]

                    start = (items[0], items[3])
                    end = (items[2], items[1])
                    
                    croppedImage = crop(image, start, end)
                    resizedImage = cv2.resize(croppedImage, (64, 128))
                    hogPositiveSet.append(hog.compute(resizedImage, (64, 128)))
                    label.append(1)
    hogPositiveSet = np.array(hogPositiveSet)
    return hogPositiveSet, label #ritorno una lista contenente tutti gli hog positivi e una lista contenente tutte le etichette


#carichiamo il negative set, effettuando anche in questo caso il preprocessing delle immagini. Poiché non esistono 
#delle bounding box già annotate in qualche file, abbiamo generato in modo casuale 4 punti per individuare i crop negativi
def loadNegativeSet():
    directory = "./negative/train_neg"
    
    hogNegativeSet = []
    label = []

    for file in os.listdir(directory):
        image = cv2.imread(f"{directory}/{file}")
        if image is not None:
            image = image_preprocessing(image)
            h, w = image.shape
            for i in range (NUM_BB):

                x_start = random.randint(0, w - int(3*w/100)-1) #Calcolo randomicamente la coordinata x_start per la creazione di una BB nel set negative
                y_start = random.randint(0, h - int(3*h/100)-1) #Calcolo randomicamente la coordinata y_start per la creazione di una BB nel set negative
                
                x_end = random.randint(x_start + int(3*w/100), w) #Calcolo randomicamente la coordinata x_end per la creazione di una BB nel set negative, discostandola del 3% dalla coordinata x_start
                y_end = random.randint(y_start + int(3*h/100), h) #Calcolo randomicamente la coordinata y_end per la creazione di una BB nel set negative, discostandola del 3% dalla coordinata y_start

                croppedImage = crop(image, (x_start, y_end), (x_end, y_start)) #Ritaglio l'immagine
                resizedImage = cv2.resize(croppedImage, (64, 128)) #Ridimensiono l'immagine
                hogNegativeSet.append(hog.compute(resizedImage, (64, 128)))
                label.append(-1)
    hogNegativeSet = np.array(hogNegativeSet)
    return hogNegativeSet, label #ritorno una lista contenente tutti gli hog negative e una lista contenente tutte le etichette

#carichiamo il validation set, effettuando la classificazione su ciasuna immagine ed effettuando il non maxima suppression
def loadValidationSet(svm, t_val):
    images = {}
    filesValidation = pd.read_csv("./split_positive/val_assignment.txt", dtype = str, header=None)

    for row in range(filesValidation.shape[0]):#itero i file di train
        imageName = filesValidation.iat[row,0]+ ".jpg"

        image = cv2.imread(f"./WiderPerson/Images/{imageName}")

        images[imageName] = nms(detMultiScale(image,svm), t_val)
    
    return images #restituiamo un dizionario la cui chiave è il nome dell'immagine mentre i valori sono le liste di tutte 
                  #le bounding box individuate dal nostro classificatore

#carichiamo le bounding box annotate nel dataset per una determinata immagine di test per poi confrontarle con quelle 
#individuate dal nostro classificatore
def loadTest(imgName):
    listBB = []
    boundyBox = pd.read_csv(f"./WiderPerson/Annotations/{imgName}.txt")
    for indexBoundyBox in range(boundyBox.shape[0]):#itero le bb che ho per quell'immagine
        rowbb = boundyBox.iat[indexBoundyBox, 0]
        if rowbb.split()[0] == "1":
            items = rowbb[2:].split()
            x0 = int(items[0])
            y0 = int(items[1])

            x1 = int(items[2])
            y1 = int(items[3])

            listBB.append((x0, y0, x1, y1))
    return listBB


def crop(image, start_point, end_point):
        x1, y1 = start_point
        x2, y2 = end_point
        return image[y2:y1, x1:x2]

#partendo dal positive set e dal negative set insieme alle relative liste contenenti le etichette, creiamo il trainSet unificando il tutto
def createTrainSet(positiveSet, labelPositive, negativeSet, labelNegative):
    trainSet = np.concatenate([positiveSet, negativeSet])
    label = np.concatenate([labelPositive, labelNegative])
    return trainSet, label

#addestriamo il nostro modello stimando il miglior iperparametro di C per la nostra SVM
#una volta addestrato il modello lo salviamo per utilizzi futuri
def train(trainSet, label):
    param_grid = {'C': [0.01, 0.1, 1, 10]}
    svm_classifier = svm.LinearSVC(dual=False)

    grid_search = GridSearchCV(svm_classifier, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(trainSet, label)
    c = grid_search.best_params_['C']
    print("Miglior parametro C:", c)

    best_svm_classifier = grid_search.best_estimator_
    joblib.dump(best_svm_classifier, 'svm_model.pkl')
    return best_svm_classifier

def image_preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

#eseguiamo la detection dei pedoni nelle nostre immagini. Ciò viene fatto su più scale permettendo individuare sia pedoni grandi che piccoli
def detMultiScale(img,svm):
    img = image_preprocessing(img)

    scale = [0.7, 1, 1.3]
    window_h = 128
    window_w = 64
    detections = []

    for k in scale:
        imge = cv2.resize(img,None,fx=k, fy=k,interpolation=cv2.INTER_LINEAR)
        img_height, img_width = imge.shape[:2]
        for y in range(0, img_height-window_h, 10):
            for x in range(0, img_width-window_w, 10):
                sliding_window = imge[y:y+window_h,x:x+window_w]
                hog_window = np.array(hog.compute(sliding_window, (64, 128))).reshape(1, -1)

                predict = svm.predict(hog_window)
                confidence = svm.decision_function(hog_window)

                #aggiungiamo la bouding box trovata soltanto se la classe assegnata dal classificatore è pari ad 1 (pedestrian) e 
                #se la sua confidence è maggiore di un determinato valore
                if predict == 1 and confidence > 1.3:
                    boundingBoxe = (int(x/k), int(y/k), int((x + window_w)/k), int((y + window_h)/k), predict, confidence)
                    detections.append(boundingBoxe)
    
    return detections

#determiniamo la percentuale di sovrapposizione tra due bounding box
def calculate_iou(bb1, bb2):     
    ax, ay, bx, by, _, _ = bb1
    cx, cy, dx, dy, _, _ = bb2

    xMax = max(ax, cx)
    yMax = max(ay, cy)
    xMin = min(bx, dx)
    yMin = min(by, dy)

    inter_width = max(0, xMin - xMax + 1)
    inter_height = max(0, yMin - yMax + 1)

    interArea = inter_width * inter_height

    areaBB1 = (bx - ax + 1) * (by - ay + 1)
    areaBB2 = (dx - cx + 1) * (dy - cy + 1)

    unionArea = areaBB1 + areaBB2 - interArea

    if unionArea > 0:
        iou = interArea / float(unionArea)
    else:
        iou = 0.0

    return iou

#tramite la non maxima suppression rimuoviamo tutte quelle bounding box la cui percentuale di 
#sovrapposizione è maggiore di una determinata soglia
def nms(P, iou_threshold):

    P_sort = sorted(P, key=lambda x: x[5], reverse=True)
    F = []
    while P_sort:
        remove = []
        S = P_sort.pop()
        F.append(S)
        for T in P_sort:
            iou = calculate_iou(S,T)
            if iou > iou_threshold:
                remove.append(T)
        P_sort = [x for x in P_sort if x not in remove]
    return F

#per ciascuna immagine confrontiamo le bounding box individuate dal nostro modello con le bounding box annotate nel relativo file di testo.
#basandosi sulla percentuale di sovrapposizione tra le nostre bounding box e quelle annotate, determiniamo se siamo in presenza di TP.
#per determinare le FN prendiamo tutte le bounding box elencate in annotation che non sono state individuate dal nostro modello.
#indine per determinare FP consideriamo tutte le bounding box che sono state individuate dal nostro modello ma che non sono annotate nell'apposito file di testo
def calculateTP(images):
    TP = 0
    FN = 0
    FP = 0

    for imgName in images.keys():
        listBoundingBoxValidationSet = []
        boundyBox = pd.read_csv(f"./WiderPerson/Annotations/{imgName}.txt")
        for indexBoundyBox in range(boundyBox.shape[0]):#itero le bb che ho per quell'immagine
            rowbb = boundyBox.iat[indexBoundyBox, 0]
            if rowbb.split()[0] == "1":#se come etichetta trovo 1 (pedestrian) prendo la riga

                x1, y1, x2, y2 = rowbb[2:].split()
                bbValidation = (int(x1), int(y1), int(x2), int(y2), -1, -1)
                listBoundingBoxValidationSet.append(bbValidation)
                
                for bbCalculated in images.get(imgName):
                    if (calculate_iou(bbCalculated, bbValidation)) > 0.5:
                        TP += 1
                        images.get(imgName).remove(bbCalculated)
                        listBoundingBoxValidationSet.remove(bbValidation)
                        break
                
        if listBoundingBoxValidationSet: 
            FN += len(listBoundingBoxValidationSet)

        if images.get(imgName):
            FP += len(images.get(imgName))
    return TP, FP, FN

#applichiamo la formula dell'F1Score per determinare l'accuratezza del nostro modello
def calculateF1Score(values):
    TP, FP, FN = values
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return F1


if __name__ == "__main__":
    #dovendo stimare il miglior valore di t (soglia di sovrapposizione utilizzata in detMultiScale) ci serviamo della variabile 
    #best_t che nel corso del main verrà assegnata nel modo migliore
    best_t = None

    #effettuiamo una verifica controllando se il modello è già stato addestrato o se siamo alla prima esecuzione dell'algoritmo
    #nel primo caso ci occupiamo di caricare il modello già addestrato per poi effettuare la classificazione delle immagini di test
    #nel secondo caso invochiamo tutte le funzione definite sopra costruendo il trainSet ed effettuando l'addestramento
    if os.path.exists("svm_model.pkl"):
        svm_classifier = joblib.load("svm_model.pkl")
        best_t = joblib.load("threshold.pkl")
    else:
        hogPositiveSet, labelPositive = loadPositiveSet()
        hogNegativeSet, labelNegative = loadNegativeSet()
        trainSet, label = createTrainSet(hogPositiveSet, labelPositive, hogNegativeSet, labelNegative)
        svm_classifier = train(trainSet, label)

        #qui iniziamo la fase di validation in cui confrontiamo l'F1Score ottenuto al variare di t. 
        #Alla fine di questa fase verrà determinato il valore di t migliore con i relativi TP, FN, FP,
        #recall, precision ed infine F1Score
        t_values = [0.1, 0.35, 0.5, 0.75]
        best_score = 0
        values = (0, 0, 0)
        best_values = ()

        for t_val in t_values:
            images = loadValidationSet(svm_classifier, t_val)
            values = calculateTP(images)
            F1 = calculateF1Score(values)
        
            if F1 >= best_score:
                best_score = F1
                best_t = t_val
                best_values = values

        TP, FP, FN = best_values
        print(f"best threshold = {best_t}")
        print(f"F1 score = {best_score}")
        print(f"TP = {TP}, FP = {FP}, FN = {FN}")

        joblib.dump(values, "TP_FP_FN.pkl")
        joblib.dump(best_t, "threshold.pkl")

    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    imgName = "000059.jpg"
    imgTestPath = f"./WiderPerson/Images/{imgName}"
    imageTest = cv2.imread(imgTestPath)

    #una volta selezionata un'immagine dal testSet ricaviamo le bounding box con il nostro algoritmo, tramite il 
    #peopleDetector già presente in openCV e otteniamo le bounding box già annotate nell'apposito file di testo
    ourDetections = nms(detMultiScale(imageTest, svm_classifier), best_t)
    cv2Detections, _ = hog.detectMultiScale(imageTest)
    annotatedDetections = loadTest(imgName)

    #mostriamo i tre insiemi di bounding box sull'immagine precedentemente caricata
    for (x, y, x1, y1, _, _) in ourDetections:
        cv2.rectangle(imageTest, (x, y), (x1, y1), (255, 0, 0), 2)

    for (x, y, w, h) in cv2Detections:
        cv2.rectangle(imageTest, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, x1, y1) in annotatedDetections:
        cv2.rectangle(imageTest, (x, y), (x1, y1), (0, 0, 255), 2)
    
    cv2.imshow("Immagine", imageTest)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
