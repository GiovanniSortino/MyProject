import os
import threading
import  torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import scipy.optimize   

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, mappaFeatures):
    plt.clf()
    if not plt.fignum_exists(1):  # Controlla se la figura esiste già
        plt.figure(figsize=(16,10))

    plt.imshow(pil_img)
    ax = plt.gca() 
        
    for key in mappaFeatures.keys():
        if mappaFeatures[key].get_exitFlag() is False and mappaFeatures[key].get_entryFlag() is False:
            xmin, ymin, xmax, ymax = mappaFeatures[key].get_boundingbox().detach().numpy()

            xmin = int(xmin)
            xmax = int(xmax)
            ymin = int(ymin)
            ymax = int(ymax)

            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
            ax.text(xmin, ymin, key, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.draw()  # Ridisegna la finestra
    plt.show(block=True) 

def detect(model, im, transform = None, threshold_confidence = 0.7):
    if transform is None:
        # standard PyTorch mean-std input image normalization
        transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img = transform(im).unsqueeze(0)

    # demo model only support by default images with aspect ratio between 0.5 and 2
    # if you want to use images with an aspect ratio outside this range
    # rescale your image so that the maximum size is at most 1333 for best results
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'

    # propagate through the model
    outputs = model(img)

    # keep only predictions with a confidence > threshold_confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    person_class_index = CLASSES.index('person')
    keep = (probas[:, person_class_index] > threshold_confidence)

    # Convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def loadPathValidationSet(pathDirectory):
    return [loadPathVideo(os.path.join(pathDirectory, directory)) for directory in os.listdir(pathDirectory)]

def loadPathVideo(videoDirectory):
    imageValidationPath = os.path.join(videoDirectory, 'img1')
    return [os.path.join(imageValidationPath, foto) for foto in os.listdir(imageValidationPath)]
    
def cropImage(bboxes_scaledList, frame):
    x1, y1, x2, y2 = bboxes_scaledList
    x1, y1, x2, y2 = abs(int(x1)), abs(int(y1)), abs(int(x2)), abs(int(y2)) #faccio abs perché le bb alcune volte sono negative
    imageCrop = frame[y1:y2, x1:x2]
    imageCrop = cv2.resize(imageCrop,(224,224))    
    return imageCrop

def createModel():
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output) 
    return model

def preprocessing(image):
    image = np.expand_dims(image, axis=0) 
    preprocessed_image = preprocess_input(image)
    return preprocessed_image

def get_features(model, preprocessed_images):
    features = model(preprocessed_images)
    return features

def createMatriceCosti(oldMap, actualMap, vgg16, pathFrameActual): 
    n_rows = len(oldMap.keys())
    n_cols = len(actualMap.keys())
    matriceCosti = np.zeros((n_rows, n_cols))
    row = 0
    col = 0

    listKeyOld = oldMap.keys()
    listKeyActual = actualMap.keys()
    
    listaFeatureActual = []

    frameActual = np.array(Image.open(pathFrameActual))
    
    #Calcolo le features attuali
    for idActual in listKeyActual: 
        bb = actualMap[idActual].get_boundingbox()
        featureActual = None
        if bb is not None:
            imageCroped = cropImage(bb, frameActual)
            preprocessed_image = preprocessing(imageCroped)
            featureActual = get_features(vgg16, preprocessed_image)
            actualMap[idActual].set_features(featureActual) 
        else:
            featureActual = actualMap[idActual].get_features()
        listaFeatureActual.append(featureActual)

    #Calcolo le features vecchie e calcolo la distanza con tutte quelle attuali
    for idOld in listKeyOld:
        featureOld = oldMap[idOld].get_features()

        for featureActual in listaFeatureActual:           
            distanza = 1 - cosine_similarity(featureOld,featureActual)
            matriceCosti[row][col] = distanza
            col += 1
        row += 1
        col = 0   

    return matriceCosti

def cosine_similarity(tensor1, tensor2):
    if isinstance(tensor1, tf.Tensor):
        tensor1 = torch.tensor(tensor1.numpy())
    if isinstance(tensor2, tf.Tensor):
        tensor2 = torch.tensor(tensor2.numpy())

    dot_product = torch.sum(tensor1 * tensor2)
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)
    cosine_sim = dot_product / (norm_tensor1 * norm_tensor2)
    return cosine_sim

def createPersonActualMap(bboxes_scaled):
    personsActualMap = {}
    for id in range(len(bboxes_scaled)):                
        personsActualMap[id] = Person(bboxes_scaled[id])     
    return personsActualMap 

def getNewEntry(idsActualKey, idsOldKey):
    return list(set(idsActualKey).difference(idsOldKey))

def getExit(idsOldKey, idSelectedOld):
    return list(set(idsOldKey).difference(idSelectedOld))

def hungarian(matriceDeiCosti, personsOldMap, personsActualMap, threshold):
    idsActualKey = list(personsActualMap.keys()) 
    idsOldKey = list(personsOldMap.keys())

    oldRowHungarian, actualColumnHungarian = scipy.optimize.linear_sum_assignment(matriceDeiCosti)
    selectedOld = []
    selectedActual = []

    for i in range(len(oldRowHungarian)):
        row = oldRowHungarian[i]
        col = actualColumnHungarian[i]

        selectedOld.append(idsOldKey[row])
        selectedActual.append(idsActualKey[col])
        if matriceDeiCosti[row][col] >= threshold:
            personsActualMap[idsActualKey[col]].set_overThershold(True)
            print(f"Sono stati associati due persone diverse o/a {idsOldKey[row]}-{idsActualKey[col]} matrix r/c:{row,col}, Valore: {matriceDeiCosti[row][col]}")

    print("###################################################################################################################")
    print("Lista Selected Actual in Hungarian ",selectedActual)
    print("Lista Selected Old in Hungarian ",selectedOld)
    print("###################################################################################################################")
    return len(selectedActual), selectedOld, selectedActual

def main():
    validationSetPath =""

    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    validationSet = loadPathValidationSet(validationSetPath)
    vgg16 = createModel()    
    N = 5
    threshold = 0.65

    for video in validationSet:
        firstTime = True
        idMax = 0

        parts = video[0].split("\\")
        videoName = parts[-3]
        personsOldMap = {}

        exitFrameNumber = 1

        for pathFrameActual in video:
            personsActualMap = {}
            frameActual = Image.open(pathFrameActual)
            _, bboxes_scaled = detect(model, frameActual)
            personsActualMap = createPersonActualMap(bboxes_scaled)    

            if not firstTime:
                print("----------------------------------------------------Sono nell'if----------------------------------------------------")
                print("frame attuale: ", pathFrameActual)
                idsActualKey = list(personsActualMap.keys()) 
                idsOldKey = list(personsOldMap.keys())
                matriceDeiCosti = createMatriceCosti(personsOldMap, personsActualMap, vgg16, pathFrameActual)
                n_associazioni, idSelectedOld, idSelectedActual = hungarian(matriceDeiCosti,personsOldMap,personsActualMap, threshold)
                mappaMomentanea = {}

                new_entry = getNewEntry(idsActualKey, idSelectedActual)
                exit = getExit(idsOldKey,idSelectedOld)

                #GESTIONE ASSEGNAMENTO
                for i in range(n_associazioni):
                    keyActual = idSelectedActual[i]
                    keyOld = idSelectedOld[i]
                    if personsActualMap[keyActual].get_overThershold():
                        mappaMomentanea[keyOld] = personsOldMap[keyOld].set_exitFlag(True).increase_exitFrameNumber()
                        mappaMomentanea[idMax] = personsActualMap[keyActual].set_entryFlag(True).increase_entryFlag()
                        print("Ho aggiunto un nuovo id (doppia associazione)(split per via della soglia): ", idMax, keyOld)
                        idMax += 1
                    else:
                        n_frame = personsOldMap[keyOld].get_entryFrameNumber()
                        mappaMomentanea[keyOld] = personsActualMap[keyActual].set_entryFlag(True).set_entryFrameNumber(n_frame + 1)
                        print("ENTRO: ", keyOld, mappaMomentanea[keyOld].get_entryFrameNumber())
                        if mappaMomentanea[keyOld].get_entryFrameNumber() >= N:
                            mappaMomentanea[keyOld].set_entryFlag(False)
                            
                print("MappaMomentanea GESTIONE ASSEGNAMENTO", list(mappaMomentanea.keys()))
                
                #GESTIONE INGRESSO
                print("bb non ancora associte (ingressi): ", list(new_entry))
                while new_entry:
                    element = new_entry.pop()
                    mappaMomentanea[idMax] = personsActualMap[element].set_entryFlag(True).increase_entryFlag()
                    print(f"ho associato ",idMax, element, personsActualMap[element])
                    idMax += 1
                print("MappaMomentanea GESTIONE INGRESSO", list(mappaMomentanea.keys()))

                #GESTIONE USCITA
                print("elementi non trovati: ", list(exit))
                while exit:
                    id = exit.pop()
                    if personsOldMap[id].get_entryFlag() :
                        print("L'elemento stava per entrare ma è stato soppresso: ",id)
                    else:
                        mappaMomentanea[id] = personsOldMap[id]
                        mappaMomentanea[id].set_exitFlag(True).increase_exitFrameNumber()
                        print(f"USCITA id: {id} numero di frame: {mappaMomentanea[id].get_exitFrameNumber()}")
                    
                #GESTIONE RIMOZIONE DA MAPPA
                daCancellare = [id for id in mappaMomentanea.keys() if mappaMomentanea[id].get_exitFrameNumber() >= N]
                print("Elementi da cancellare: ", daCancellare)
                for id in daCancellare:
                    mappaMomentanea.pop(id)
                    print("RIMOSSO DALLA MAPPA: ", id)
                print("MappaMomentanea GESTIONE RIMOZIONE", list(mappaMomentanea.keys()))
            
                print("id frame precedente ", list(personsOldMap.keys()), len(list(personsOldMap.keys())))
                print("id frame attuale ", list(mappaMomentanea.keys()), len(list(mappaMomentanea.keys())))

                personsOldMap = mappaMomentanea.copy() 
            else:
                print("---------------------------------------------------Sono nell'else---------------------------------------------------")
                for id in personsActualMap.keys():
                    bb = personsActualMap[id].get_boundingbox()
                    imageCroped = cropImage(bb, np.array(frameActual))
                    preprocessed_image = preprocessing(imageCroped)
                    personsActualMap[id].set_features(get_features(vgg16, preprocessed_image)).set_entryFlag(True).increase_entryFlag()
                personsOldMap = personsActualMap.copy()
                print("id frame attuale ", list(personsOldMap.keys()), len(list(personsOldMap.keys())))
                firstTime = False
                idMax = len(bboxes_scaled)
            
            # with open(f'{videoName}.txt', 'a') as predFile:
            #     for id in personsOldMap.keys():
            #         person = personsOldMap[id]
            #         if(person.get_entryFlag() is False and person.get_exitFlag() is False):
            #             bbox = person.get_boundingbox()
            #             if bbox is not None:
            #                 x1, y1, x2, y2 = bbox

            #                 bb_width = abs(x2 - x1)
            #                 bb_height = abs(y2 - y1)

            #                 predFile.write(f"{exitFrameNumber},{id},{x1},{y1},{bb_width},{bb_height},1\n")         
            # exitFrameNumber += 1
        
            my_thread = threading.Thread(target=plot_results, args=(frameActual, personsOldMap))
            my_thread.start()

class Person:
    def __init__(self, boundingbox, features = None, exitFlag = False, exitFrameNumber = 0, overThresholdFlag = False, entryFlag = False, entryFrameNumber = 0):
        self._boundingbox = boundingbox
        self._features = features
        self._exitFlag = exitFlag
        self._exitFrameNumber = exitFrameNumber
        self.overThresholdFlag = overThresholdFlag
        self.entryFlag = entryFlag
        self.entryFrameNumber = entryFrameNumber

    def __repr__(self):
        return (f"Person(boundingbox={self._boundingbox}, "
                f"exitFlag={self._exitFlag}, "
                f"exitFlag={self.overThresholdFlag}, "
                f"entryFlag={self.entryFlag}, "
                f"exitFrameNumber={self._exitFrameNumber})")

    def get_boundingbox(self):
        return self._boundingbox

    def set_boundingbox(self, boundingbox):
        self._boundingbox = boundingbox
        return self

    def get_features(self):
        return self._features

    def set_features(self, features):
        self._features = features
        return self

    def get_exitFlag(self):
        return self._exitFlag

    def set_exitFlag(self, exitFlag):
        self._exitFlag = exitFlag
        return self

    def get_exitFrameNumber(self):
        return self._exitFrameNumber

    def set_exitFrameNumber(self, exitFrameNumber):
        self._exitFrameNumber = exitFrameNumber
        return self

    def get_overThershold(self):
        return self.overThresholdFlag

    def set_overThershold(self, overThreshold):
        self.overThresholdFlag = overThreshold
        return self
    
    def get_entryFlag(self):
        return self.entryFlag
    
    def set_entryFlag(self, entryFlag):
        self.entryFlag = entryFlag
        return self
    
    def get_entryFrameNumber(self):
        return self.entryFrameNumber
    
    def set_entryFrameNumber(self, entryFrameNumber):
        self.entryFrameNumber = entryFrameNumber
        return self

    def increase_exitFrameNumber(self):
        self._exitFrameNumber += 1
        return self
    
    def increase_entryFlag(self):
        self.entryFrameNumber += 1
        return self

if __name__ == '__main__':
    main()