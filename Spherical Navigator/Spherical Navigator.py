import cv2
import numpy as np

#questa funzione si occupa di inserire tutti gli elementi grafici nell'immagine di output. In particolare gestisce i tasti play e pause
#e la barra di progressione del video associata con il tempo trascorso.
def draw_interface(image, is_pause, num_frames, actual_frame, fps, total_minute, total_second):
    line_y = image.shape[0] - 70 
    cv2.line(image, (20, line_y), (image.shape[1] - 20, line_y), (255, 255, 255), 2)

    progress = actual_frame / num_frames #restituisce un valore tra 0 e 1 che indica l'avanzamento del video
    progress_x = int(progress * (image.shape[1] - 20) + 10)
    cv2.circle(image, (progress_x, line_y), 7, (255, 255, 255), -1)

    center = (int(image.shape[1] / 2), int(image.shape[0] - 40))

    #gestione avanzamento pallina
    elapsed_time = int(actual_frame / fps) #calcola il tempo trascorso in secondi. divide il numero di frame per i frame al secondo
    elapsed_minutes = elapsed_time // 60 
    elapsed_seconds = elapsed_time % 60
    cv2.putText(image, f"{elapsed_minutes:02d}:{elapsed_seconds:02d}", (20, line_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) 
    cv2.putText(image, f"{total_minute:02d}:{total_second:02d}", (image.shape[1] - 65, line_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #gestione simboli play/pause
    if is_pause:
        triangle_size = 10 
        play_points = np.array([(center[0] - triangle_size, center[1] - triangle_size), 
                                (center[0] - triangle_size, center[1] + triangle_size), 
                                (center[0] + triangle_size, center[1])]) 
        cv2.drawContours(image, [play_points], 0, (255, 255, 255), -1) 
    else:
        pause_width = 6 
        pause_height = 10 
        cv2.rectangle(image, (center[0] - pause_width, center[1] - pause_height), 
                     (center[0] - pause_width // 2, center[1] + pause_height), (255, 255, 255), -1) 
        cv2.rectangle(image, (center[0] + pause_width // 2, center[1] - pause_height), 
                     (center[0] + pause_width, center[1] + pause_height), (255, 255, 255), -1)

#crea il piano tangente alla sfera
def create_plan(h_out, w_out, alpha, rho):
    u = np.arange(0, w_out)
    v = np.arange(0, h_out)
    #u e v rappresentano tutte le righe e tutte le colonne dell'immagine di output

    #facciamo coincidere il centro dell'immagine con quello del piano spostandolo e capovolgendolo. Infine riportiamo le coordinate nel range 0-1 tenendo in conto il FOV
    f = rho
    x_piano = f 
    y_piano = ((u - (w_out/2))/(w_out/2)) * (np.tan(np.radians(alpha)/2))
    z_piano = (((h_out/2) - v)/(h_out/2)) * (np.tan(np.radians(alpha)/2))

    y, z = np.meshgrid(y_piano, z_piano) #meshgrid prende i vettori di input convertendoli in matrici
    x = np.full_like(y, x_piano) #crea una matrice della stessa dimensione di y. I valori di ogni cella saranno uguali a x_piano

    return np.stack((x, y, z)) #unisce le matrici x, y e z creando un "cubo"

#gestione delle matrici di rotazione e dello spostamento del piano tangente
def rotate(coordinate_piano, yaw, pitch):
    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                        [np.sin(yaw), np.cos(yaw), 0],
                        [0, 0, 1]])
            
    pitch_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])
    
    rotationMatrix = np.dot(yaw_matrix, pitch_matrix) #calcoliamo la matrice delle rotazioni 

    coordinate_piano = np.tensordot(rotationMatrix, coordinate_piano, axes=([1], [0])) #prodotto tra matrice e "cubo", permette di effettuare la rotazione

    return coordinate_piano

#gestione dello zoom
def zoom(coordinate_piano, f):
    coordinate_piano[0,:,:] = np.full_like(coordinate_piano[0,:,:], f)
    return coordinate_piano

#normalizzazione delle coordinate del piano 
def normalize(coordinate_piano):
    norma = np.sqrt(coordinate_piano[0, :, :] ** 2 + coordinate_piano[1, :, :] ** 2 + coordinate_piano[2, :, :] ** 2)

    return coordinate_piano / norma

#gestione del passaggio da coordinate cartesiane a coordinate sferiche
def convert(coordinate_piano):
    phi = (np.arctan2(coordinate_piano[1, :, :], coordinate_piano[0, :, :]) % (2 * np.pi)) / (2 * np.pi)
    theta = (np.arccos(coordinate_piano[2, :, :]) % np.pi) / np.pi
    return phi, theta

#gestione della creazione dell'immagine. Tramite il remap "implementiamo" l'interpolazione bilineare (cv2.INTER_LINEAR)
def createImage(img, phi, theta, w_equirettangolo, h_equirettangolo, map_x, map_y):
    map_x = phi * w_equirettangolo
    map_y = theta * h_equirettangolo

    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR)

#gestione della navigazione nel video. Si ottiene variando gli angoli yaw e pitch utilizzati successivamente per il calcolo della matrici di rotazione
#dividiamo l'angolo per f per far si che lo spostamento nelle varie direzioni sia proporzionale allo zoom utilizzato.
def update_yaw_pitch(key, yaw_radian, pitch_radian,f):
    if key == ord('a'):  # Freccia sinistra
        yaw_radian -= np.radians(10)/f
    elif key == ord('d'):  # Freccia destra
        yaw_radian += np.radians(10)/f
    elif key == ord('w'):  # Freccia su
        pitch_radian -= np.radians(10)/f
    elif key == ord('s'):  # Freccia giu'
        pitch_radian += np.radians(10)/f

    return yaw_radian, pitch_radian

def play_pause(key, is_pause):
    if key == ord('p'):
        is_pause = not is_pause
    return is_pause

def restart(key, video, actual_frame):
    if key == ord('r'):
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return 0
    return actual_frame

def update_zoom(key, f):
    if key == ord('z'): #zoom-in
        f += 0.1
    elif key == ord('x') and f > 0.8: #zoom-out
        f -= 0.1
    return f

if __name__ == '__main__':

    video = cv2.VideoCapture("video\\video_1.MP4")

    if video is None or not video.isOpened():
        raise IOError("video not found")
    else:
        ret, img = video.read()
        h_equirettangolo, w_equirettangolo, canali = img.shape

        alpha = 60 #definizione del FOV
        rho = 1
        f = rho

        #dimensioni immagine di output
        h_out = 400 
        w_out = h_out

        #inizializzazione delle matrici utili alla fase di remap
        map_x = np.zeros((h_out, w_out), np.float32)
        map_y = np.zeros((h_out, w_out), np.float32)  

        #definizione vista iniziale 
        yaw = np.radians(180)
        pitch = np.radians(0)

        #definizione dei parametri per la gestione della grafica
        is_pause = False
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_frame = 0
        fps = video.get(cv2.CAP_PROP_FPS)

        total_time = int(num_frames / fps) #calcolo del tempo complessivo del video. Divide il numero complessivo di frame per i frame al secondo
        total_minute = total_time // 60 
        total_second = total_time % 60

        coordinate_piano_base = create_plan(h_out, w_out, alpha, rho) #crea il piano base a cui verranno applicate tutte le successive modifiche
        
        while True:
            coordinate_piano = zoom(coordinate_piano_base, f)
            coordinate_piano = rotate(coordinate_piano, yaw, pitch)
            coordinate_piano = normalize(coordinate_piano)
            phi, theta = convert(coordinate_piano)            
            imgOutput = createImage(img, phi, theta, w_equirettangolo, h_equirettangolo, map_x, map_y)

            draw_interface(imgOutput, is_pause, num_frames, actual_frame, fps, total_minute, total_second)
            cv2.imshow("assignment1 - spherical to planar", imgOutput)

            #gestione avanzamento frame
            if not is_pause and actual_frame < num_frames - 1:
                ret, img = video.read()
                actual_frame = actual_frame + 1
                if(actual_frame + 1 == num_frames):
                    is_pause = True

            #gestione input utente
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            else:
                if actual_frame < num_frames - 1:
                    is_pause = play_pause(key, is_pause)
                yaw, pitch = update_yaw_pitch(key, yaw, pitch, f)
                f = update_zoom(key,f)
                actual_frame = restart(key,video, actual_frame)
    cv2.destroyAllWindows()