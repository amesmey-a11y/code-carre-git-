from ugot import ugot
import cv2
import numpy as np
import time
import math
from ultralytics import YOLO

class RobotController:
    def __init__(self, ip):
        print(f"Initialisation du système... Connexion à {ip}  test_mov final marche.py:10 - test_caméra.py:10")
        self.got = ugot.UGOT()
        self.got.initialize(ip)
        self.got.open_camera()
        self.got.balance_start_balancing()
        
        # Modèle YOLOv8 (détecte les humains ET 79 autres types d'objets)
        self.model = YOLO('yolov8n.pt') 
        
        self.speed = 14
        self.robot_angle = 90.0
        self.pos_x, self.pos_y = 300, 300
        self.map_canvas = np.zeros((600, 600, 3), dtype=np.uint8)
        
        # --- CONFIGURATION DES SEUILS DE DISTANCE (CM) ---
        self.seuil_humain = 22.0      # Seuil d'arrêt et de déviation pour les humains
        self.seuil_obstacle = 15.0    # Seuil d'arrêt et de déviation pour les objets généraux
        self.seuil_verif_cote = 35.0  # Zone d'analyse pour décider du côté de fuite
        
        # --- CONFIGURATION DE L'HOMOGRAPHIE (PIXELS -> CM) ---
        PTS_PIXELS = np.array([
            [200, 240],  # 1. Coin Haut-Gauche (pixel)
            [440, 240],  # 2. Coin Haut-Droite (pixel)
            [560, 440],  # 3. Coin Bas-Droite (pixel)
            [80,  440]   # 4. Coin Bas-Gauche (pixel)
        ], dtype=np.float32)

        PTS_REELS_CM = np.array([
            [-15.0, 40.0], # X = -15cm, Y = 40cm
            [15.0,  40.0], # X = 15cm,  Y = 40cm
            [15.0,  15.0], # X = 15cm,  Y = 15cm
            [-15.0, 15.0]  # X = -15cm, Y = 15cm
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(PTS_PIXELS, PTS_REELS_CM)

        # --- CONFIGURATION DU FILTRE DE LUMIÈRE (CLAHE) ---
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # États mémoire pour stabiliser l'affichage et l'évitement
        self.last_human_state = np.zeros((3,), dtype=bool)
        self.last_obstacle_state = np.zeros((3,), dtype=bool)
        self.last_deviation_state = False
        self.last_direction = "droite"
        self.last_gauche_libre = True
        self.last_droite_libre = True
        
        time.sleep(2.0)
        
    def pix_to_cm(self, x, y):
        """Convertit les coordonnées de pixels (x, y) en coordonnées réelles (X_cm, Y_cm)."""
        point = np.array([[[x, y]]], dtype=np.float32)
        point_cm = cv2.perspectiveTransform(point, self.H)
        return point_cm[0][0][0], point_cm[0][0][1]

    def update_map(self):
        """Dessine la trajectoire en temps réel sur la mini-carte 2D."""
        rad = math.radians(self.robot_angle)
        self.pos_x += self.speed * 0.05 * math.cos(rad) * 2
        self.pos_y -= self.speed * 0.05 * math.sin(rad) * 2
        cv2.circle(self.map_canvas, (int(self.pos_x), int(self.pos_y)), 2, (0, 255, 255), -1)
        cv2.imshow("Mini-Carte 2D", self.map_canvas)

    def flush_camera_buffer(self):
        """Vide le flux d'images accumulées pour forcer le temps réel strict."""
        for _ in range(6):
            self.got.read_camera_data()

    def process_detection_matrix(self, frame):
        """
        Analyse l'environnement, calcule les distances réelles, remplit la 
        matrice HUD et détermine si une déviation d'urgence est requise.
        """
        h, w = frame.shape[:2]
        w3 = w // 3
        
        matrix = np.zeros((2, 3), dtype=bool)
        deviation_necessaire = False
        
        gauche_occupee = False
        droite_occupee = False

        # Prétraitement de l'image (Luminosité stable)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_stable = self.clahe.apply(gray)
        
        # Détection YOLO sur chaque frame pour une réactivité instantanée (zéro lag)
        results = self.model(frame, verbose=False)
        
        human_in_danger_zone = np.zeros((3,), dtype=bool)
        obstacle_in_danger_zone = np.zeros((3,), dtype=bool)

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy()) 
            obj_name = self.model.names[cls]    
            
            # Point de base au sol au milieu du bas de la boîte
            cx = (x1 + x2) / 2
            x_cm, y_cm = self.pix_to_cm(cx, y2) 
            
            # Couloir de collision du robot (largeur de -15cm à +15cm)
            dans_le_couloir = (-15.0 <= x_cm <= 15.0)
            
            # Détermination de la colonne (Gauche, Centre, Droite)
            col = 0 if x_cm < -5.0 else (1 if x_cm < 5.0 else 2)

            # Analyse de l'occupation latérale pour le contournement
            if y_cm <= self.seuil_verif_cote:
                if x_cm < 0:
                    gauche_occupee = True
                else:
                    droite_occupee = True

            # --- LOGIQUE DE TRAITEMENT ET COLORATION SELON LA CLASSE ---
            if cls == 0:  # HUMAIN
                if y_cm > self.seuil_humain:
                    # Mode Suivi (Loin) -> Cadre Orange
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 120, 0), 2)
                    cv2.putText(frame, f"Humain: {y_cm:.1f} cm", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 2)
                elif y_cm <= self.seuil_humain and dans_le_couloir:
                    # Mode Danger (Proche) -> Cadre Rouge Épais + Alerte
                    deviation_necessaire = True
                    human_in_danger_zone[col] = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"STOP HUMAIN: {y_cm:.1f} cm", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            
            else:  # TOUT AUTRE OBSTACLE
                if y_cm > self.seuil_obstacle:
                    # Mode Suivi Obstacle (Loin) -> Cadre Magenta/Rose
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                    cv2.putText(frame, f"{obj_name}: {y_cm:.1f} cm", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
                elif y_cm <= self.seuil_obstacle and dans_le_couloir:
                    # Mode Danger Obstacle (Proche) -> Cadre Rouge Épais + Alerte
                    deviation_necessaire = True
                    obstacle_in_danger_zone[col] = True
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
                    cv2.putText(frame, f"STOP {obj_name.upper()}: {y_cm:.1f} cm", (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                                
        # Sauvegarde des états
        self.last_human_state = human_in_danger_zone
        self.last_obstacle_state = obstacle_in_danger_zone
        self.last_deviation_state = deviation_necessaire
        self.last_gauche_libre = not gauche_occupee
        self.last_droite_libre = not droite_occupee

        # Choix de la direction de fuite
        if deviation_necessaire:
            if self.last_droite_libre and not self.last_gauche_libre:
                self.last_direction = "droite"
            elif self.last_gauche_libre and not self.last_droite_libre:
                self.last_direction = "gauche"
            elif self.last_gauche_libre and self.last_droite_libre:
                self.last_direction = "droite"
            else:
                self.last_direction = "recul"

        # Remplissage final de la matrice HUD
        matrix[0, :] = self.last_human_state
        matrix[1, :] = self.last_obstacle_state
        
        # --- RENDU VISUEL DE LA MATRICE (HUD) ---
        start_x, start_y = 15, 15
        cell_w, cell_h = 75, 30
        for r in range(2):
            for c in range(3):
                x_cell = start_x + c * cell_w
                y_cell = start_y + r * cell_h
                color = (100, 100, 100) 
                label = f"H:{['G','C','D'][c]}" if r == 0 else f"O:{['G','C','D'][c]}"
                
                if r == 0 and matrix[r, c]: color = (0, 255, 0)   # Vert si Humain < 22cm
                if r == 1 and matrix[r, c]: color = (0, 0, 255)   # Rouge si Obstacle < 15cm
                
                cv2.rectangle(frame, (x_cell, y_cell), (x_cell + cell_w, y_cell + cell_h), color, -1)
                cv2.rectangle(frame, (x_cell, y_cell), (x_cell + cell_w, y_cell + cell_h), (255, 255, 255), 1)
                cv2.putText(frame, label, (x_cell + 12, y_cell + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Indicateur de l'état des flancs
        etat_txt = f"Flancs -> G:{'LIBRE' if self.last_gauche_libre else 'BLOQUE'} | D:{'LIBRE' if self.last_droite_libre else 'BLOQUE'}"
        cv2.putText(frame, etat_txt, (start_x, start_y + 2 * cell_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Lignes verticales des 3 couloirs de vision
        cv2.line(frame, (w3, 0), (w3, h), (200, 200, 200), 1, cv2.LINE_AA)
        cv2.line(frame, (2 * w3, 0), (2 * w3, h), (200, 200, 200), 1, cv2.LINE_AA)
        
        # --- LIGNES DE SEUILS PHYSIQUES ---
        # Ligne Orange pour le seuil Humain (22 cm)
        cv2.line(frame, (0, int(h * 0.55)), (w, int(h * 0.55)), (0, 120, 255), 1, cv2.LINE_AA)
        # Ligne Rouge plus basse pour le seuil Obstacle (15 cm)
        cv2.line(frame, (0, int(h * 0.72)), (w, int(h * 0.72)), (0, 0, 255), 1, cv2.LINE_AA)
        
        if deviation_necessaire:
            self.got.show_light_rgb_effect(255, 0, 0, 3)
            
        return deviation_necessaire

    def eviter_obstacle(self, angle_cible):
        """Arrête le robot, choisit la meilleure voie et contourne en boîte."""
        direction = self.last_direction
        print(f"🛑 Zone de sécurité franchie ! Évitement par : {direction.upper()}  test_mov final marche.py:214 - test_caméra.py:214")
        self.got.balance_move_speed(0, 0)
        time.sleep(0.1)

        if direction == "recul":
            self.got.play_sound("lion")
            self.got.balance_move_speed(180, self.speed)
            time.sleep(0.6)
            self.got.balance_move_speed(0, 0)
            self.tourner_recaler(angle_cible)
            self.flush_camera_buffer()
            return

        turn_sens = 3 if direction == "droite" else 2
        turn_oppose = 2 if direction == "droite" else 3

        # Séquence de contournement en boîte
        self.got.balance_turn_speed(turn_sens, 40)
        time.sleep(0.4)
        self.got.balance_move_speed(0, self.speed)
        time.sleep(0.4)

        self.got.balance_turn_speed(turn_oppose, 40)
        time.sleep(0.4)
        self.got.balance_move_speed(0, self.speed)
        time.sleep(0.8)

        self.got.balance_turn_speed(turn_oppose, 40)
        time.sleep(0.4)
        self.got.balance_move_speed(0, self.speed)
        time.sleep(0.4)

        self.got.balance_turn_speed(turn_sens, 40)
        time.sleep(0.4)

        # Réalignement Boussole/IMU
        self.tourner_recaler(angle_cible)
        self.flush_camera_buffer()
        self.got.balance_move_speed(0, self.speed)

    def naviguer_segment(self, distance, angle_cible):
        """Fait avancer le robot et analyse le flux en direct."""
        duree = distance / self.speed
        start = time.time()
        
        while time.time() - start < duree:
            data = self.got.read_camera_data()
            if data:
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                deviation_declenchee = self.process_detection_matrix(frame)
                
                if deviation_declenchee:
                    self.eviter_obstacle(angle_cible)
                    start = time.time() # Réinitialise le temps de navigation après l'évitement
                
                self.got.balance_move_speed(0, self.speed)
                self.update_map()
                cv2.imshow("Vision Robot", frame)
                
            if cv2.waitKey(1) == ord('q'): 
                break
        self.got.balance_move_speed(0, 0)

    def tourner_recaler(self, angle_cible):
        """Pivote proprement le robot vers l'angle visé."""
        diff = (angle_cible - self.robot_angle)
        direction = 3 if diff < 0 else 2
        self.got.balance_turn_speed(direction, 90)
        time.sleep(abs(diff) / 90)
        self.got.balance_turn_speed(direction, 0)
        self.robot_angle = float(angle_cible)
        time.sleep(0.2)

    def run_mission(self):
        try:
            cv2.rectangle(self.map_canvas, (120, 120), (480, 480), (255, 0, 0), 1)
            # Parcours en carré
            segments = [(180, 0), (120, -90), (180, -180), (120, -270)]
            
            for dist, angle in segments:
                self.naviguer_segment(dist, angle)
                self.tourner_recaler(angle)
                
            print("🏁 Mission accomplie ! Retour au point de départ.  test_mov final marche.py:297 - test_caméra.py:297")
            self.got.play_sound("victory")
        finally:
            self.got.balance_stop_balancing()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    # /!\ METTRE L'IP CORRECTE ICI (Vérifie bien sur l'écran de ton robot !)
    robot = RobotController("172.16.1.62") 
    robot.run_mission()