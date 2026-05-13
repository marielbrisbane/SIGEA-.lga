import cv2
import math
from collections import deque, Counter


class GestureRecognizer:

    def __init__(self):
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70
        )
        self.frame_skip       = 2
        self.frame_count      = 0
        self.gesture_history  = deque(maxlen=9)
        self.last_result      = None
        self.lm_history       = deque(maxlen=24)
        self.dynamic_cooldown = 0
        self.DYNAMIC_COOLDOWN = 30


    # ══ UTILITARIOS ═══════════════════════════════════════════════════════════

    def _d(self, p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _palm(self, lm):
        return self._d(lm[0], lm[9]) or 1e-6

    def _dn(self, lm, a, b):
        """Distancia normalizada entre dois landmarks"""
        return self._d(lm[a], lm[b]) / self._palm(lm)

    def _aberto(self, lm, tip, pip, th=1.15):
        """Dedo aberto: ponta mais longe do pulso que articulacao"""
        r = self._d(lm[tip], lm[0]) / (self._d(lm[pip], lm[0]) or 1e-6)
        return r > th

    def _fechado(self, lm, tip, pip, th=0.93):
        r = self._d(lm[tip], lm[0]) / (self._d(lm[pip], lm[0]) or 1e-6)
        return r < th

    def _polegar(self, lm):
        palm = self._palm(lm)
        return (self._d(lm[4], lm[0]) / palm) > (self._d(lm[3], lm[0]) / palm) * 1.10

    def _dedos(self, lm):
        """(Polegar, Indicador, Medio, Anelar, Mindinho)"""
        return (
            self._polegar(lm),
            self._aberto(lm, 8,  6),
            self._aberto(lm, 12, 10),
            self._aberto(lm, 16, 14),
            self._aberto(lm, 20, 18),
        )

    def _mao_horizontal(self, lm):
        """True se a mao estiver orientada horizontalmente"""
        dy = abs(lm[5].y - lm[17].y)
        dx = abs(lm[5].x - lm[17].x)
        return dx > dy

    def _palma_cima(self, lm):
        """True se palma virada para cima (y do pulso > y dos MCPs)"""
        return lm[0].y > lm[9].y

    def _dedos_juntos(self, lm):
        """True se indicador, medio, anelar e mindinho estiverem proximos"""
        d1 = self._dn(lm, 8, 12)
        d2 = self._dn(lm, 12, 16)
        d3 = self._dn(lm, 16, 20)
        return d1 < 0.20 and d2 < 0.20 and d3 < 0.20


    # ══ NUMERACAO LGA 0-9 ═════════════════════════════════════════════════════
    # Baseado na imagem oficial "Numeracao em LAS" (WA0012)

    def _numero(self, lm):
        T, I, M, A, Min = self._dedos(lm)
        horiz = self._mao_horizontal(lm)
        palma_up = self._palma_cima(lm)
        juntos = self._dedos_juntos(lm)

        # 0 — punho fechado curvado (todos fechados, nao completamente)
        if not T and not I and not M and not A and not Min:
            # verificar que nao e punho completamente fechado (dedos ligeiramente abertos)
            d_tip_base = self._dn(lm, 8, 5)
            if 0.20 < d_tip_base < 0.45:
                return ('LGA 0', 0.82)

        # 1 — mao aberta horizontal palma para cima
        if T and I and M and A and Min and horiz and palma_up:
            return ('LGA 1', 0.85)

        # 2 — polegar levantado (thumbs up), punho fechado
        if T and not I and not M and not A and not Min and not horiz:
            return ('LGA 2', 0.87)

        # 3 — dedos curvados juntos e pequenos (semi-fechado)
        if not T and not I and not M and not A and not Min:
            d_tip_base = self._dn(lm, 8, 5)
            if d_tip_base < 0.20:
                return ('LGA 3', 0.80)

        # 4 — mao aberta dedos separados (todos abertos, nao juntos)
        if T and I and M and A and Min and not juntos:
            return ('LGA 4', 0.84)

        # 5 — mao horizontal dedos juntos fechados (karate)
        if not T and I and M and A and Min and horiz and juntos:
            return ('LGA 5', 0.84)

        # 6 — circulo com polegar e indicador (OK)
        d_T_I = self._dn(lm, 4, 8)
        if d_T_I < 0.25 and M and A and Min:
            return ('LGA 6', 0.85)

        # 7 — mao aberta horizontal palma para baixo (todos abertos, horiz, palma baixo)
        if T and I and M and A and Min and horiz and not palma_up:
            return ('LGA 7', 0.83)

        # 8 — dedos curvados fechados (punho mais fechado que 3)
        todos_curv = (
            self._fechado(lm, 8, 6, th=0.85) and
            self._fechado(lm, 12, 10, th=0.85) and
            self._fechado(lm, 16, 14, th=0.85) and
            self._fechado(lm, 20, 18, th=0.85)
        )
        if todos_curv and not T:
            return ('LGA 8', 0.82)

        # 9 — mao de lado, dedos fechados (vista lateral)
        if not T and not I and not M and not A and not Min:
            # diferencia do 0 pelo angulo da mao
            dx = abs(lm[5].x - lm[17].x)
            dy = abs(lm[5].y - lm[17].y)
            if dy > dx:  # mao mais vertical = de lado
                return ('LGA 9', 0.80)

        return None


    # ══ LETRAS LGA ════════════════════════════════════════════════════════════
    # Baseado nas fotos reais da parede e no modelo A-H do PAP

    def _letra(self, lm):
        T, I, M, A, Min = self._dedos(lm)
        horiz = self._mao_horizontal(lm)

        # A — indicador apontando para o lado (horizontal), restantes fechados
        # Foto WA0033 e WA0049: indicador estendido horizontal, polegar dobrado
        if I and not T and not M and not A and not Min and horiz:
            # indicador aponta para o lado
            dx = abs(lm[8].x - lm[5].x)
            dy = abs(lm[8].y - lm[5].y)
            if dx > dy * 1.2:
                return ('LGA A', 0.86)

        # D — mao aberta horizontal palma para baixo (tipo D invertido)
        # Foto WA0031: todos os dedos esticados para a frente, palma baixo
        if not T and I and M and A and Min and horiz:
            palma_down = not self._palma_cima(lm)
            if palma_down:
                return ('LGA D', 0.85)

        # E — punho fechado com dedos curvados para cima
        # Foto WA0032 e WA0046: punho fechado, polegar visivel ao lado
        todos_fechados = (
            self._fechado(lm, 8, 6) and
            self._fechado(lm, 12, 10) and
            self._fechado(lm, 16, 14) and
            self._fechado(lm, 20, 18)
        )
        if todos_fechados and not T and not horiz:
            return ('LGA E', 0.85)

        # I — mindinho levantado, restantes fechados
        # Foto WA0045: so o mindinho esticado para cima
        if Min and not T and not I and not M and not A and not horiz:
            if lm[20].y < lm[17].y:  # mindinho aponta para cima
                return ('LGA I', 0.87)

        # O — mao horizontal com dedos juntos curvados
        # Foto WA0047: mao horizontal, todos os dedos curvados juntos
        if horiz and not T:
            dedos_curv_juntos = (
                self._fechado(lm, 8, 6, th=1.10) and
                self._fechado(lm, 12, 10, th=1.10) and
                self._dn(lm, 8, 12) < 0.22
            )
            if dedos_curv_juntos:
                return ('LGA O', 0.84)

        # S — punho completamente fechado
        # Foto WA0034: punho fechado solido, sem polegar visivel
        punho_total = (
            self._fechado(lm, 8, 6, th=0.88) and
            self._fechado(lm, 12, 10, th=0.88) and
            self._fechado(lm, 16, 14, th=0.88) and
            self._fechado(lm, 20, 18, th=0.88)
        )
        if punho_total and not T and not horiz:
            return ('LGA S', 0.84)

        # 1/I com polegar — indicador e mindinho levantados (foto WA0050)
        if I and Min and T and not M and not A and not horiz:
            return ('LGA Y', 0.83)

        # B — 4 dedos esticados juntos verticais (do modelo PAP)
        if not T and I and M and A and Min and not horiz:
            d_I_M = self._dn(lm, 8, 12)
            d_M_A = self._dn(lm, 12, 16)
            if d_I_M < 0.18 and d_M_A < 0.18:
                return ('LGA B', 0.84)

        # C — mao semi-aberta em forma de C (do modelo PAP)
        d_T_Min = self._dn(lm, 4, 20)
        sem_extremos = (
            not self._aberto(lm, 8, 6, th=1.30) and
            not self._fechado(lm, 8, 6, th=0.80)
        )
        if 0.35 < d_T_Min < 0.65 and sem_extremos:
            return ('LGA C', 0.81)

        # F — indicador e polegar tocam, medio anelar mindinho abertos (do modelo PAP)
        d_T_I = self._dn(lm, 4, 8)
        if d_T_I < 0.22 and M and A and Min and not horiz:
            return ('LGA F', 0.84)

        # G — polegar e indicador horizontais tipo pistola (do modelo PAP)
        if T and I and not M and not A and not Min:
            dx = abs(lm[8].x - lm[5].x)
            dy = abs(lm[8].y - lm[5].y)
            if dx > dy * 0.9:
                return ('LGA G', 0.82)

        # H — indicador e medio paralelos horizontais (do modelo PAP)
        if not T and I and M and not A and not Min:
            d_I_M = self._dn(lm, 8, 12)
            dx = abs(lm[8].x - lm[5].x)
            dy = abs(lm[8].y - lm[5].y)
            if d_I_M < 0.20 and dx > dy * 0.7:
                return ('LGA H', 0.82)

        return None


    # ══ SAUDACOES DINAMICAS LGA ═══════════════════════════════════════════════

    def _saudacao(self, lm):
        if len(self.lm_history) < 10:
            return None

        frames = list(self.lm_history)
        T, I, M, A, Min = self._dedos(lm)

        xs = [f[0].x for f in frames]
        ys = [f[0].y for f in frames]
        mov_h = max(xs) - min(xs)
        mov_v = max(ys) - min(ys)

        # OLÁ — mao aberta com aceno horizontal
        if T and I and M and A and Min:
            if mov_h > 0.13 and mov_v < 0.07:
                return ('LGA Olá', 0.82)

        # TUDO BEM — polegar levantado oscilando verticalmente
        if T and not I and not M and not A and not Min:
            if mov_v > 0.10 and mov_h < 0.07:
                dirs = [ys[k+1]-ys[k] for k in range(len(ys)-1)]
                mud  = sum(1 for k in range(len(dirs)-1) if dirs[k]*dirs[k+1] < 0)
                if mud >= 2:
                    return ('LGA Tudo Bem', 0.80)

        # OBRIGADO — mao aberta movimento diagonal
        if T and I and M and A and Min:
            if mov_h > 0.07 and mov_v > 0.05:
                palms = [self._d(f[0], f[9]) for f in frames]
                if max(palms) - min(palms) > 0.015:
                    return ('LGA Obrigado', 0.78)

        # SIM — punho fechado oscilando verticalmente
        if not T and not I and not M and not A and not Min:
            if mov_v > 0.09 and mov_h < 0.06:
                dirs = [ys[k+1]-ys[k] for k in range(len(ys)-1)]
                mud  = sum(1 for k in range(len(dirs)-1) if dirs[k]*dirs[k+1] < 0)
                if mud >= 2:
                    return ('LGA Sim', 0.83)

        # NAO — indicador oscilando horizontalmente
        if not T and I and not M and not A and not Min:
            if mov_h > 0.11:
                dirs = [xs[k+1]-xs[k] for k in range(len(xs)-1)]
                mud  = sum(1 for k in range(len(dirs)-1) if dirs[k]*dirs[k+1] < 0)
                if mud >= 2:
                    return ('LGA Não', 0.84)

        # POR FAVOR — mao aberta movimento circular
        if T and I and M and A and Min:
            if mov_h > 0.06 and mov_v > 0.06:
                cx = sum(xs)/len(xs); cy = sum(ys)/len(ys)
                dists = [math.sqrt((x-cx)**2+(y-cy)**2) for x,y in zip(xs,ys)]
                if max(dists)-min(dists) < 0.04:
                    return ('LGA Por Favor', 0.77)

        return None


    # ══ PIPELINE PRINCIPAL ════════════════════════════════════════════════════

    def recognize_gesture(self, lm):
        self.lm_history.append(lm)

        # 1. Saudacoes dinamicas
        if self.dynamic_cooldown > 0:
            self.dynamic_cooldown -= 1
        else:
            din = self._saudacao(lm)
            if din:
                self.dynamic_cooldown = self.DYNAMIC_COOLDOWN
                return {'gesture': din[0], 'confidence': din[1]}

        # 2. Numeros LGA
        num = self._numero(lm)
        if num:
            return {'gesture': num[0], 'confidence': num[1]}

        # 3. Letras LGA
        let = self._letra(lm)
        if let:
            return {'gesture': let[0], 'confidence': let[1]}

        return {'gesture': 'Desconhecido', 'confidence': 0.0}


    def _smooth(self, result):
        g = result['gesture']
        if g == 'Desconhecido':
            return result
        if any(s in g for s in ['Olá','Tudo','Obrigado','Sim','Não','Por']):
            return result
        self.gesture_history.append(g)
        melhor = Counter(self.gesture_history).most_common(1)[0][0]
        return {'gesture': melhor, 'confidence': result['confidence']}


    # ══ PROCESSAR FRAME ═══════════════════════════════════════════════════════

    def process_frame(self, frame):
        self.frame_count += 1

        if self.frame_count % self.frame_skip != 0:
            if self.last_result:
                self._draw(frame, self.last_result)
            return frame, self.last_result, None

        small  = cv2.resize(frame, (320, 240))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res    = self.hands.process(rgb)

        gesture_result = None
        landmarks_data = None

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            lm   = hand.landmark

            raw            = self.recognize_gesture(lm)
            gesture_result = self._smooth(raw)
            self.last_result = gesture_result

            landmarks_data = [
                {'index': idx, 'x': float(p.x), 'y': float(p.y), 'z': float(p.z)}
                for idx, p in enumerate(lm)
            ]

            h, w = frame.shape[:2]
            for conn in self.mp_hands.HAND_CONNECTIONS:
                s, e = conn
                x1, y1 = int(lm[s].x * w), int(lm[s].y * h)
                x2, y2 = int(lm[e].x * w), int(lm[e].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 200, 224), 2)
            for pt in lm:
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 200, 224), -1)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 1)
        else:
            self.gesture_history.clear()
            self.lm_history.clear()
            self.last_result      = None
            self.dynamic_cooldown = 0

        self._draw(frame, gesture_result)
        return frame, gesture_result, landmarks_data


    def _draw(self, frame, result):
        h, w = frame.shape[:2]

        if result and result['gesture'] != 'Desconhecido':
            g    = result['gesture']
            conf = result['confidence']

            if 'LGA ' in g and any(c.isdigit() for c in g):
                cor = (0, 210, 255)     # amarelo — numeros
            elif any(s in g for s in ['Olá','Tudo','Obrigado','Sim','Não','Por']):
                cor = (255, 140, 0)     # laranja — saudacoes
            else:
                cor = (80, 255, 140)    # verde — letras

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-100), (w, h), (8, 18, 30), -1)
            cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

            cv2.putText(frame, g, (14, h-58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.80, cor, 2, cv2.LINE_AA)

            cat = ('Número LGA' if any(c.isdigit() for c in g) else
                   'Saudação LGA' if any(s in g for s in ['Olá','Tudo','Obrigado','Sim','Não','Por']) else
                   'Letra LGA')
            cv2.putText(frame, cat, (14, h-34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 160, 185), 1, cv2.LINE_AA)

            bw = int((w-28) * conf)
            cv2.rectangle(frame, (14, h-20), (w-14, h-10), (20, 40, 60), -1)
            cv2.rectangle(frame, (14, h-20), (14+bw,  h-10), cor, -1)
            cv2.putText(frame, f'{int(conf*100)}%', (w-52, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, cor, 1, cv2.LINE_AA)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h-38), (w, h), (8, 18, 30), -1)
            cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
            cv2.putText(frame, 'Mostre um gesto LGA...', (14, h-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (70, 110, 140), 1, cv2.LINE_AA)

    def get_smoothed_gesture(self):
        return self.last_result
