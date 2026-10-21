"""
scripts/survey123/textos.py
───────────────────────────
Generación de textos de reportes ciudadanos por INTENCIÓN.

Un texto = SALUDO + ESTRUCTURA(intención) rellenada con VOCABULARIO(categoría)
+ LUGAR + TIEMPO + CIERRE.

  · 6 intenciones: queja, denuncia, solicitud, consulta, sugerencia, agradecimiento
  · ~20 estructuras de frase, cada una con un id (Q1, S2, …)
  · vocabulario propio de cada una de las 10 categorías

El id de estructura (plantilla) permite partir los datos de modo que las
estructuras de evaluación NUNCA aparezcan en el entrenamiento: así se mide
si un clasificador generaliza a redacciones nuevas, y no si memorizó frases.

El sentimiento se deriva de la intención y del tono:
  queja, denuncia            → negativo
  agradecimiento             → positivo
  solicitud, consulta,
  sugerencia                 → neutro, o negativo si el tono es de hartazgo
"""
import random
from typing import Tuple

INTENCIONES = ["queja", "denuncia", "solicitud", "consulta", "sugerencia", "agradecimiento"]

LUGAR = ["en mi barrio", "en la calle principal", "frente a la escuela", "en la esquina de mi casa",
         "cerca del parque", "en la parada del bus", "junto al mercado", "en mi cuadra",
         "en el pasaje", "a media cuadra del UPC", "frente a la iglesia", "en la entrada del conjunto",
         "por el redondel", "en la bajada al barrio", "atrás del centro de salud"]
TIEMPO = ["hace dos semanas", "desde hace un mes", "desde el lunes", "hace varios días",
          "desde hace meses", "otra vez esta semana", "desde el fin de semana", "hace tres días"]
SALUDO = ["", "", "", "Buenos días. ", "Buenas tardes. ", "Hola, ", "Como vecino del sector: ", "Estimados: "]

# ── Vocabulario por categoría ─────────────────────────────
# problema: hecho negativo en indicativo (queja)
# accion:   infinitivo de lo que se pide (solicitud, consulta)
# mejora:   infinitivo de una propuesta (sugerencia)
# logro:    infinitivo de lo que se hizo bien (agradecimiento)
# hecho:    conducta de terceros en indicativo (denuncia)
VOCAB = {
    "vias": {
        "problema": ["hay un bache enorme", "la calle está destruida", "el adoquinado está levantado",
                     "la vereda está rota"],
        "accion": ["tapar el bache", "asfaltar la calle", "reparar la vereda"],
        "mejora": ["poner reductores de velocidad", "señalizar el paso peatonal", "ampliar la vereda"],
        "logro": ["arreglar el bache", "asfaltar la calle", "reparar la vereda"],
    },
    "alumbrado": {
        "problema": ["las lámparas no funcionan", "el poste de luz está quemado",
                     "la calle está completamente oscura", "la luminaria parpadea toda la noche"],
        "accion": ["cambiar la lámpara del poste", "revisar el alumbrado", "instalar más luminarias"],
        "mejora": ["poner luminarias LED", "iluminar mejor el pasaje", "instalar luces solares"],
        "logro": ["cambiar las lámparas", "arreglar el alumbrado", "instalar las nuevas luminarias"],
    },
    "basura": {
        "problema": ["el recolector no pasa", "la basura se acumula en la vereda",
                     "los contenedores están desbordados", "hay fundas de basura regadas"],
        "accion": ["cambiar el horario del recolector", "poner un contenedor", "recoger la basura acumulada"],
        "mejora": ["poner contenedores de reciclaje", "hacer una campaña de reciclaje", "colocar más basureros"],
        "logro": ["poner los contenedores nuevos", "mejorar la recolección", "limpiar el botadero"],
        "hecho": ["botan escombros en la quebrada", "queman basura en el terreno baldío",
                  "un camión deja desechos en la noche"],
    },
    "agua": {
        "problema": ["no tenemos agua", "el agua llega turbia", "hay una fuga de agua",
                     "la presión del agua es muy baja"],
        "accion": ["arreglar la fuga", "revisar la presión del agua", "restablecer el servicio de agua"],
        "mejora": ["avisar con tiempo los cortes de agua", "cambiar las tuberías viejas", "instalar medidores nuevos"],
        "logro": ["arreglar la fuga", "restablecer el servicio de agua", "cambiar la tubería"],
    },
    "alcantarillado": {
        "problema": ["la alcantarilla está tapada", "se inunda todo cuando llueve",
                     "huele a desagüe", "el colector está roto"],
        "accion": ["limpiar las alcantarillas", "destapar el colector", "revisar el desagüe"],
        "mejora": ["limpiar las rejillas antes de las lluvias", "ampliar el colector", "hacer canales para el agua lluvia"],
        "logro": ["limpiar las alcantarillas", "reparar el colector", "destapar el desagüe"],
    },
    "seguridad": {
        "problema": ["no hay patrullaje", "da miedo caminar de noche", "hay muchos robos",
                     "la zona es muy insegura"],
        "accion": ["enviar patrullaje", "instalar una cámara de vigilancia", "poner un botón de pánico"],
        "mejora": ["formar una brigada barrial", "instalar alarmas comunitarias", "poner más cámaras"],
        "logro": ["enviar más patrullaje", "instalar las cámaras", "atender nuestro llamado"],
        "hecho": ["roban celulares a los estudiantes", "asaltan a la gente en la parada",
                  "venden droga en la esquina", "rompen los vidrios de los carros"],
    },
    "ruido": {
        "problema": ["el ruido no deja dormir", "la música está a todo volumen",
                     "el taller hace un ruido insoportable", "las alarmas suenan toda la noche"],
        "accion": ["controlar el ruido", "revisar el permiso del local", "hacer un operativo de control"],
        "mejora": ["poner horarios para la música", "controlar los locales nocturnos", "multar el ruido excesivo"],
        "logro": ["controlar el ruido", "clausurar el local ruidoso", "hacer el operativo"],
        "hecho": ["hay fiestas clandestinas hasta la madrugada", "un bar funciona sin permiso",
                  "un local pone parlantes en la vereda"],
    },
    "transporte": {
        "problema": ["los buses no paran", "el bus tarda más de media hora",
                     "los buses pasan llenos", "la parada no tiene cubierta"],
        "accion": ["aumentar la frecuencia de buses", "poner una parada", "cubrir la parada"],
        "mejora": ["crear una ruta nueva", "poner horarios en las paradas", "agregar buses en hora pico"],
        "logro": ["aumentar la frecuencia de buses", "poner la nueva parada", "crear la nueva ruta"],
    },
    "areas_verdes": {
        "problema": ["el parque está abandonado", "los juegos infantiles están rotos",
                     "no cortan el césped", "el área verde está llena de basura"],
        "accion": ["dar mantenimiento al parque", "reparar los juegos", "cortar el césped"],
        "mejora": ["sembrar más árboles", "poner máquinas de ejercicio", "hacer un huerto comunitario"],
        "logro": ["remodelar el parque", "reparar los juegos", "sembrar los árboles"],
    },
    "animales": {
        "problema": ["hay perros callejeros agresivos", "hay muchos perros abandonados",
                     "un perro atropellado sigue en la calle", "los perros rompen las fundas de basura"],
        "accion": ["recoger a los perros abandonados", "hacer una campaña de esterilización", "retirar el animal muerto"],
        "mejora": ["hacer campañas de adopción", "vacunar a las mascotas del barrio", "esterilizar a los perros callejeros"],
        "logro": ["rescatar a los perritos", "hacer la campaña de esterilización", "vacunar a las mascotas"],
        "hecho": ["maltratan a un perro en una casa", "abandonan cachorros en el parque",
                  "envenenaron a varios perros"],
    },
}
CATEGORIAS_DENUNCIA = [c for c, v in VOCAB.items() if "hecho" in v]

# ── Estructuras por intención (id, plantilla) ─────────────
# {p} problema · {a} accion · {m} mejora · {g} logro · {h} hecho · {l} lugar · {t} tiempo
ESTRUCTURAS = {
    "queja": [
        ("Q1", "{P} {l} {t} y nadie hace nada."),
        ("Q2", "Estoy cansado, {p} {l} {t}."),
        ("Q3", "{P} {l}. Ya reportamos {t} y seguimos igual."),
        ("Q4", "Es una vergüenza: {p} {l} {t}."),
        ("Q5", "Los vecinos estamos molestos porque {p} {l}."),
    ],
    "denuncia": [
        ("D1", "Denuncio que {h} {l} {t}."),
        ("D2", "Quiero denunciar que {h} {l}."),
        ("D3", "{H} {l} {t}, pido que intervengan las autoridades."),
        ("D4", "Les informo que {h} {l}, esto es ilegal."),
    ],
    "solicitud": [
        ("S1", "Solicito {a} {l}."),
        ("S2", "Por favor, necesitamos {a} {l}."),
        ("S3", "Pido al Municipio {a} {l} lo antes posible."),
        ("S4", "Requerimos {a} {l}, gracias de antemano."),
    ],
    "consulta": [
        ("C1", "¿Cuándo van a {a} {l}?"),
        ("C2", "Quisiera saber si está previsto {a} {l}."),
        ("C3", "¿A quién debo acudir para {a} {l}?"),
    ],
    "sugerencia": [
        ("G1", "Sería bueno {m} {l}."),
        ("G2", "Propongo {m} {l}."),
        ("G3", "Sugiero {m} {l} para mejorar el sector."),
    ],
    "agradecimiento": [
        ("A1", "Gracias por {g} {l}."),
        ("A2", "Quiero agradecer al Municipio por {g} {l}."),
        ("A3", "Excelente trabajo, gracias por {g} {l} tan rápido."),
    ],
}
CIERRE = {
    "queja": ["", "", " Exigimos una solución urgente.", " Es indignante.", " Los vecinos estamos cansados."],
    "denuncia": ["", "", " Por favor actúen.", " Tenemos miedo.", " Ya no se puede vivir así."],
    "solicitud": ["", "", " Gracias.", " Quedo atento.", " Saludos."],
    "consulta": ["", "", " Gracias.", " Agradezco la respuesta.", " Saludos."],
    "sugerencia": ["", "", " Gracias.", " Saludos.", " Sería de gran ayuda."],
    "agradecimiento": ["", "", " Felicitaciones al equipo.", " Sigan así.", " Muy agradecido."],
}
# Tono de hartazgo: convierte una intención neutra en sentimiento negativo
HARTAZGO = [" Ya van meses esperando y nada.", " Estamos hartos de esperar.",
            " Es la tercera vez que lo pido.", " Nadie nos hace caso."]

NEGATIVAS = {"queja", "denuncia"}
NEUTRAS = {"solicitud", "consulta", "sugerencia"}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def sentimiento_de(intencion: str, hartazgo: bool) -> str:
    if intencion in NEGATIVAS:
        return "negativo"
    if intencion == "agradecimiento":
        return "positivo"
    return "negativo" if hartazgo else "neutro"


def generar_texto(categoria: str, intencion: str, rnd: random.Random,
                  hartazgo: bool = False, estructuras=None) -> Tuple[str, str]:
    """
    Devuelve (texto, id_estructura). `estructuras` permite restringir a un
    subconjunto de ids (para separar entrenamiento y evaluación).
    """
    if intencion == "denuncia" and categoria not in CATEGORIAS_DENUNCIA:
        categoria = rnd.choice(CATEGORIAS_DENUNCIA)
    opciones = [e for e in ESTRUCTURAS[intencion] if estructuras is None or e[0] in estructuras]
    eid, plantilla = rnd.choice(opciones)
    v = VOCAB[categoria]
    p = rnd.choice(v["problema"])
    h = rnd.choice(v["hecho"]) if "hecho" in v else p
    s = plantilla.format(p=p, P=_cap(p), a=rnd.choice(v["accion"]), m=rnd.choice(v["mejora"]),
                         g=rnd.choice(v["logro"]), h=h, H=_cap(h),
                         l=rnd.choice(LUGAR), t=rnd.choice(TIEMPO))
    s += rnd.choice(HARTAZGO) if hartazgo and intencion in NEUTRAS else rnd.choice(CIERRE[intencion])
    s = _cap(s)
    saludo = rnd.choice(SALUDO)
    if saludo:
        bajar = saludo.endswith(", ") or saludo.endswith(": ")
        s = saludo + (s[0].lower() + s[1:] if bajar and not s.startswith("¿") else s)
    return s, eid


def elegir_intencion(score: float, categoria: str, rnd: random.Random) -> Tuple[str, bool]:
    """
    Intención a partir del sentimiento latente del reporte (ver generar_reportes.py).
    Devuelve (intencion, hartazgo).
    """
    if score < 0.3:
        p_den = {"seguridad": 0.6, "ruido": 0.4, "basura": 0.3, "animales": 0.4}.get(categoria, 0.0)
        return ("denuncia" if rnd.random() < p_den else "queja"), False
    if score > 1.25:
        return "agradecimiento", False
    intencion = rnd.choices(["solicitud", "consulta", "sugerencia"], weights=[0.5, 0.25, 0.25])[0]
    return intencion, rnd.random() < 0.15
