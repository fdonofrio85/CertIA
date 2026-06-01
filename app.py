"""
Revisor Unit Rotary - Pulling v2
DLS-080 / NOVA-081
"""
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata
import json
import io
import os
import sys
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Revisor Unit Rotary", page_icon="🛢️", layout="wide")

# ── Colores por tarifa ────────────────────────────────────────────────────────
COLORES = {
    'DTM':    '#2ecc71',   # verde claro
    'RIG':    '#a8d5a2',   # verde suave
    'RIGS':   '#87CEEB',   # celeste
    'RIGWOW': '#F0E68C',   # amarillo
    'RIGNC':  '#FFB6C1',   # rosa
    'THR':    '#FF6B6B',   # rojo
    '?':      '#D3D3D3',   # gris
}
COLOR_TEXTO = {
    'DTM':'#145a32','RIG':'#1a5c18','RIGS':'#1a5276',
    'RIGWOW':'#7d6608','RIGNC':'#78281f','THR':'#922b21','?':'#555555',
}

def badge_html(tarifa):
    bg = COLORES.get(tarifa, '#D3D3D3')
    fg = COLOR_TEXTO.get(tarifa, '#333')
    label = str(tarifa)
    return f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:4px;font-weight:bold;font-size:12px">{label}</span>'

st.markdown("""
<style>
.stDataFrame { font-size: 12px; }
div[data-testid="metric-container"] { background:#f8f9fa; border-radius:8px; padding:8px; }
.diff-row { background: #fff3cd !important; }
thead tr th { background: #2c3e50 !important; color: white !important; font-size:12px; }
</style>
""", unsafe_allow_html=True)

# ── Normalización ─────────────────────────────────────────────────────────────
def normalizar(texto):
    t = str(texto).lower().strip()
    t = unicodedata.normalize('NFKD', t)
    return ''.join(c for c in t if not unicodedata.combining(c))

def hay(t, *p): return any(x in t for x in p)

# ── Keywords ──────────────────────────────────────────────────────────────────
CIAS = ['cia brwt','cia bwt','cia rms','cia rm&s','cia maxicom','cia maxicon','cia sga',
        'cia wtf','cia vermaz','cia burguward','cia cam ','cia mbp','cia vds','cia mero',
        'cia canada','brwt ','burgward','burguward','vermaz','hot oil','wireline',
        'wire line','logistica de cia','log. descarga','ingresa log.',
        'ingresa logistica','maquina vial']

DTM_KW = ['transporta equipo','transportando equipo','comienza a transportar',
    'continua transporte','realiza dtm','raeliza dtm','engancha trailer',
    'engancha casilla','desengancha trailer','desmonta equipo completo',
    'termina de desmontar equipo','atraca y monta equipo','atraca y nivela equipo',
    'atraca, nivela y monta','monta equipo completo','termina de montar equipo',
    'atraca + monta','cont. montando eq','cerco perimetral','cercos perimetrales',
    'serco perimetral','cerco pedimetral','arma puente','desarma puente',
    'desmonta puente','retira puente','arma y conecta puente','arma kit de inyeccion',
    'arma puente de inyeccion','pump off','pump-off','consigna tablero',
    'desconsigna tablero','retira consigna','delimita locacion','delimita ingreso',
    'corre equipo','retira cabeza de aib','desmonta cabeza de aib',
    'coloca cabeza de aib','acerca cabeza de aib','monta cabeza de aib',
    'coloca aib en marcha','desempaqueta pozo','desmonta pileta',
    'carga en scada','fin de pozo en sistema','realiza desenganche de pozo',
    'desenganche de pozo',
    'desmonta eq.','demonta eq.','desmonta eq ','demonta eq ',
    'corre cuadro de maniobra','sanea sector de montaje',
    'arrima cabeza de aib','acerca cabeza de aib con guinche',
    'coloca v/b de maniobra + realiza enganche',
    'realiza enganche de pozo']

RIGS_KW = ['simulacro','sistema secra','limitador de carrera','crown-o-matic',
    'crowno-matic','sistema de frenos','chequeo general de equipo','chequeo preoperativo',
    'chequeos de eq pre jornada','chequeos de rutina','reune al personal',
    'reunion con personal','realiza reunion','espera asf','en espera de asf',
    'espera hot oil','espera camion','espera tbg','espera operador','espera quimico',
    'espera cia','en espera de abastecedor','espera abastecedor',
    'en espera de luz diurna','controla presion estatica','observa presion estatica',
    'continua obs psi','continua observando presion','observa pozo cerrado',
    'observa pozo sin venteo','mide con detector de 4 gases','controla fluencia',
    'observa pozo desplaza','detencion de tarea','detiene tarea',
    'se detiene la tarea','rellena con tierra','rellena zona de atraque',
    'acondiciona htas para entrega','consigna htas para entrega',
    'espera diseno','espera diseño','espera el diseño',
    'supervisor se retira','supervisor se dirige',
    'realiza gdt a personal',
    'bandejones de tbg',
    'limpieza de bop']

RIG_KW = ['limpieza de htas op','limpieza a htas op','limpia htas op',
    'limpieza de htas operativas','limpieza a htas operativas',
    'abundante limpieza de htas','realiza limpieza de htas',
    'realiza abundante limpieza','limpia herramientas operativas',
    'borde ecologico','bandeja ecologica','borde con tierra',
    'levanta suelo cont','retira suelo cont','prueba de hermeticidad',
    'carrera de calibracion','carrera de desagote','pistoneo','porta copa',
    'barra maestra','elementos de pistoneo','libra ancla','posiciona ancla',
    'fija ancla','sopletea','seca pines','lava pines','lava varillas',
    'circulando','circula pozo','pescador','pesca con resultado','arma pescador',
    'cap tester','cup tester','arroja probador','larga probador',
    'prueba de funcionamiento','bop de varillas','bop de v/b','instala bop',
    'monta bop','circuito hco','circuito hidraulico','llave hidraulica',
    'llave foster','llave wesco','llave oil country','acondiciona htas operativas',
    'baja cable','bajando cable','comienza a bajar cable','continua bajando cable',
    'sube cable','corre cable','prueba de bop','prueba bop','cierre parcial',
    'cierre total bop','prueba de cierre','chequea bop',
    'invierte carro','invierte caro',
    'afloja corrida','retira brida',
    'coloca niple','coloca colgador','coloca brida',
    'desenfunda bba','profundiza bba','profundizando',
    'agrega v/b','agrega (','retira v/b de maniobra',
    'coloca v/b de maniobra','retira tapas','guarda cuplas',
    'tensiona hta','tenciona hta','tensiona herramienta',
    'realiza maniobra de pesca','maniobra de pesca',
    'arma diseño','desarma diseño','arma en superficie',
    'coloca dispositivo','retira dispositivo',
    'observa ancla','realiza video',
    'limpieza de bop',
    'bandejones de tbg',
    'desconecta circuito','retira bop de tbg',
    'coloca brida v','ajusta corrida',
    'arma diseño final','arma diseño compuesto','arma diseño de',
    'arma en superficie','desarma diseño',
    'coloca instalacion','arma instalacion']

VIENTO_KW = ['viento','rafaga','km/h','kms/h','temporal de lluvia','temporal de nieve',
              'factor climatico','condicion climatica','retira nieve']

OPERA_CLIMA = ['continua sacando','continua bajando','sigue sacando','sigue bajando',
               'continua operando','opera con temporal','opera con viento',
               'operando con viento','operando con temporal','opera a pesar',
               'continua con maniobra','disminucion de viento','observa mejora de viento']

def patch_xlsx_unit_rotary_si(original_bytes, row_values):
    import zipfile as zf, re as _re
    zin = zf.ZipFile(io.BytesIO(original_bytes))
    ss_xml = zin.read('xl/sharedStrings.xml').decode('utf-8')
    sis = _re.findall(r'<si>(.*?)</si>', ss_xml, _re.DOTALL)
    tarifa_ss = {}
    for i, si in enumerate(sis):
        txt = _re.sub(r'<[^>]+>', '', si).strip()
        if txt in ['DTM','RIG','RIGS','RIGWOW','RIGNC','THR']:
            tarifa_ss[txt] = i
    wb_xml   = zin.read('xl/workbook.xml').decode('utf-8')
    rels_xml = zin.read('xl/_rels/workbook.xml.rels').decode('utf-8')
    rid_m = _re.search(r'<sheet[^>]+name="TIME SUMMARY"[^>]+r:id="([^"]+)"', wb_xml)
    if not rid_m: return original_bytes
    rid   = rid_m.group(1)
    tgt_m = _re.search(rf'Id="{_re.escape(rid)}"[^>]+Target="([^"]+)"', rels_xml)
    if not tgt_m: return original_bytes
    sheet_file = 'xl/' + tgt_m.group(1)
    sheet_xml  = zin.read(sheet_file).decode('utf-8')
    for excel_row, tarifa in row_values.items():
        if tarifa not in tarifa_ss: continue
        ss_idx   = tarifa_ss[tarifa]
        cell_ref = f'S{excel_row}'
        pat = _re.compile(
            rf'<c r="{_re.escape(cell_ref)}"([^>]*?)(?:\s*t="[^"]*")?([^>]*?)>.*?</c>',
            _re.DOTALL)
        m = pat.search(sheet_xml)
        if m:
            attrs = (m.group(1) + ' ' + m.group(2)).strip()
            attrs = _re.sub(r'\s*t="[^"]*"', '', attrs).strip()
            new_c = (f'<c r="{cell_ref}" {attrs} t="s"><v>{ss_idx}</v></c>'
                     if attrs else f'<c r="{cell_ref}" t="s"><v>{ss_idx}</v></c>')
            sheet_xml = sheet_xml[:m.start()] + new_c + sheet_xml[m.end():]
    buf = io.BytesIO()
    with zf.ZipFile(buf, 'w') as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            ni   = zf.ZipInfo(info.filename)
            ni.compress_type = info.compress_type
            ni.date_time     = info.date_time
            ni.external_attr = info.external_attr
            if info.filename == sheet_file:
                ni.compress_type = zf.ZIP_DEFLATED
                data = sheet_xml.encode('utf-8')
            zout.writestr(ni, data)
    return buf.getvalue()


LIMITES_DTM = {
    'BM': (
        ['retiro de vastago','retira vastago','desengancha pozo','desenganche de pozo',
         'retiro vastago','saca vastago','retira el vastago',
         'desempaqueta vastago','desempaqueta el vastago','desempaqueta  vastago',
         'retira ratigan','retira el ratigan','saca ratigan',
         'tensiona herramienta','tensiona hta','tenciona hta'],
        ['coloca vastago','levanta y coloca vastago','asienta vastago','coloca el vastago',
         'arma vastago','arma el vastago','coloca vastago nuevo','arma vastago nuevo',
         'coloca disco y celda','coloca en boca de pozo ajusta y torque',
         'coloca v/b de maniobra','ajusta y torque'],
    ),
    'BES': (
        ['tensiona colgador','tenciona colgador','desempaqueta pozo',
         'tensiona herramienta','tenciona herramienta'],
        ['asienta colgador','empaqueta pozo','empaqueta el pozo','asienta y empaqueta',
         'empaqueta c/','asienta el colgador','sienta colgador'],
    ),
    'INY': (
        ['tensiona colgador','tenciona colgador','desempaqueta pozo',
         'tensiona herramienta','tenciona herramienta'],
        ['asienta colgador','empaqueta pozo','empaqueta el pozo','asienta y empaqueta','empaqueta c/'],
    ),
    'PCP': (
        ['saca stuffing box','retira stuffing box','sacar stuffing box',
         'stuffing box y vastago','saca stafing','retira stafing'],
        ['coloca vastago','coloca stuffing box','coloca stafing','coloca el vastago'],
    ),
}

# ── Reglas duras ──────────────────────────────────────────────────────────────
def regla_dura(t, ph, hrs, pileta):
    phU = ph.upper().strip()

    if 'simops' in t:
        pass  # SIMOPS sin RIG claro: continuar al KNN

    if hay(t,'espera luz diurna','en espera de luz diurna','espera luz dirna','luz diurna'):
        return 'DTM', 'Espera de luz diurna → DTM'

    if hay(t,'tarifa horaria reducida','conflicto gremial','paro gremial','medida gremial'):
        return 'THR', 'THR → revisión manual'

    if hay(t,'soldadura','trabajo de soldadura'):
        return 'RIGNC', 'Soldadura → RIGNC'
    # Mantenimiento mecánico del equipo → RIGNC (pero no si es solo una charla GDT)
    if hay(t,'personal de mantenimiento mecanico','mantenimiento mecanico del equipo',
              'control de cadena de transmision','cadena de transmision'):
        if not hay(t,'gdt a personal','charla'):
            return 'RIGNC', 'Mantenimiento mecánico → RIGNC'

    # Clima vs operación
    opera = any(p in t for p in OPERA_CLIMA)
    parado = hay(t,'parado por viento','detenido por viento','detiene maniobra por viento',
                 'detencion de tarea por viento','parado por lluvia','parado por nieve',
                 'eq parado por factor climatico')
    if opera and any(p in t for p in ['viento','temporal','lluvia','nieve']):
        return 'RIG', 'Opera con clima adverso → RIG'
    if parado:
        return 'RIGWOW', 'Parada por clima'
    if hay(t,'km/h','kms/h') and hay(t,'viento','rafaga') and not opera:
        return 'RIGWOW', 'Registro velocidad de viento'
    if hay(t,'factor climatico','condicion climatica') and not opera:
        return 'RIGWOW', 'Condición climática'
    if hay(t,'temporal de lluvia','temporal de nieve') and not opera:
        return 'RIGWOW', 'Temporal → RIGWOW'
    if hay(t,'retira nieve','vierte sal para evitar congelamiento'):
        return 'RIGWOW', 'Condición invernal'

    if hay(t,'detiene transporte','detiene el transporte','detiene viaje'):
        return 'RIGS', 'Detiene transporte → RIGS (espera)'

    if 'apit' in t.split() or ' apit ' in t or t.startswith('apit') or t.endswith('apit'):
        return 'RIGS', 'APIT → RIGS'

    # GDT con cia externa
    if 'gdt' in t and any(c in t for c in CIAS):
        return 'RIGS', 'GDT de compañía externa → RIGS'

    # Charlas
    es_charla = hay(t,'charla de inicio','charla de inidio','cambio de turno',
                    'inicio de jornada','inicio jornada','lectura de ats','lectura al ats')
    if es_charla:
        if phU in ('PRE','POST'): return 'DTM', f'Charla en {phU} → DTM'
        if phU: return 'RIGS', f'Charla en {phU} → RIGS'

    if any(k in t for k in DTM_KW):
        return 'DTM', 'Regla contractual DTM'

    # Presiones
    es_p = hay(t,'despresuriza','toma presion','toma presiones','controla presiones',
               'registro de presiones')
    if es_p and not hay(t,'atraca','saca','baja','inyect','bbea'):
        tar = 'DTM' if hrs <= 0.5 else 'RIGS'
        return tar, f'Presiones {hrs}h → {tar}'

    # Espacea bba
    if hay(t,'espacea','espaceando') and hay(t,'bba','bomba','aparejo'):
        tar = 'DTM' if hrs <= 0.5 else 'RIG'
        return tar, f'Espacea bba {hrs}h → {tar}'

    # Prueba producción
    if hay(t,'prueba de produccion','prueba produccion') and 'hermeticidad' not in t:
        tar = 'DTM' if hrs <= 0.5 else 'RIGS'
        return tar, f'Prueba producción {hrs}h → {tar}'

    if any(c in t for c in CIAS):
        return 'RIGS', 'Compañía externa → RIGS'

    if any(k in t for k in RIGS_KW):
        return 'RIGS', 'Regla RIGS'

    if hay(t,'operador de pileta','pileta y bba','pileta y bomba'):
        tar = 'RIG' if pileta else 'RIGS'
        return tar, f'Pileta {"propia" if pileta else "rentada"} → {tar}'

    if hay(t,'inyecta','bbea','bombea') and hay(t,'asf','agua de formacion'):
        tar = 'RIG' if pileta else 'RIGS'
        return tar, f'ASF pileta {"propia" if pileta else "rentada"} → {tar}'

    if any(k in t for k in RIG_KW):
        return 'RIG', 'Regla RIG'

    # Stuffing box (todas las variantes) → RIGS si está solo (fin DTM POST o POOHI)
    if hay(t, 'stuffing box', 'stafing box', 'staffing box', 'stufing box',
              'staffing', 'stufing'):
        if not any(p in t for p in ['coloca vastago', 'retira vastago', 'saca vastago']):
            return 'RIGS', 'Stuffing box (operación) → RIGS'

    # Saca/baja/sacando/bajando + v/b o tbg o tubing → RIG
    # Tolera números intercalados: "saca (55) v/b", "continua bajando (30) tbg"
    import re
    tiene_movimiento = re.search(r'\b(saca|baja|sacando|bajando|continua sacando|comienza a sacar|'
                                 r'continua bajando|comienza a bajar|cont\. bajando|cont\. sacando)\b', t)
    tiene_material   = re.search(r'\b(v/b|varilla|varillas|tbg|tubing|tbgs|barras huecas)\b', t)
    if tiene_movimiento and tiene_material:
        return 'RIG', 'Saca/baja + material → RIG'

    # Profundiza + material → RIG
    tiene_profundiza = re.search(r'\b(profundiza|profundizando|continua profundizando)\b', t)
    if tiene_profundiza and re.search(r'\b(v/b|varilla|bba|bomba|tbg|bh\b|barras)', t):
        return 'RIG', 'Profundiza + material → RIG'

    return None, None

def detectar_ambiguedad(t, pileta):
    cats = set()
    if any(k in t for k in DTM_KW): cats.add('DTM')
    if any(k in t for k in RIGS_KW): cats.add('RIGS')
    if any(k in t for k in RIG_KW): cats.add('RIG')
    if any(c in t for c in CIAS): cats.add('RIGS')
    if hay(t,'operador de pileta','pileta y bba') or \
       (hay(t,'inyecta','bbea','bombea') and 'asf' in t):
        cats.add('RIG' if pileta else 'RIGS')
    if any(p in t for p in VIENTO_KW): cats.add('RIGWOW')
    return 'RIG' in cats and 'RIGS' in cats

# ── KNN ───────────────────────────────────────────────────────────────────────
TRAINING_PATH = os.path.join(os.path.dirname(__file__), 'training_data.json')

@st.cache_resource
def cargar_modelo():
    if not os.path.exists(TRAINING_PATH):
        return None
    with open(TRAINING_PATH, encoding='utf-8') as f:
        data = json.load(f)
    textos = [normalizar(d.get('ph','') + ' ' + d.get('op','')) for d in data]
    labels = [d['si'] for d in data]
    ops = [d.get('op','') for d in data]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=1, sublinear_tf=True)
    X = vec.fit_transform(textos)
    return {'vec': vec, 'X': X, 'labels': labels, 'ops': ops}

def knn_predict(op, ph, modelo, k=7):
    t = normalizar(ph + ' ' + op)
    xq = modelo['vec'].transform([t])
    sims = cosine_similarity(xq, modelo['X'])[0]
    top = np.argsort(sims)[::-1][:k]
    votos = Counter()
    for i in top: votos[modelo['labels'][i]] += sims[i]
    tarifa = votos.most_common(1)[0][0]
    total = sum(votos.values())
    pct = votos[tarifa]/total if total > 0 else 0
    sim_max = float(sims[top[0]])
    if sim_max >= 0.85 and pct >= 0.7: conf = 'alta'
    elif sim_max >= 0.6 and pct >= 0.5: conf = 'media'
    else: conf = 'revision'
    return tarifa, conf, f'KNN sim={sim_max:.2f} consenso={pct:.0%}'

def clasificar_fila(op, ph, hrs, pileta, modelo):
    """
    Returns: (tarifa, confianza, motivo, flag_simops)
    flag_simops=True si hay SIMOPS → columna separada de revisión aunque se clasifique
    """
    import re
    t = normalizar(op)
    phU = ph.upper().strip()
    es_pre_post = phU in ('PRE','POST')
    flag_simops = 'simops' in t

    # SIMOPS: si hay operación RIG activa → RIG; si no → KNN decide, siempre baja confianza
    if flag_simops:
        tiene_mov = re.search(r'\b(saca|baja|sacando|bajando|continua sacando|'
                              r'comienza a sacar|continua bajando|comienza a bajar|'
                              r'bbea|inyecta|bombea|profundiza)\b', t)
        tiene_mat = re.search(r'\b(v/b|varilla|varillas|tbg|tubing|barras huecas|asf)\b', t)
        tiene_rig = any(k in t for k in RIG_KW)
        if (tiene_mov and tiene_mat) or tiene_rig:
            return 'RIG', 'revision', 'SIMOPS con operación RIG activa', True
        # Para SIMOPS sin RIG claro, dejar que continúe al KNN con flag

    # Ambigüedad RIG+RIGS — el KNN decide, confianza baja

    # Regla dura
    tarifa, motivo = regla_dura(t, ph, hrs, pileta)
    if tarifa:
        if es_pre_post and tarifa == 'RIG':
            # En PRE/POST no puede ser RIG — forzar RIGS como mejor alternativa
            return 'RIGS', 'revision', f'Phase {phU}: RIG → corregido a RIGS (revisar)', False
        if not es_pre_post and phU and tarifa == 'DTM':
            # Fuera de PRE/POST no puede ser DTM — continuar al KNN
            tarifa = None  # forzar caída al KNN
        if tarifa:
            conf = 'revision' if tarifa in ('THR','RIGNC') else 'alta'
            return tarifa, conf, motivo, False

    # KNN
    if modelo:
        tarifa, conf, motivo = knn_predict(op, ph, modelo)
        if es_pre_post and tarifa == 'RIG':
            return 'RIGS', 'revision', f'Phase {phU}: RIG → corregido a RIGS (revisar)', False
        if not es_pre_post and phU and tarifa == 'DTM':
            return 'RIGS', 'revision', f'Phase {phU}: DTM fuera de PRE/POST → RIGS (revisar)', False
        return tarifa, conf, motivo, False

    return 'RIGS', 'revision', 'Sin modelo — clasificación por defecto (revisar)', False

def validar_limites(df_pre, df_post, sistema_habia='BM', sistema_instala='BM'):
    alertas = []
    # PRE: límite según sistema que HABÍA (fin de intervención anterior)
    kw_fin, _  = LIMITES_DTM.get(sistema_habia, ([],[]))
    # POST: límite según sistema que se INSTALA (inicio de siguiente DTM)
    _, kw_inicio = LIMITES_DTM.get(sistema_instala, ([],[]))

    if not df_pre.empty:
        op = normalizar(str(df_pre.iloc[-1].get('Operation','')))
        if not any(k in op for k in kw_fin):
            texto = str(df_pre.iloc[-1].get('Operation',''))[:80]
            alertas.append(('warn',
                f'Posible error en fin de PRE ({sistema_habia}): "{texto}"'))
    if not df_post.empty:
        op = normalizar(str(df_post.iloc[0].get('Operation','')))
        if not any(k in op for k in kw_inicio):
            texto = str(df_post.iloc[0].get('Operation',''))[:80]
            alertas.append(('warn',
                f'Posible error en inicio de POST ({sistema_instala}): "{texto}"'))
    return alertas

# ── App ───────────────────────────────────────────────────────────────────────
def main():
    st.title("🛢️ Certificador IA - Pulling")
    st.markdown('<p style="color:#cc0000;font-size:22px;font-weight:bold;margin-top:-10px">⚠️ Versión de entrenamiento</p>', unsafe_allow_html=True)

    modelo = cargar_modelo()

    # Sidebar
    with st.sidebar:
        st.header("🔍 Filtros")
        filtro_confianza = st.multiselect(
            "Confianza",
            ['alta', 'media', 'baja'],
            default=['alta', 'media', 'baja']
        )
        solo_rev     = st.checkbox("Solo revisiones (confianza baja)")
        solo_diff_ur = st.checkbox("Diferencia CoMan vs App")
        busqueda     = st.text_input("Buscar en operación", "")

    # ── Configuración del pozo ───────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        tipo_pulling = st.selectbox(
            "Tipo de equipo",
            ["🏗️ Pulling Pesado", "🔧 Pulling Liviano"],
            index=0,
            help="Pesado = pileta propia (equipo 080/081) · Liviano = sin pileta propia"
        )
        pileta = "Pesado" in tipo_pulling
    with c2:
        sistema_habia = st.selectbox(
            "Sistema que **había**",
            ["BM", "BES", "INY", "PCP"],
            help="Sistema de extracción existente al iniciar la intervención"
        )
    with c3:
        sistema_instala = st.selectbox(
            "Sistema que se **instala**",
            ["BM", "BES", "INY", "PCP"],
            help="Sistema de extracción que se deja al finalizar"
        )
    st.divider()

    archivo = st.file_uploader("Subir reporte OpenWells (TIME SUMMARY)",
                                type=['xlsx','xls'])
    if not archivo:
        st.info("👆 Subí el Excel de OpenWells para comenzar")
        return

    try:
        # Leer y cachear los bytes — necesario para el export en Streamlit Cloud
        archivo_bytes = archivo.read()
        # Si cambió el archivo, resetear correcciones guardadas
        if st.session_state.get('archivo_nombre') != archivo.name:
            st.session_state['correcciones_super'] = {}
        st.session_state['archivo_bytes'] = archivo_bytes
        st.session_state['archivo_nombre'] = archivo.name
        xls = pd.ExcelFile(io.BytesIO(archivo_bytes))
        if 'TIME SUMMARY' not in xls.sheet_names:
            st.error(f"No se encontró 'TIME SUMMARY'. Hojas: {xls.sheet_names}")
            return
        df_raw = pd.read_excel(io.BytesIO(archivo_bytes), sheet_name='TIME SUMMARY', header=0)
    except Exception as e:
        st.error(f"Error al leer archivo: {e}")
        return

    df = df_raw[df_raw['Operation'].notna() &
                (df_raw['Operation'].astype(str).str.strip().str.len() > 3)].copy().reset_index(drop=True)

    if len(df) == 0:
        st.error("No se encontraron filas con operación.")
        return

    st.success(f"**{archivo.name}** — {len(df)} filas")

    # Validación de límites PRE/POST
    df_pre  = df[df['Phase'].astype(str).str.upper().str.strip() == 'PRE']
    df_post = df[df['Phase'].astype(str).str.upper().str.strip() == 'POST']
    # PRE usa sistema que HABÍA, POST usa sistema que se INSTALA
    alertas = validar_limites(df_pre, df_post, sistema_habia, sistema_instala)
    if alertas:
        with st.expander(f"⚠️ Posibles errores de fase detectados ({len(alertas)}) — el análisis continúa igual"):
            for _, msg in alertas:
                st.warning(msg)
    elif not df_pre.empty or not df_post.empty:
        partes = []
        if not df_pre.empty: partes.append(f"PRE ok ({len(df_pre)} filas)")
        if not df_post.empty: partes.append(f"POST ok ({len(df_post)} filas)")
        st.success(f"✅ Límites DTM validados — {' · '.join(partes)}")

    # Clasificar
    with st.spinner("Clasificando..."):
        rows = []
        for _, row in df.iterrows():
            op    = str(row.get('Operation','') or '')
            ph    = str(row.get('Phase','') or '')
            hrs   = float(row['Hrs']) if pd.notna(row.get('Hrs')) else 0.0
            comp  = str(row.get('Unit Rotary','') or '')
            fecha = str(row.get('Report date','') or '')[:10]
            from_ = str(row.get('From','') or '')
            to_   = str(row.get('To','') or '')
            code  = str(row.get('Code','') or '')
            task  = str(row.get('Task','') or '')
            activ = str(row.get('Activity','') or '')
            npt   = str(row.get('NPT','') or '')
            nptd  = str(row.get('NPT Detail','') or '')
            try:
                tar, conf, mot, sim_flag = clasificar_fila(op, ph, hrs, pileta, modelo)
            except Exception as _e:
                tar, conf, mot, sim_flag = 'RIGS', 'revision', f'Error interno: {_e}', False
            rows.append({
                'Fecha':fecha,'From':from_,'To':to_,'Hrs':hrs,
                'Code':code,'Phase':ph,'Task':task,'Activity':activ,
                'Company':comp,'NPT':npt,'NPT Detail':nptd,
                'Operación':op,'App':tar,
                'Confianza':conf,'Motivo':mot,
                '_diff':False,'_simops':sim_flag
            })
    df_res = pd.DataFrame(rows)


    # Filtrar
    conf_map = {'alta':'alta', 'media':'media', 'revision':'baja'}
    df_res['_conf_label'] = df_res['Confianza'].map(conf_map).fillna('baja')

    mask = df_res['_conf_label'].isin(filtro_confianza)
    if solo_rev:     mask &= (df_res['Confianza'] == 'revision')
    if solo_diff_ur: mask &= (df_res['Company'] != df_res['App'])
    if busqueda:     mask &= df_res['Operación'].str.contains(busqueda, case=False, na=False)
    df_view = df_res[mask].copy()
    st.caption(f"Mostrando {len(df_view)} de {len(df_res)} filas")

    # ── Tabla interactiva ─────────────────────────────────────────────
    TARIFAS = ['DTM','RIG','RIGS','RIGWOW','RIGNC','THR','?']
    CONF_ICON = {'alta':'✅','media':'⚠️','revision':'🔴'}




    # Emojis de color por tarifa
    COLOR_SYM = {
        'DTM':    '🟢 DTM',
        'RIG':    '🟩 RIG',
        'RIGS':   '🔵 RIGS',
        'RIGWOW': '🟡 RIGWOW',
        'RIGNC':  '🔴 RIGNC',
        'THR':    '🟥 THR',
    }
    TARIFAS_COLOR = [
        '🟢 DTM',
        '🟩 RIG',
        '🔵 RIGS',
        '🟡 RIGWOW',
        '🔴 RIGNC',
        '🟥 THR',
    ]

    df_view = df_view.copy()

    # Columnas con emoji de color
    df_view['App_c']     = df_view['App'].map(lambda x: COLOR_SYM.get(x, x))
    df_view['Company_c'] = df_view['Company'].map(lambda x: COLOR_SYM.get(x, x) if x in COLOR_SYM else x)
    df_view['Super_c']   = df_view['App'].map(lambda x: COLOR_SYM.get(x, x))  # arranca = App

    # Estado: solo texto, sin iconos
    df_view['Estado'] = df_view.apply(
        lambda r: ('SIMOPS · ' if r.get('_simops') else '') +
                  ('baja'  if r['Confianza'] == 'revision' else
                   'media' if r['Confianza'] == 'media'    else 'alta'),
        axis=1
    )

    # Restaurar correcciones guardadas en session_state
    saved = st.session_state.get('correcciones_super', {})
    def get_saved_super(idx):
        return saved.get(idx, {}).get('Super_c', df_view.loc[idx, 'App_c'])
    def get_saved_obs(idx):
        return saved.get(idx, {}).get('Observación', '')

    df_view['Observación'] = df_view.index.map(get_saved_obs)
    df_view['Super_c']     = df_view.index.map(get_saved_super)

    df_show = df_view[['From','To','Hrs','Code','Phase','Task','Activity',
                        'NPT','NPT Detail','Operación',
                        'Company_c','App_c','Super_c','Estado','Observación']].copy()

    edited = st.data_editor(
        df_show,
        column_config={
            'From':        st.column_config.TextColumn('From', width=58, disabled=True),
            'To':          st.column_config.TextColumn('To', width=58, disabled=True),
            'Hrs':         st.column_config.NumberColumn('Hrs', width=45, format="%.2f", disabled=True),
            'Code':        st.column_config.TextColumn('Code', width=55, disabled=True),
            'Phase':       st.column_config.TextColumn('Phase', width=62, disabled=True),
            'Task':        st.column_config.TextColumn('Task', width=62, disabled=True),
            'Activity':    st.column_config.TextColumn('Activity', width=70, disabled=True),
            'Company_c':   st.column_config.TextColumn('Unit Rotary', width=100, disabled=True),
            'NPT':         st.column_config.TextColumn('NPT', width=50, disabled=True),
            'NPT Detail':  st.column_config.TextColumn('NPT Detail', width=80, disabled=True),
            'Operación':   st.column_config.TextColumn('Operation', width='large', disabled=True),
            'App_c':       st.column_config.TextColumn('🤖 App', width=100, disabled=True),
            'Super_c':     st.column_config.SelectboxColumn(
                               '✏️ Super', width=110, options=TARIFAS_COLOR,
                               help='Arranca igual a la App. Cambiá si la app se equivocó.'
                           ),
            'Estado':      st.column_config.TextColumn('Confianza', width=80, disabled=True),
            'Observación': st.column_config.TextColumn(
                               '📝 Observación', width=200,
                               help='Escribí por qué la app se equivocó o cualquier aclaración.'
                           ),
        },
        hide_index=True,
        use_container_width=True,
        height=620,
        key='tabla_principal'
    )

    idx_vis_list = list(df_view.index)
    if 'correcciones_super' not in st.session_state:
        st.session_state['correcciones_super'] = {}

    # ── Botón guardar cambios ─────────────────────────────────────────────────
    col_save, col_info = st.columns([1, 4])
    with col_save:
        guardar = st.button("💾 Guardar cambios", type="primary", use_container_width=True)
    with col_info:
        n_guardadas = len(st.session_state['correcciones_super'])
        if n_guardadas:
            st.info(f"✅ {n_guardadas} correcciones guardadas. Podés filtrar sin perder los cambios.")
        else:
            st.caption("Editá la columna Super y guardá antes de filtrar.")

    # Solo guardar si el super apretó el botón
    if guardar:
        edited_df = edited.copy()
        for _vi, (_, _row_ed) in enumerate(edited_df.iterrows()):
            if _vi < len(idx_vis_list):
                _orig_pos  = idx_vis_list[_vi]
                _sc  = _row_ed.get('Super_c', '')
                _obs = _row_ed.get('Observación', '')
                _app_c     = df_res.iloc[_orig_pos]['App'] if _orig_pos < len(df_res) else ''
                _app_emoji = COLOR_SYM.get(_app_c, _app_c)
                if _sc != _app_emoji or _obs:
                    st.session_state['correcciones_super'][_orig_pos] = {
                        'Super_c': _sc, 'Observación': _obs
                    }
                elif _orig_pos in st.session_state['correcciones_super']:
                    del st.session_state['correcciones_super'][_orig_pos]
        n_corr = len(st.session_state['correcciones_super'])
        st.success(f"✅ Cambios guardados — {n_corr} correcciones en total")
        st.rerun()

    # Para estadísticas y export, usar edited directamente
    edited_df = edited.copy()

    # ── Estadísticas — mostrar si hay correcciones guardadas ─────────────────
    if st.session_state.get('correcciones_super'):
        st.divider()
        _ss  = st.session_state['correcciones_super']
        _tot = len(df_res)
        _mods = {'alta':0,'media':0,'revision':0}
        _EM = {'🟢 DTM':'DTM','🟩 RIG':'RIG','🔵 RIGS':'RIGS',
               '🟡 RIGWOW':'RIGWOW','🔴 RIGNC':'RIGNC',
               '🟥 THR':'THR','⚪ REVISAR':'RIGS'}
        for _pos, _vals in _ss.items():
            if _pos < len(df_res):
                _corr = _EM.get(_vals.get('Super_c',''), _vals.get('Super_c',''))
                _app  = df_res.iloc[_pos]['App']
                _cf   = df_res.iloc[_pos]['Confianza']
                if _corr and _corr != _app:
                    if _cf in _mods: _mods[_cf] += 1
        _mod_t = sum(_mods.values())
        _ta = len(df_res[df_res['Confianza']=='alta'])
        _tm = len(df_res[df_res['Confianza']=='media'])
        _tb = len(df_res[df_res['Confianza']=='revision'])
        _et = (_tot - _mod_t) / _tot * 100 if _tot else 0
        _ea = (_ta - _mods['alta'])     / _ta * 100 if _ta else 100
        _em = (_tm - _mods['media'])    / _tm * 100 if _tm else 100
        _eb = (_tb - _mods['revision']) / _tb * 100 if _tb else 100
        st.markdown('**📊 Performance de la clasificación**')
        sc0, sc1, sc2, sc3 = st.columns(4)
        sc0.metric('🎯 Global',          f'{_tot-_mod_t} / {_tot}',  f'{_et:.1f}% éxito')
        sc1.metric('✅ Alta confianza',   f'{_ta-_mods["alta"]} / {_ta}',  f'{_ea:.1f}% éxito')
        sc2.metric('⚠️ Media confianza',  f'{_tm-_mods["media"]} / {_tm}', f'{_em:.1f}% éxito')
        sc3.metric('🔴 Baja confianza',   f'{_tb-_mods["revision"]} / {_tb}', f'{_eb:.1f}% éxito')
        st.divider()

    # ── Calcular estadísticas de performance (para export) ───────────────────

    _EMOJI_MAP = {'🟢 DTM':'DTM','🟩 RIG':'RIG','🔵 RIGS':'RIGS',
                  '🟡 RIGWOW':'RIGWOW','🔴 RIGNC':'RIGNC',
                  '🟥 THR':'THR','⚪ REVISAR':'RIGS'}
    total_filas  = len(df_res)
    modificadas  = 0
    mods_por_conf = {'alta':0,'media':0,'revision':0}
    for _vi, (_, _re) in enumerate(edited_df.iterrows()):
        if _vi < len(idx_vis_list):
            _pos = idx_vis_list[_vi]
            if _pos < len(df_res):
                _app = df_res.iloc[_pos]['App']
                _sup = _EMOJI_MAP.get(_re.get('Super_c',''), _re.get('Super_c',''))
                _cf  = df_res.iloc[_pos]['Confianza']
                if _sup and _sup != _app and _sup != '?':
                    modificadas += 1
                    if _cf in mods_por_conf:
                        mods_por_conf[_cf] += 1
    tot_alta  = len(df_res[df_res['Confianza']=='alta'])
    tot_media = len(df_res[df_res['Confianza']=='media'])
    tot_baja  = len(df_res[df_res['Confianza']=='revision'])

    # Export
    st.divider()

    EMOJI_TO_CODE = {
        '🟢 DTM':'DTM','🟩 RIG':'RIG','🔵 RIGS':'RIGS',
        '🟡 RIGWOW':'RIGWOW','🔴 RIGNC':'RIGNC',
        '🟥 THR':'THR','⚪ REVISAR':'?',
    }

    idx_orig     = df[df['Operation'].notna()].index.tolist()
    idx_visibles = list(df_view.index)

    # Correcciones: session_state tiene TODAS las filas (incluso las filtradas)
    saved_ss = st.session_state.get('correcciones_super', {})
    correcciones  = {}
    observaciones = {}
    for pos, vals in saved_ss.items():
        se = vals.get('Super_c', '')
        correcciones[pos]  = EMOJI_TO_CODE.get(se, se)
        observaciones[pos] = vals.get('Observación', '')
    # Sobreescribir con valores actuales del editor (filas visibles)
    for vi, (_, row_ed) in enumerate(edited_df.iterrows()):
        if vi < len(idx_visibles):
            orig_pos = idx_visibles[vi]
            se = row_ed.get('Super_c', '')
            correcciones[orig_pos]  = EMOJI_TO_CODE.get(se, se)
            observaciones[orig_pos] = row_ed.get('Observación', '')
    # Filas sin corrección: usar la App
    for pos, row_res in df_res.iterrows():
        if pos not in correcciones:
            correcciones[pos] = row_res['App']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**⬇️ Excel original revisado**")
        st.caption("Mismo archivo con todos los formatos, solo Unit Rotary SI completada")
        if st.button('Preparar Excel revisado', type='primary', use_container_width=True):
            # Usar los bytes cacheados al momento de subir el archivo
            original_bytes = st.session_state.get('archivo_bytes', b'')
            if not original_bytes:
                st.error('Error: no se encontraron los bytes del archivo. Volvé a subir el Excel.')
                st.stop()
            row_values = {}
            for pos, corr in correcciones.items():
                if corr and corr != '?' and pos < len(idx_orig):
                    row_values[idx_orig[pos] + 2] = corr
            patched = patch_xlsx_unit_rotary_si(original_bytes, row_values)
            nombre  = archivo.name.rsplit('.',1)[0]
            st.download_button(
                label='📥 Descargar Excel revisado',
                data=patched,
                file_name=nombre+'_REVISADO.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='dl_excel'
            )


    with col2:
        st.markdown("**📊 Tabla para retroalimentar**")
        st.caption("Lo que se ve en la tabla + correcciones + observaciones")
        if st.button("Preparar tabla de retroalimentación", use_container_width=True):
            retro = []
            for vi, (_, row_ed) in enumerate(edited_df.iterrows()):
                if vi < len(idx_visibles):
                    op = idx_visibles[vi]
                    rr = df_res.iloc[op] if op < len(df_res) else pd.Series()
                    se = row_ed.get('Super_c','')
                    corr = EMOJI_TO_CODE.get(se, se)
                    obs  = row_ed.get('Observación','')
                    ce   = row_ed.get('Company_c','')
                    comp_code = EMOJI_TO_CODE.get(ce, ce)
                    retro.append({
                        'From':        row_ed.get('From',''),
                        'To':          row_ed.get('To',''),
                        'Hrs':         row_ed.get('Hrs',''),
                        'Code':        row_ed.get('Code',''),
                        'Phase':       row_ed.get('Phase',''),
                        'Task':        row_ed.get('Task',''),
                        'Activity':    row_ed.get('Activity',''),
                        'Unit Rotary': comp_code,
                        'NPT':         row_ed.get('NPT',''),
                        'NPT Detail':  row_ed.get('NPT Detail',''),
                        'Operation':   row_ed.get('Operación',''),
                        'App':         rr.get('App','') if len(rr) else '',
                        'Confianza':   rr.get('Confianza','') if len(rr) else '',
                        'Motivo':      rr.get('Motivo','') if len(rr) else '',
                        'Super':       corr,
                        'Observación': obs,
                        'SIMOPS':      'SI' if (rr.get('_simops') if len(rr) else False) else '',
                    })
            df_retro = pd.DataFrame(retro)

            # Hoja de estadísticas
            _ea = (tot_alta  - mods_por_conf['alta'])  / tot_alta  * 100 if tot_alta  else 100
            _em = (tot_media - mods_por_conf['media']) / tot_media * 100 if tot_media else 100
            _eb = (tot_baja  - mods_por_conf['revision']) / tot_baja * 100 if tot_baja else 100
            _et = (total_filas - modificadas) / total_filas * 100 if total_filas else 0
            stats_data = {
                'Métrica': [
                    'Total filas',
                    'Correctas sin modificar', 'Modificadas por super', '% éxito global',
                    'Alta confianza — total', 'Alta — correctas', 'Alta — modificadas', '% éxito Alta',
                    'Media confianza — total', 'Media — correctas', 'Media — modificadas', '% éxito Media',
                    'Baja confianza — total', 'Baja — correctas', 'Baja — modificadas', '% éxito Baja',
                ],
                'Valor': [
                    total_filas,
                    total_filas - modificadas, modificadas, f"{_et:.1f}%",
                    tot_alta,  tot_alta  - mods_por_conf['alta'],  mods_por_conf['alta'],  f"{_ea:.1f}%",
                    tot_media, tot_media - mods_por_conf['media'], mods_por_conf['media'], f"{_em:.1f}%",
                    tot_baja,  tot_baja  - mods_por_conf['revision'], mods_por_conf['revision'], f"{_eb:.1f}%",
                ]
            }
            df_stats = pd.DataFrame(stats_data)

            buf2 = io.BytesIO()
            with pd.ExcelWriter(buf2, engine='openpyxl') as writer:
                df_retro.to_excel(writer, sheet_name='Retroalimentacion', index=False)
                df_stats.to_excel(writer, sheet_name='Estadisticas', index=False)
            buf2.seek(0)
            nombre = archivo.name.rsplit('.',1)[0]
            st.download_button(
                label="📥 Descargar tabla retroalimentación",
                data=buf2,
                file_name=nombre+'_RETRO.xlsx',
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='dl_retro'
            )


if __name__ == '__main__':
    main()
