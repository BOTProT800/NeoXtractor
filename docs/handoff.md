# Handoff — exportación GLB con esqueleto y animaciones

Punto de continuación al 27 de septiembre de 2026. El plan completo, con el
razonamiento y las pruebas de cada decisión, está en
[plan-exportacion-glb.md](plan-exportacion-glb.md). Este documento es el resumen
operativo para retomar el trabajo en una sesión nueva.

## Dónde está todo

| | |
| --- | --- |
| Rama | `feat/glb-export`, 11 commits sobre `main` (`438da76`) |
| Remoto | `https://github.com/BOTProT800/NeoXtractor` — la rama está subida, `main` intacto |
| PR | sin abrir: `https://github.com/BOTProT800/NeoXtractor/pull/new/feat/glb-export` |
| Pruebas | 188 con Blender, 182 + 6 saltadas sin él (así corre CI) |

## Entorno que hace falta reconstruir

Nada de esto está en el repositorio y una sesión nueva no lo tendrá.

**Blender como paquete Python.** El Blender instalado es 2.79 y **no** tiene
importador glTF. Se usa `bpy` desde PyPI, en un entorno aparte porque pesa
~670 MB y va atado a CPython 3.11:

```
uv venv --python 3.11 C:/Users/vicen/AppData/Local/Temp/blv
uv pip install --python C:/Users/vicen/AppData/Local/Temp/blv/Scripts/python.exe bpy==4.5.14
```

La ruta debe ser **corta**. El árbol de addons de Blender es profundo y un
prefijo largo empuja `io_scene_gltf2` más allá del límite de 260 caracteres de
Windows, donde Python deja de ver parte del addon y el importador falla con un
`ModuleNotFoundError` engañoso. Con el scratchpad de la sesión la ruta llegaba
a 264 caracteres y fallaba por eso.

Para correr las pruebas de Blender:

```
set NEOX_BLENDER_PYTHON=C:/Users/vicen/AppData/Local/Temp/blv/Scripts/python.exe
uv run pytest
```

Sin esa variable las 6 pruebas de Blender se saltan y el resto pasa igual.

**Muestras.** El juego está en `E:\Instaladores\Main\Cyber Hunter`. Los NPK se
leen en el sitio, no hace falta copiarlos. `configs/cyber_hunter.json` ya tiene
las opciones correctas (`decryption_key: null`, `info_size: 28`).

- `res/npc.npk` → `tiejiayong/tiejiayong_03.mesh` y `tiejiayong/tiejiayong.gis`
  (el modelo con el que se validó todo)
- `res/character/transformers.npk` → `jianzao_dunpai.mesh` + `.gis`, 22 clips
- `res/character/male.npk` → 190 mallas y 2402 `.gis` de biblioteca compartida

`temp/` tiene el `.mesh` y el `.gis` de tiejiayong; está en `.gitignore`.

## Qué está hecho y verificado

**Primera meta (exportación GLB con esqueleto): cumplida** para la variante
verificada — versión 2 / tipo 1 / bone kind 1, contrastada con
`tiejiayong_03.mesh`. El usuario confirmó el esqueleto en su visor.

**Segunda meta (animaciones): funcionando.** Se descifró el formato `RGIS`
(`.gis`) desde cero, y los clips reales del juego se exportan y reproducen.

Verificación: lector independiente escrito contra la especificación, skinning
en CPU desde los accessors exportados, `pygltflib` y `trimesh` como lectores de
terceros, e importación y posado reales en Blender 4.5.14. En los tres modelos
probados, Blender y la evaluación en CPU **coinciden a 5e-06**.

## Decisiones que costaron y no hay que rehacer

Cada una se resolvió midiendo, no suponiendo. Están documentadas en el código
con su evidencia.

**Las matrices de hueso son globales en disposición de vector fila.** La
traslación está en la última fila. Se detecta de los datos comprobando cuál de
las dos ranuras es la afín `(0,0,0,1)`. Ver `core/mesh_converter/skeleton.py`.

**NeoX no necesita conversión de coordenadas.** Ya es diestro con Y arriba. Se
midió comparando la normal geométrica del bobinado contra las normales
guardadas: producto escalar medio `+0.99`, 100 % de triángulos a favor. Hubo un
periodo en que el exportador reflejaba X, heredado del visor sin comprobarlo;
eso producía una imagen especular que **el sombreado ocultaba** porque el
bobinado se invertía para compensar. `TestHandedness` lo cubre ahora.

**Los TRS de RGIS son relativos al padre y los cuaterniones van `(x,y,z,w)`.**
Contrastado contra el `.mesh`: error de traslación 2.68 leído como local frente
a 87.28 como global, y con `(x,y,z,w)` seis de diez huesos reproducen la
rotación local exactamente; con `(w,x,y,z)`, ninguno.

**Los huesos se enlazan por nombre, nunca por índice.** Cada clip dirige un
subconjunto del rig en su propio orden.

**La pose de referencia del `.gis` no es el bind pose del `.mesh`.** En
`jianzao_dunpai` difieren 4 de 10 huesos, todos mitades de pares
izquierda/derecha. El skin sale del `.mesh`; la referencia solo se guarda para
diagnóstico.

## Formato RGIS

Dos envoltorios, mismo registro de clip dentro. `read_gis` acepta ambos.

```
contenedor (NPC y rigs por personaje):
  "RGIS" | u16 versión | u16 desconocido | u16 nº clips | u32 nº huesos ref
  char[32] × huesos            nombres
  float[10] × huesos           pose de referencia: T(3) Q(4) S(3)
  u32 0
  ... registros de clip ...

registro de clip (también suelto, sin contenedor, en la biblioteca compartida):
  char[32] nombre | char[32] vacío | char[32] raíz | u16 nº huesos
  char[32] × huesos
  u16[8] cabecera      ([6] = campo de bits, [7] = nº keyframes)
  float × keys         tiempos en MILISEGUNDOS
  por hueso:
    u8 t_anim, u8 r_anim, u8 s_anim, u8 relleno
    vec3 float32 × (keys si animado, si no 1)
    quat float32 o float16 × (keys si animado, si no 1)
    vec3 float16 × (keys si animado, si no 1)     escala
  u8 0
```

`cabecera[6]` es un **campo de bits**, no un enum:

| Bit | Significado |
| --- | --- |
| `0x0004` | puesto siempre; codificación base |
| `0x0002` | rotaciones en float16 en vez de float32 |
| `0x0100` | un array de tiempos por hueso — **no descifrado** |

`cabecera[0]` vale 30 en unos archivos y `0xFFFF` en otros, así que los fps se
derivan de los tiempos.

## Pendientes, en orden de utilidad

**1. La variante `0x0100`.** Afecta a 1 de 45 clips muestreados (`walk_f` en
`jianzao_guanmu.gis`) y detiene la lectura de los 9 restantes de ese archivo.
Lo que se sabe: tras la tabla de huesos vienen los tiempos compartidos, y luego
cada hueso trae `u16 contador` + sus propios tiempos antes de la cabecera de
canales. Al leer esa cabecera posterior hay un **desajuste de 2 bytes** que no
se resolvió. Esos clips se saltan con su causa en `RGISFile.skipped`.

**2. El validador oficial de Khronos.** Nunca se ejecutó. **No existe como
paquete Python** — se buscó el índice completo de PyPI (46 MB) y solo hay
parsers. Requiere el binario de KhronosGroup/glTF-Validator o Node, ninguno
disponible ni autorizado. Lo cubre parcialmente
`tests/support/gltf_spec_check.py`, declarado en su propio módulo como *no*
siendo el validador oficial.

**3. Otras variantes de `.mesh`.** Solo está verificada con archivo real la
versión 2 / tipo 1 / kind 1. Un `.mesh` de tipo 5 (índices de 16 bits) o de los
tipos cuantizados 20-23 confirmaría que la detección automática acierta ahí.

**4. Datos que se descartan.** El parser salta, por malla: 28 B por hueso
(parecen volúmenes de colisión) y 12 B por vértice (probablemente tangentes,
útiles para normal maps). Ninguno es animación.

**5. Convención de UV.** El exportador glTF conserva `v`; el IQE escribe
`1 - v`. No afecta al rig; hace falta una textura real para decidir.

**6. Submallas.** El parser ignora los bloques adicionales. Si alguna variante
usa paleta de huesos por bloque, el cambio se concentra en cómo se construye
`bone_to_slot` en `gltf_scene.py`.

## Cómo exportar

Interfaz: abrir el NPK, abrir la malla en el visor, **Save As → glTF 2.0 Binary
(GLB) Format with animations (.gis)...**. Busca los `.gis` de la misma carpeta
del NPK abierto y los ofrece con filtro y selección múltiple.

Consola:

```
uv run python tools/diagnose_mesh.py temp/tiejiayong_03.mesh --anim temp/tiejiayong.gis
uv run python tools/diagnose_mesh.py modelo.mesh --anim modelo.gis --clips walk_f,idle
```

El script imprime además qué leyó el parser, qué convención detectó y por qué,
y qué descartó. Es lo que convierte un «se ve raro» en algo diagnosticable.

Verificar un `.glb` en Blender:

```
C:/Users/vicen/AppData/Local/Temp/blv/Scripts/python.exe tests/support/blender_check.py modelo.glb
```

## Mapa de módulos

```
core/anim_loader/rgis.py           lector RGIS, layout en su docstring
core/mesh_converter/skeleton.py    contrato: disposición, papel, conversión, TRS
core/mesh_converter/animation.py   modelo de clips y puente desde RGIS
core/mesh_converter/gltf_scene.py  escena compartida por .gltf y .glb
core/mesh_converter/formats/glb.py contenedor binario
gui/widgets/animation_picker.py    diálogo de selección múltiple
tools/diagnose_mesh.py             diagnóstico y exportación por consola
tests/support/blender_check.py     verificador ejecutable sobre cualquier .glb
tests/support/gltf_reader.py       lector independiente, sin código compartido
tests/support/gltf_spec_check.py   reglas de la especificación (no es Khronos)
tests/support/synthetic.py         fixtures binarios .mesh y .gis
```
