# Handoff — exportación GLB con esqueleto y animaciones

Punto de continuación al 28 de septiembre de 2026. El plan completo, con el
razonamiento y las pruebas de cada decisión, está en
[plan-exportacion-glb.md](plan-exportacion-glb.md). Este documento es el resumen
operativo para retomar el trabajo en una sesión nueva.

## Antes de nada: mira cada exportación en un visor real

> Mis 188 pruebas no detectaron que exportaba modelos en espejo, porque todas
> comparaban contra la misma suposición equivocada. Lo viste tú abriendo el
> archivo. Merece la pena mirar cada exportación en un visor real antes de
> darla por buena.

Esto es una regla de trabajo, no una anécdota. Las pruebas comparan la
exportación con lo que el código *cree* que significa el archivo de origen;
cuando esa creencia es falsa, todas coinciden entre sí y pasan. Lo único que no
comparte la suposición es mirar el modelo y compararlo con el juego.

Para que mirar cueste un comando:

```
uv run python tools/diagnose_mesh.py modelo.mesh --anim modelo.gis --validate --render walk_f
```

escribe `modelo.glb`, lo pasa por el validador oficial de Khronos y dibuja
`modelo.png`: vistas desde +Z (el frente glTF), +X, −Z y +Y con los ejes
rotulados, las mismas con el esqueleto por delante, y fotogramas del clip.
Necesita `NEOX_BLENDER_PYTHON` y `NEOX_GLTF_VALIDATOR` (ver abajo). La imagen
no dice si coincide con el juego: eso hay que compararlo a ojo con algo
asimétrico (la mano del arma, un logotipo, el peinado).

**Repasar el trabajo con esta regla en mente ya encontró dos cosas.** La
justificación escrita de que NeoX no necesita espejo era falsa (ver
«Decisiones»), y el visor de la propia aplicación sigue mostrando la imagen
especular del GLB (ver «Pendientes», punto 1).

## Dónde está todo

| | |
| --- | --- |
| Rama | `feat/glb-export` sobre `main` (`438da76`) |
| Remoto | `https://github.com/BOTProT800/NeoXtractor` — la rama está subida, `main` intacto |
| PR | sin abrir: `https://github.com/BOTProT800/NeoXtractor/pull/new/feat/glb-export` |
| Pruebas | 205 con Blender y el validador; 197 + 8 saltadas solo con el validador (así correrá CI); 185 + 20 saltadas sin ninguno |

## Entorno que hace falta reconstruir

Nada de esto está en el repositorio y una sesión nueva no lo tendrá.

**Blender como paquete Python.** El Blender instalado en el equipo de Windows
es 2.79 y **no** tiene importador glTF. Se usa `bpy` desde PyPI, en un entorno
aparte porque pesa ~670 MB y va atado a CPython 3.11:

```
uv venv --python 3.11 C:/Users/vicen/AppData/Local/Temp/blv
uv pip install --python C:/Users/vicen/AppData/Local/Temp/blv/Scripts/python.exe bpy==4.5.14
set NEOX_BLENDER_PYTHON=C:/Users/vicen/AppData/Local/Temp/blv/Scripts/python.exe
```

La ruta debe ser **corta**. El árbol de addons de Blender es profundo y un
prefijo largo empuja `io_scene_gltf2` más allá del límite de 260 caracteres de
Windows, donde Python deja de ver parte del addon y el importador falla con un
`ModuleNotFoundError` engañoso.

En Linux (la sesión en la nube del 28 de septiembre) funciona igual con
`blv/bin/python`, y renderiza sin pantalla. Si Qt no carga, faltan librerías
del sistema: `apt-get install libegl1 libgl1 libxkbcommon0 libfontconfig1
libdbus-1-3` y `QT_QPA_PLATFORM=offscreen`.

**Validador oficial de Khronos.** No está en PyPI; está en npm como
`gltf-validator`, publicado por Khronos desde KhronosGroup/glTF-Validator.
Necesita Node, que en el equipo de Windows no estaba instalado:

```
npm install --prefix C:/tmp/kv gltf-validator@2.0.0-dev.3.10
set NEOX_GLTF_VALIDATOR=C:/tmp/kv/node_modules/gltf-validator
```

La versión está fijada porque `tests/support/khronos.py` lista los mensajes
informativos que emite esa versión. Sin las dos variables, las pruebas de
Blender y del validador se saltan y el resto pasa igual.

**Muestras.** El juego está en `E:\Instaladores\Main\Cyber Hunter`. Los NPK se
leen en el sitio, no hace falta copiarlos. `configs/cyber_hunter.json` ya tiene
las opciones correctas (`decryption_key: null`, `info_size: 28`). La sesión en
la nube **no** las tiene: lo que dependa de un modelo real hay que hacerlo en
el equipo.

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

**Validador oficial de Khronos: pasado en los fixtures sintéticos.** Once
formas de salida distintas (8 y 16 bits, varias raíces, padres posteriores,
hueso 255, sin huesos, matrices horneadas, espejo, clips lineales y
escalonados, clip leído por RGIS en `.glb` y `.gltf`): **0 errores y 0
avisos**. El único mensaje informativo es `UNUSED_OBJECT` en `TEXCOORD_0`,
porque aún no se exportan materiales. Una prueba de control confirma que un
archivo roto a propósito sí devuelve errores. **Falta pasarlo sobre los
modelos reales** (`--validate`).

Verificación de las metas: lector independiente escrito contra la
especificación, skinning en CPU desde los accessors exportados, `pygltflib` y
`trimesh` como lectores de terceros, e importación y posado reales en Blender
4.5.14. En los tres modelos probados, Blender y la evaluación en CPU
**coinciden a 5e-06**.

## Decisiones que costaron y no hay que rehacer

Cada una se resolvió midiendo, salvo la lateralidad, que se resolvió mirando.

**Las matrices de hueso son globales en disposición de vector fila.** La
traslación está en la última fila. Se detecta de los datos comprobando cuál de
las dos ranuras es la afín `(0,0,0,1)`. Ver `core/mesh_converter/skeleton.py`.

**NeoX no necesita espejo para glTF, y eso lo decide la vista, no una
medida.** Hasta el 27 de septiembre el exportador reflejaba X, heredado del
visor. El usuario abrió `tiejiayong_03` en Blender, lo comparó con el juego y
vio una imagen especular; se quitó el espejo. Esa observación es la única
evidencia, y es buena porque la referencia fue el juego.

La justificación que se escribió entonces era **falsa**: «bobinado frente a
normales guardadas, producto escalar medio +0.99, 100 % de triángulos a
favor». Para un espejo `M`, `cross(M a, M b) = det(M) · M · cross(a, b)`, así
que reflejar posiciones y normales e invertir el bobinado da **exactamente la
misma** concordancia. Una fuente zurda con caras en sentido horario puntúa
igual. Tampoco sirve «X simétrico respecto a cero»: es precisamente lo único
que un espejo en X deja igual.
`TestHandedness::test_winding_agreement_cannot_tell_a_mirror` lo deja fijado
para que nadie vuelva a citarla como prueba. `--conversion mirror_x` en
`diagnose_mesh.py` exporta la versión en espejo para compararlas lado a lado.

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

**1. El visor de la aplicación y el exportador IQE siguen reflejando X.**
`gui/renderers/mesh_renderer.py:43` y `core/mesh_converter/formats/iqe.py:120`
niegan X, con una proyección OpenGL corriente. Es decir: **muestran la imagen
especular del GLB**. Si la observación en Blender es correcta, el visor de
NeoXtractor lleva tiempo enseñando los modelos al revés; si no, el GLB está
mal. Se resuelve abriendo el mismo modelo en el visor de la app, el GLB con
`--render`, y comparando ambos con el juego. No se ha tocado porque es un
cambio visible en la aplicación y la decisión es del usuario. Al ver el
espejo en `tiejiayong_03`, el usuario lo comparó **con el propio juego**
(Cyber Hunter), según confirmó el 28 de septiembre: la referencia es la buena,
y eso apunta a que el que está al revés es el visor.

**2. Pasar el validador y el render sobre los modelos reales.** `tiejiayong_03`
con sus 10 clips y `jianzao_dunpai` con sus 22, con `--validate --render`. En
la nube solo hubo fixtures.

**3. Decidir la lateralidad desde los propios datos.** Una alternativa a
depender de la vista: en rigs con nombres izquierda/derecha (los personajes
jugables de `male.npk` probablemente usen el esquema Biped, `Bip01 L Hand`,
`Bip01 L Toe0`), el lado de los huesos «L» frente a la dirección hacia la que
apuntan los dedos de los pies decide si el modelo está en espejo sin mirar el
juego. `tiejiayong` no sirve: sus huesos se llaman `bone_01`…

**4. La variante `0x0100`.** Afecta a 1 de 45 clips muestreados (`walk_f` en
`jianzao_guanmu.gis`) y detiene la lectura de los 9 restantes de ese archivo.
Lo que se sabe: tras la tabla de huesos vienen los tiempos compartidos, y luego
cada hueso trae `u16 contador` + sus propios tiempos antes de la cabecera de
canales. Al leer esa cabecera posterior hay un **desajuste de 2 bytes** que no
se resolvió. Esos clips se saltan con su causa en `RGISFile.skipped`.

**5. Otras variantes de `.mesh`.** Solo está verificada con archivo real la
versión 2 / tipo 1 / kind 1. Un `.mesh` de tipo 5 (índices de 16 bits) o de los
tipos cuantizados 20-23 confirmaría que la detección automática acierta ahí.

**6. Datos que se descartan.** El parser salta, por malla: 28 B por hueso
(parecen volúmenes de colisión) y 12 B por vértice (probablemente tangentes,
útiles para normal maps). Ninguno es animación.

**7. Convención de UV y materiales.** El exportador glTF conserva `v`; el IQE
escribe `1 - v`. Hace falta una textura real para decidir. Exportar materiales
con textura resolvería además el aviso `UNUSED_OBJECT` del validador y daría
otra forma de ver el espejo: un texto en la textura se leería al revés.

**8. Submallas.** El parser ignora los bloques adicionales. Si alguna variante
usa paleta de huesos por bloque, el cambio se concentra en cómo se construye
`bone_to_slot` en `gltf_scene.py`.

**9. CI.** `.github/workflows/tests.yml` instala ahora Node y el validador. CI
solo corre en push a `main`, así que ese paso **no se ha ejecutado nunca**;
revisarlo en el primer push tras fusionar.

## Cómo exportar

Interfaz: abrir el NPK, abrir la malla en el visor, **Save As → glTF 2.0 Binary
(GLB) Format with animations (.gis)...**. Busca los `.gis` de la misma carpeta
del NPK abierto y los ofrece con filtro y selección múltiple.

Consola:

```
uv run python tools/diagnose_mesh.py temp/tiejiayong_03.mesh --anim temp/tiejiayong.gis --validate --render idle
uv run python tools/diagnose_mesh.py modelo.mesh --anim modelo.gis --clips walk_f,idle
uv run python tools/diagnose_mesh.py modelo.mesh --conversion mirror_x --out modelo_espejo.glb --render
```

El script imprime qué leyó el parser, qué convención detectó y por qué, qué
descartó, el informe del validador y dónde dejó la imagen. Es lo que convierte
un «se ve raro» en algo diagnosticable.

Sobre un `.glb` ya exportado:

```
<blender-python> tools/render_glb.py modelo.glb --clip walk_f
<blender-python> tests/support/blender_check.py modelo.glb
```

## Mapa de módulos

```
core/anim_loader/rgis.py           lector RGIS, layout en su docstring
core/mesh_converter/skeleton.py    contrato: disposición, papel, conversión, TRS
core/mesh_converter/animation.py   modelo de clips y puente desde RGIS
core/mesh_converter/gltf_scene.py  escena compartida por .gltf y .glb
core/mesh_converter/formats/glb.py contenedor binario
gui/widgets/animation_picker.py    diálogo de selección múltiple
tools/diagnose_mesh.py             diagnóstico, exportación, validación y render
tools/render_glb.py                hoja de vistas de un .glb, con Blender
tests/support/blender_check.py     verificador ejecutable sobre cualquier .glb
tests/support/khronos.py           validador oficial de Khronos desde Python
tests/support/khronos_validate.js  el mismo, del lado de Node
tests/support/gltf_reader.py       lector independiente, sin código compartido
tests/support/gltf_spec_check.py   reglas de la especificación (respaldo sin Node)
tests/support/synthetic.py         fixtures: .mesh y .gis binarios, rigs, figura
```
