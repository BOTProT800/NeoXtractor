Plan de exportación GLB con esqueleto y preparación para animaciones

Análisis del código local realizado el 17 de septiembre de 2026. **Primera meta implementada el 21 de septiembre de 2026**; el estado de cada etapa y los pendientes están al final del documento. Las secciones de análisis se conservan como registro de los defectos originales.

La primera meta es exportar un `.glb` que conserve la geometría en reposo, la jerarquía de huesos y las influencias por vértice, y que permita posar ambos lados del personaje correctamente. La segunda meta será exportar animaciones sobre ese mismo esqueleto validado.

El problema observado puede originarse en varias capas: lectura binaria, interpretación de las matrices, construcción de la jerarquía y asociación de pesos. Se reprodujeron defectos concretos en el código actual. Falta un modelo real del caso reportado para establecer cuáles de ellos causaron aquella deformación.

**Punto de partida del proyecto.**

El recorrido relevante es `MeshLoader → MeshParser0 → MeshData/Bones → convert_mesh → exportador`. La interfaz exporta `mesh.raw_data`, mientras que el visor aplica sus propias transformaciones a una copia. El registro de formatos permite incorporar GLB sin rehacer los menús.

- [Parser binario](../core/mesh_loader/parsers/new_parser.py): obtiene padres, nombres, matrices, cuatro índices y cuatro pesos por vértice.
- [Tipos de datos](../core/mesh_loader/types.py): `Bones.matrix` no especifica si representa transformaciones locales, globales o inversas de reposo, ni su convención matemática.
- [Exportador actual](../core/mesh_converter/formats/gltf.py): genera `.gltf` JSON con un buffer incrustado en base64. Todavía no genera el contenedor binario GLB.
- [Registro de formatos](../core/mesh_converter/__init__.py) y [guardado desde la interfaz](../gui/widgets/tab_window_ui/mesh_viewer.py): puntos de integración del nuevo formato y de los errores de validación.
- [Visor](../gui/renderers/mesh_renderer.py) y [shader](../shaders/mesh.vert): dibujan geometría estática y líneas de huesos; no evalúan deformación mediante pesos.
- [Detección de recursos](../core/npk/detection.py): reconoce firmas como `RAWANIMA` y `SKELETON`. No se encontró un pipeline de animación esquelética NeoX conectado al exportador 3D. Las animaciones del visor Cocos pertenecen a otro flujo.

**Hallazgos confirmados y su efecto.**

Los siete hallazgos se verificaron contra el código de partida y están corregidos. Cada uno tiene ahora una regresión que falla con el código anterior; la comprobación aparece entre corchetes.

1. **El tipo 5 lee índices de huesos con el ancho incorrecto.** *(Corregido.)* En `new_parser.py` el primer `elif` capturaba los tipos 4 y 5 para seleccionar posiciones de media precisión, y el siguiente `elif`, que seleccionaría índices `uint16` para el tipo 5, nunca se ejecutaba. Se consumían menos bytes de índices y se empezaba a leer pesos antes de su posición real. Las dos decisiones —precisión de posiciones y ancho de índices— son ahora independientes. Reproducción con el parser original sobre el fixture sintético: índices `[0, 1, 0, 0]` y pesos `[0.75, 0.25, 0, 0]` se leían como `[0, 0, 1, 0]` y `[0, 0, 0, 0.75]`, y el flujo terminaba 12 bytes antes de su posición correcta. [`tests/test_parser_binary.py`: `test_joint_indices_and_weights_survive_the_round_trip`, `test_the_reader_consumes_exactly_the_influence_block`]

2. **Las matrices inversas de reposo no están enlazadas al skin.** *(Corregido.)* `inverseBindMatrices` estaba comentado, de modo que glTF asumía identidad. Ahora el skin referencia un accessor `MAT4` generado en el orden de la paleta de joints, con `B[j] = inverse(G[j])` derivado de las matrices globales ya convertidas. [Referencia normativa](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skins). [`tests/test_gltf_scene.py`: `TestInverseBindMatrices`]

3. **La jerarquía pierde enlaces si el padre aparece después del hijo.** *(Corregido.)* El enlace se creaba solo si el nodo padre ya existía. Ahora se crean todos los nodos y después se enlazan padres e hijos. Reproducción con el exportador original sobre el rig asimétrico: de cinco joints, tres quedaban inalcanzables desde la escena (`[3, 4, 6]`); con dos raíces, dos de tres. [`tests/test_gltf_scene.py`: `TestHierarchy`]

4. **La raíz artificial deja los conteos inconsistentes.** *(Corregido.)* El parser ampliaba padres, nombres y matrices al insertar `dummy_root` pero no `bones.count`, y el accessor de matrices usaba el contador anterior. `count` se actualiza al insertar la raíz, `validate_rig` comprueba la coincidencia y el exportador dimensiona los accessors por la longitud real de los arrays. [`tests/test_parser_binary.py`: `test_multiple_roots_keep_the_bone_count_in_sync`]

5. **Se confunden índices de nodos con posiciones dentro de la lista de joints.** *(Corregido.)* Los valores 255 y 65535 se sustituían por `root_node`, que incluía el desplazamiento de los nodos auxiliares. Ahora `JOINTS_0` indexa `skin.joints` mediante los mapas explícitos `bone_to_slot` y `slot_to_node`, y el centinela se deriva del ancho declarado (`255` solo con índices de 8 bits). Reproducción con el exportador original: con 257 huesos y una influencia válida sobre el hueso 255, escribía el índice 3. [Referencia normativa](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#skinned-mesh-attributes). [`tests/test_gltf_scene.py`: `test_bone_255_keeps_its_slot_with_16_bit_indices`]

6. **La validación actual no protege la exportación.** *(Corregido.)* `MeshData.validate` comparaba índices de vértices con la cantidad de caras y rechazaba un triángulo válido de tres vértices. Se separó en `validate_geometry()` y `validate_rig()`, que devuelven listas de problemas legibles y comprueban tamaños de arrays, límites de índices, padres, raíces, ciclos, matrices finitas y no singulares, conteos y pesos. El exportador las llama antes de escribir. [`tests/test_parser_binary.py`: `test_a_valid_triangle_passes_geometry_validation` y siguientes]

7. **Hay rutas que producen influencias vacías sin impedir un skin aparente.** *(Corregido.)* La condición que combinaba `or` y `and` se agrupó explícitamente; el tipo 100 ya no intenta leer influencias del flujo. Un modelo con huesos pero sin ninguna influencia utilizable exporta el esqueleto **sin** skin y lo advierte; si solo parte de los vértices carece de influencia, la exportación falla con los índices afectados en lugar de asignarlos a la raíz. [`tests/test_gltf_scene.py`: `test_bones_without_any_influence_export_without_a_skin`, `test_partially_unskinned_geometry_is_refused`]

**Los demás exportadores usan ya el mismo contrato.**

ASCII, PMX y SMD leían la traslación del hueso desde `matrix[0, 3]` o `matrix.flatten()[:3]`, incompatibles con la disposición de vector fila. Los tres pasan ahora por `core/mesh_converter/skeleton.py`, cada uno en el espacio que su formato pide, y se corrigieron de paso los defectos de jerarquía equivalentes a los del exportador glTF:

- **ASCII**: posición global real del hueso y cuaternión de orientación real, en vez de basura de la primera fila y un `0 0 0 1` fijo.
- **SMD**: el bloque `skeleton` es relativo al padre, con ángulos de Euler XYZ extraídos de la transformación local en vez de ceros fijos. Los nodos se emiten por índice ascendente, así que varias raíces ya no pierden ramas y un esqueleto sin raíz ya no lanza `ValueError`. Se escriben además los enlaces de peso por vértice, que antes se descartaban dejando solo el hueso dominante.
- **PMX**: posición absoluta real del hueso; los huesos se emiten en orden topológico, que es lo que el formato exige y lo que cubre varias raíces y padres almacenados después de sus hijos.

Los tres respetan el centinela derivado de `joint_index_bits` y **no reasignan a la raíz** una influencia inválida: la descartan y lo registran. PMX obliga a que todo vértice referencie un hueso, así que un vértice sin influencia utilizable cae en el primero y se informa cuántos, en lugar de disimularlo.

Conservan su geometría sin convertir, así que el esqueleto se resuelve con `IDENTITY_CONVERSION` para que malla y huesos sigan en el mismo espacio.

**Hipótesis contrastadas sobre la convención de matrices.**

El visor obtenía la posición de cada hueso desde `matrix.T[:3, 3]`, es decir `matrix[3, :3]`, y el exportador IQE calculaba `M[i] @ inverse(M[padre])`. Ambas operaciones solo son coherentes si las matrices son **transformaciones globales de reposo almacenadas en convención de vector fila** (traslación en la última fila). Esa es ahora la interpretación declarada por omisión, y está documentada en `Bones` y en `core/mesh_converter/skeleton.py`.

No se asume que todas las variantes NeoX la compartan. El módulo separa dos ejes independientes y configurables:

- **Disposición** (`MatrixStorage`): vector fila o vector columna. Se **detecta de los propios datos** comprobando cuál de las dos ranuras es la afín `(0, 0, 0, 1)`: la última columna implica vector fila, la última fila implica vector columna. Cuando las matrices no tienen traslación las dos disposiciones son indistinguibles y el módulo lo dice explícitamente en sus diagnósticos en lugar de afirmar una detección.
- **Papel** (`MatrixRole`): global de reposo, inversa de enlace o local. No es deducible de la forma de la matriz, así que se declara; el valor por omisión es global de reposo. Como diagnóstico, si los orígenes de los huesos caen a más de diez veces el radio de la malla de su centro, se advierte que el papel o la disposición declarados pueden ser incorrectos.

El defecto concreto que producía la deformación quedó identificado. El exportador original serializaba `matrix.T` en orden de filas, que glTF vuelve a leer en orden de columnas: el resultado neto es que glTF recibía la matriz almacenada tal cual, con la traslación en la ranura afín. **Todos los huesos colapsaban al origen del modelo.** La pose de reposo seguía siendo exacta —identidad contra identidad— mientras cada pivote era incorrecto, de modo que cualquier rotación de hueso giraba la malla alrededor del origen en vez de la articulación. Es exactamente el síntoma reportado y la razón por la que la identidad en reposo no sirve como criterio de aceptación. Verificado con el exportador original sobre el rig asimétrico: los cinco orígenes salían en `[0, 0, 0]` cuando el rig los sitúa en `(0,0,0)`, `(1,2,0)`, `(-3,2,0)`, `(1,4,0)` y `(-3,5,0)`.

Transponer una matriz para cambiar su interpretación (`to_contract_matrix`) y serializarla en orden de columnas (`_serialize_column_major`) son ahora dos funciones distintas y documentadas como tales.

**Conversión de coordenadas.**

El visor invierte X y cambia el orden de los triángulos; el exportador glTF original exportaba la geometría original. La diferencia se resolvió con una conversión declarada, `CoordinateConversion`, aplicada de forma coherente:

- posiciones mediante `C`,
- normales mediante `inverse(C).T`,
- transformaciones mediante `C @ M @ inverse(C)`,
- orden de triángulos invertido cuando `det(C) < 0`.

El valor por omisión para glTF y GLB es `NEOX_TO_GLTF`, el espejo en X que ya aplicaban el visor y el exportador IQE, y que convierte la base zurda de NeoX en la base diestra que glTF requiere. `IDENTITY_CONVERSION` exporta la base original sin cambios. Las inversas de enlace se derivan de las globales **ya convertidas**, nunca se convierten dos veces.

El parser sigue omitiendo datos de algunos bloques adicionales/submallas y deduce variantes mediante tamaños. Si el modelo afectado usa una paleta de huesos por bloque, la corrección se limita a construir `bone_to_slot` de otra manera: el resto del exportador ya trabaja sobre esos mapas. Esa paleta continúa siendo una posibilidad a investigar, no un hecho establecido.

**Arquitectura implementada.**

```
core/mesh_loader/types.py          Bones.matrix documentado, joint_index_bits,
                                   validate_geometry() / validate_rig()
core/mesh_converter/skeleton.py    contrato del esqueleto: MatrixStorage,
                                   MatrixRole, CoordinateConversion, Skeleton,
                                   build_skeleton, decompose_trs
core/mesh_converter/gltf_scene.py  escena compartida: nodos, skin, accessors,
                                   buffer, validación de influencias
core/mesh_converter/formats/gltf.py  JSON + buffer base64
core/mesh_converter/formats/glb.py   mismo escena, contenedor binario
```

El contrato interno usa vectores columna y matrices 4×4:

```text
G[j] = transformación global del hueso j en reposo
L[j] = inverse(G[parent[j]]) @ G[j]    # raíz: L[j] = G[j]
B[j] = inverse(G[j])
S[j, pose] = G[j, pose] @ B[j]
v[pose] = sum(weight[v, j] * S[j, pose] @ v[reposo])
```

Los nodos de hueso se escriben como traslación, cuaternión y escala locales, porque glTF solo permite animar TRS. La descomposición se verifica recomponiendo la matriz; un hueso cuya transformación local no sea reproducible por TRS (cizallamiento, eje degenerado) se escribe como matriz horneada y se informa por diagnóstico, en lugar de aproximarlo en silencio. La escena es idéntica para `.gltf` y `.glb`: lo único que cambia es cómo viaja el buffer.

**Política sobre pesos.**

Se permite, y se documenta en el código: fusionar índices duplicados dentro de un vértice, renormalizar cuando la suma ya está próxima a 1, y escribir las ranuras sin influencia como índice 0 con peso 0. Tras normalizar en `float64` se corrige el residuo de `float32` sobre la influencia dominante para que la suma almacenada sea exactamente 1.

Se rechaza la exportación, en lugar de disimularla: peso positivo sobre un centinela o un índice fuera de rango, pesos negativos o no finitos, y una suma que se desvíe más de `1e-2` de 1. Ese último umbral existe precisamente para no ocultar un desalineamiento del parser mediante renormalización: los archivos reales llegan normalizados salvo error de redondeo.

**Comprobaciones ejecutadas.**

Suite de regresión en `tests/`, 124 pruebas, ejecutada con `pytest 9.1.1` sobre Python 3.13.5 del entorno `.venv`. Sin Blender instalado quedan 119 pruebas y 5 saltadas, que es como corre en CI:

```
tests/support/synthetic.py        fixtures: blobs .mesh byte a byte y MeshData
tests/support/gltf_reader.py      lector independiente de .gltf y .glb
tests/support/gltf_spec_check.py  comprobador estructural contra la especificación
tests/support/skinning.py         skinning en CPU desde los accessors exportados
tests/test_parser_binary.py       12 pruebas del lector binario y la validación
tests/test_skeleton_contract.py   23 pruebas del contrato de matrices
tests/test_gltf_scene.py          28 pruebas de escena, skin y geometría
tests/test_glb_container.py       12 pruebas del contenedor y del ida y vuelta
tests/test_skinning_deformation.py 13 pruebas de deformación
tests/test_save_integration.py     4 pruebas del guardado desde la interfaz
tests/test_other_exporters.py     24 pruebas de ASCII, PMX y SMD
tests/test_blender_import.py       5 pruebas de importación real en Blender
tests/support/blender_check.py     verificador ejecutable sobre cualquier .glb
```

El lector de `tests/support/gltf_reader.py` no comparte código ni constantes con el escritor: el contenedor, los tipos de chunk, los tamaños de componente y el orden de columnas se derivan otra vez de la especificación. Todas las aserciones de los exportadores leen el **archivo exportado**, no el estado intermedio del constructor.

Comprobado con datos sintéticos:

- Reposo: la pose reconstruida desde el GLB reproduce la geometría de origen con error máximo `0.0` en el rig asimétrico. Tolerancia declarada: `1e-5` veces la diagonal de la malla.
- **Pivotes**: los orígenes de los huesos en el archivo exportado coinciden con la disposición del rig, con una aserción adicional que rechaza el caso degenerado en que todos colapsan al origen. Esta es la prueba que habría detectado la deformación reportada, y la que la identidad en reposo no cubre.
- Movimiento de la raíz: traslación rígida del conjunto, rotación alrededor del pivote esperado, conservación de todas las distancias entre vértices, y recuperación exacta de la geometría al volver a reposo.
- Ramas independientes: rotar un brazo mueve ese brazo y su antebrazo alrededor del pivote correcto y deja inmóviles el otro brazo, su antebrazo y la raíz. Se prueba cada lado por separado sobre un fixture asimétrico —brazo izquierdo en x=1 con antebrazo 2 unidades arriba, derecho en x=-3 con antebrazo 3 arriba— para detectar intercambios. Mover una hoja no altera a su padre.
- Padres fuera de orden y varias raíces: todos los joints alcanzables desde la escena, enlaces coincidentes con la jerarquía de origen.
- Índices de 8 y 16 bits: el mismo rig exportado con ambos anchos produce posiciones deformadas idénticas.
- Hueso 255 válido con índices de 16 bits: conserva su ranura y su nodo; el hueso 256 también.
- Mezcla de pesos: un vértice ligado 60/40 cae en el punto calculado a mano.
- Rotación más traslación en la descomposición local: `L = inverse(G_padre) @ G_hijo` verificado contra aritmética hecha a mano con rotaciones de 90°, donde un orden de multiplicación invertido da el signo opuesto. Escala no uniforme, base espejada y cizallamiento cubiertos por separado.
- Reordenar el almacenamiento de los huesos no mueve ningún hueso.
- Rig grande y enmarañado: 200 huesos con padres en orden aleatorio, cadenas profundas, 600 vértices con una a cuatro influencias e índices de 16 bits. Sin incumplimientos estructurales, todos los joints alcanzables, error en reposo `1.4e-06` frente a una tolerancia de `6.1e-04`. Al rotar el subárbol más grande, ningún vértice sin influencia en ese subárbol se mueve.
- Contenedor GLB: cabecera, versión, longitud total declarada, orden de chunks, relleno a 4 bytes, relleno con espacios en JSON y con ceros en BIN, escritura little-endian comprobada contra la lectura big-endian, y rechazo de una longitud corrompida.
- `.gltf` y `.glb` producen la misma escena y los mismos accessors; el empaquetado no modifica el rig.
- Salida sin huesos: sigue funcionando, sin skin y con un único nodo.
- Cotas de los accessors calculadas sobre los valores `float32` realmente escritos, comprobado con coordenadas grandes.

**Verificado con implementaciones de terceros.**

- `pygltflib 1.16.5`: abre el contenedor, reconoce el skin, los joints y las inversas de enlace, y decodifica las posiciones con los mismos valores.
- `trimesh 5.1.0`: carga la escena, ve la geometría, el grafo de nodos y todos los huesos por nombre, y decodifica las mismas posiciones. Es una segunda implementación sin relación con la anterior.

**Verificado en Blender 4.5.14 LTS.**

Blender se instaló como paquete Python (`bpy`), no como aplicación: el Blender del equipo es 2.79 y no tiene importador glTF. El verificador está en `tests/support/blender_check.py` y se ejecuta sobre cualquier `.glb`; `tests/test_blender_import.py` lo integra en la suite y se salta si no hay intérprete con `bpy`.

El importador glTF de Blender es un consumidor independiente: resuelve el skin, construye el armature y vincula los grupos de vértices sin saber nada de este proyecto. Comprobado sobre seis fixtures —rig asimétrico con índices de 8 y 16 bits, varias raíces, padres almacenados después de sus hijos, 257 huesos con influencia en el 255, y un rig aleatorio de 60 huesos con vértices de hasta cuatro influencias—, todos en verde:

- Geometría en reposo idéntica a la exportada (error máximo `0.000e+00` en el rig asimétrico, `4.2e-06` en el aleatorio).
- **Cada cabeza de hueso cae en el origen de su joint exportado**, y una aserción adicional rechaza el caso en que todos colapsan al origen.
- Jerarquía de padres coincidente, incluidas las raíces múltiples y los padres fuera de orden.
- Los grupos de vértices reproducen las influencias exportadas, nombre a nombre y peso a peso.
- Al aplicar a un hueso una transformación expresada **en espacio de armature**, se mueven exactamente los vértices de su subárbol y ningún otro. La predicción `S = D` para todo el subárbol no depende de ninguna convención de ejes de hueso, así que la comprobación es válida sin importar cómo orientó Blender cada hueso.

El mismo verificador sobre la salida del **exportador original** falla, y confirma la causa raíz desde fuera: `bones are not all collapsed onto the origin -- largest bone offset 0.0000`, además de 7 huesos donde debía haber 5 y una jerarquía incorrecta. Nótese que la comprobación de pivotes *pasa* en ese archivo, porque compara Blender contra los nodos del propio archivo y ambos están colapsados: es exactamente la trampa de la coherencia interna, y por eso existe la aserción de no-colapso.

Blender también descarta los vértices que ninguna cara referencia, así que el verificador empareja vértices por posición en vez de por índice.

**No comprobado con modelos reales.** No se encontraron muestras `.mesh`, `.glb`, `.gltf` ni archivos de juego NPK/WPK/IDX dentro del proyecto. Todo lo anterior demuestra el comportamiento del código sobre datos construidos y verificado por tres implementaciones ajenas; no reproduce el personaje del usuario ni confirma la convención de matrices de ninguna variante concreta.

**Estado de las seis etapas de la primera meta.**

1. **Fijar el caso de referencia y las pruebas de regresión.** *Hecho en su parte sintética; pendiente el caso real.* Existen fixtures para raíz e hijo, dos ramas con pesos distintos, varias raíces, padres fuera de orden, índices de 8 y 16 bits y un hueso válido con índice 255. Todos los fallos reproducidos son pruebas que fallan con el código anterior. Falta el `.mesh` que fallaba, su juego y versión, y el programa donde se probaban los huesos, para registrar hash, variante, conteos y parámetros de extracción.

2. **Corregir lectura y validación antes de exportar.** *Hecho.* Precisión de posiciones y ancho de índices son decisiones separadas; el ancho se conserva en `Bones.joint_index_bits` y define el centinela; la validación está separada en geometría y rig; el límite de índices de caras apunta a vértices.

3. **Definir un contrato explícito para el esqueleto.** *Hecho.* `core/mesh_converter/skeleton.py` expone identidad de hueso de origen, padre, local de reposo, global de reposo e inversa de enlace, conservando la matriz original y el nombre original para diagnóstico y para la futura asociación de clips. La convención se comprobó con pruebas de traslación **más rotación**.

4. **Reparar el skin y generar una escena común para glTF/GLB.** *Hecho.* Nodos primero, enlaces después; mapas explícitos hueso→ranura y ranura→nodo; inversas en el orden de la paleta; TRS locales verificados por recomposición; geometría, UV y normales sin transformar dos veces.

5. **Añadir el contenedor `.glb` e integrarlo en la interfaz.** *Hecho.* `core/mesh_converter/formats/glb.py` empaqueta la misma escena; GLB registrado en `FORMATS`, de modo que aparece en «Save As» y «Save All As». La conversión ocurre **antes** de abrir el archivo, así que un error ya no deja un archivo vacío; el guardado individual informa del fallo y el guardado por lote resume cuántos se guardaron y qué falló en cada caso.

6. **Aceptar el resultado con pruebas de deformación.** *Hecho salvo el validador de Khronos.* Validación estructural, evaluación CPU de skinning desde los accessors exportados, lectura con dos implementaciones de terceros (`pygltflib`, `trimesh`) e importación y posado real en Blender 4.5.14. Todo en CI antes del empaquetado, con las pruebas de Blender saltadas si no hay intérprete con `bpy`. Falta únicamente el validador oficial de Khronos.

**Condiciones para dar por terminada la primera meta.**

| Condición | Estado |
| --- | --- |
| El GLB abre correctamente y contiene geometría, skin y jerarquía completa, con todos los huesos alcanzables desde la escena | Cumplido; confirmado por Blender, pygltflib y trimesh |
| Pasa el validador oficial de Khronos sin errores | **Pendiente**: no existe en PyPI, requiere binario externo |
| Las posiciones en reposo coinciden con la geometría de referencia (tolerancia `1e-5` × diagonal) | Cumplido, error máximo `0.0`; confirmado en Blender |
| Un giro de hueso transforma sus vértices y descendientes según sus pesos; los ajenos no se mueven; cada lado probado por separado | Cumplido; confirmado posando el rig dentro de Blender |
| El giro de la raíz mueve el conjunto coherentemente alrededor del pivote esperado y volver a reposo recupera la geometría | Cumplido; pivotes confirmados en Blender |
| Varias raíces, padres posteriores, índices de 16 bits y hueso 255 válido superan regresiones; los datos incompletos se rechazan con causa identificable | Cumplido |
| La salida sin huesos continúa funcionando | Cumplido |
| Compatibilidad declarada por variante verificada | **Pendiente**: no hay ninguna variante verificada con un archivo real |

**La primera meta no está cerrada.** El comportamiento del exportador está verificado por tres implementaciones ajenas al proyecto, incluida una importación y un posado reales en Blender. Quedan dos condiciones abiertas:

- El **validador oficial de Khronos** no se ejecutó. No existe como paquete Python: se buscó el índice completo de PyPI (46 MB) y solo hay parsers (`gltflib`, `pygltflib`, `pygltfio`, `gltfloupe`, …), ninguno es el validador. Requiere el binario de Khronos o Node, ninguno de los dos disponible ni autorizado.
- **Ninguna variante NeoX está verificada con un archivo real**, así que no puede declararse compatibilidad con ningún juego concreto.

**Pendientes, en orden de utilidad.**

1. **Muestra real.** Ruta a un `.mesh` (preferiblemente el que fallaba), juego y versión. Permitiría confirmar la disposición y el papel de las matrices por variante, registrar hash y conteos, y cerrar la etapa 1.
2. **Validador de Khronos.** Único punto de verificación que sigue sin cubrir. Requiere el binario de [KhronosGroup/glTF-Validator](https://github.com/KhronosGroup/glTF-Validator) o Node; no existe como paquete Python (índice completo de PyPI revisado). Mientras tanto, `tests/support/gltf_spec_check.py` reimplementa el subconjunto de reglas que este flujo puede incumplir —cotas y alineación de accessors, índices de skin y joints, normalización de pesos, normales unitarias, ciclos de nodos— y está declarado en el propio módulo como *no* siendo el validador oficial.
3. **Convención de UV.** El exportador glTF conserva `v` sin invertir, como hacía antes; el exportador IQE escribe `1 - v`. La diferencia no afecta al rig pero conviene resolverla con una textura real.
4. **Submallas y paletas por bloque.** El parser sigue ignorando los bloques adicionales. Si una variante usa paleta de huesos por bloque, el cambio se concentra en cómo se construye `bone_to_slot`.

**Cómo reproducir la verificación en Blender.**

El Blender instalado en el equipo (2.79, de 2017) no sirve: `bpy.ops.import_scene.gltf` no existe. Se usa `bpy` desde PyPI en un entorno aparte, porque pesa cientos de megabytes y va atado a una versión concreta de CPython, así que no es dependencia del proyecto.

```
uv venv --python 3.11 C:/tmp/blv
uv pip install --python C:/tmp/blv/Scripts/python.exe bpy==4.5.14
set NEOX_BLENDER_PYTHON=C:/tmp/blv/Scripts/python.exe
uv run pytest tests/test_blender_import.py
```

Sobre un archivo concreto:

```
C:/tmp/blv/Scripts/python.exe tests/support/blender_check.py modelo.glb
```

La ruta del entorno debe ser **corta**. El árbol de addons de Blender es muy profundo y un prefijo largo empuja `io_scene_gltf2` más allá del límite de 260 caracteres de Windows, donde Python deja de ver parte del addon y el importador falla con un `ModuleNotFoundError` engañoso. Con el scratchpad de esta sesión la ruta llegaba a 264 caracteres y fallaba por eso.

**Criterio matemático de implementación.**

La identidad en reposo es necesaria pero no suficiente: calcular una matriz y su inversa a partir de la misma interpretación equivocada hace pasar esa prueba. El código original es la demostración: su pose de reposo era exacta con todos los pivotes colapsados al origen. Por eso las pruebas comprueban por separado los orígenes de los huesos, los pivotes de rotación y las posiciones esperadas tras mover cada rama. La explicación de las matrices y de la combinación ponderada se apoya en el [tutorial de skinning de Khronos](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_020_Skins.html) y en la [especificación glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html). Un validador estructural no puede decidir si un peso pertenece al brazo correcto.

**Cómo exportar un GLB.**

Desde la interfaz: abrir la malla en el visor y usar «Save As → glTF 2.0 Binary (GLB) Format», o «Save All As» para todas las pestañas abiertas.

Desde código:

```python
from core.mesh_converter import convert_mesh
from core.mesh_converter.formats import glb
from core.mesh_loader import MeshLoader

mesh = MeshLoader().load_from_file("modelo.mesh")
with open("modelo.glb", "wb") as handle:
    handle.write(convert_mesh(mesh, glb))
```

Opciones disponibles, útiles al investigar una variante nueva:

```python
from core.mesh_converter.skeleton import (
    IDENTITY_CONVERSION, MatrixRole, MatrixStorage,
)

payload = convert_mesh(
    mesh, glb,
    conversion=IDENTITY_CONVERSION,          # exporta la base original
    matrix_storage=MatrixStorage.COLUMN_VECTOR,  # o AUTO (por omisión)
    matrix_role=MatrixRole.INVERSE_BIND,     # si la variante guarda inversas
    use_trs_nodes=False,                     # hornea matrices en los nodos
)
```

Los diagnósticos —disposición detectada y por qué, papel declarado, conversión, huesos con cizallamiento, derivas de peso, huesos lejos de la malla— se registran en el log de la aplicación y están disponibles en `build_scene(mesh).diagnostics`.

**Segunda meta: animaciones sobre el rig aceptado.**

La primera meta deja preparado lo que la segunda necesita: identidad de hueso de origen y nombre original conservados en `SkeletonBone`, TRS locales verificados por recomposición en cada nodo de hueso, y un diagnóstico explícito para los huesos cuya transformación no es reproducible por TRS y que por tanto no son animables tal cual.

1. **Resolver la relación entre recursos.** Con muestras del juego elegido, determinar qué archivo contiene el esqueleto y cuáles contienen clips, y cómo se vinculan al modelo. Investigar nombres originales, índices, hashes o paletas reales; no asumir que dos listas de huesos coinciden por orden.

2. **Crear un lector y representación de clips.** Incorporar duración, tiempos en segundos, canales por hueso, traslación, rotación y escala, y modo de interpolación. Establecer si las pistas son transformaciones absolutas, relativas o aditivas, y si están en espacio local o global. Conservar la identidad de hueso definida en la primera meta.

3. **Exportar un primer clip controlado.** Comenzar con una rotación sintética de un hueso sobre el GLB validado para probar el escritor de animaciones independientemente del lector NeoX. Después exportar un clip real sencillo. Generar samplers y canales sobre nodos TRS, comprobando tiempos, normalización y continuidad de cuaterniones. [Modelo de animaciones de Khronos](https://github.khronos.org/glTF-Tutorials/gltfTutorial/gltfTutorial_007_Animations.html).

4. **Validar y ampliar por evidencia.** Comparar posiciones/orientaciones en varios instantes con una referencia confiable, revisar interpolación y bucles, y definir el tratamiento del movimiento de la raíz. Incorporar varios clips y variantes de compresión solamente cuando sus formatos estén identificados.

La segunda meta se cierra cuando al menos un clip real del juego objetivo reproduce el movimiento esperado sobre el mismo rig, sin modificar sus pesos ni su pose de enlace para compensar errores.

El siguiente paso concreto es obtener la muestra afectada y contrastar con ella la convención declarada; el resto de la primera meta está implementado y cubierto por regresiones.
