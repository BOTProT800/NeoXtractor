# Plan de implementación: adaptar las ideas de REA a NeoXtractor

> Fuente: [morluto/rea](https://github.com/morluto/rea) (licencia MIT), revisado en su rama `main` el 2026-10-08.
> Este plan adapta **ideas y contratos**, no código. REA es TypeScript/Node y apunta a binarios nativos,
> JavaScript/Electron, .NET y web; NeoXtractor es Python/PySide6 y apunta a contenedores NeoX
> (NXPK/EXPK, IDX/WPK) y a sus recursos.

## 1. Resumen

REA ("Reverse Engineer Anything") es un servidor MCP más una CLI que permite a agentes de IA y a personas
inspeccionar artefactos sin su código fuente. Sus proveedores (Hopper, Ghidra, IDA, CDP…) no nos sirven,
pero sí cinco principios de diseño:

1. **Evidencia, no solo resultados.** Cada conclusión indica qué se *observó*, qué se *derivó*, qué se
   *infirió* y qué sigue *desconocido*, con ubicaciones (offsets) y hashes.
2. **Un núcleo, varios adaptadores.** La misma capa de aplicación sirve a la CLI y al MCP con resultados
   equivalentes.
3. **Inventario como grafo de artefactos.** Contenedor → entradas → derivados, con identidad por contenido
   (SHA-256) y estado de integridad (`verified` / `mismatched` / `not-hashed` / `unavailable`).
4. **Errores accionables.** Distinguir entrada malformada, formato no soportado, clave ausente, recurso
   faltante, etc., y proponer un remedio concreto.
5. **Comparar y reproducir.** Comparaciones deterministas entre versiones y cachés ligadas a la identidad
   exacta del objetivo.

En NeoXtractor esto se traduce en:

- un núcleo sin Qt con una canalización de decodificación trazable;
- verificación de CRC;
- extracción con manifiesto;
- una CLI real con `--json`;
- comparación de parches del juego;
- detección asistida de claves XOR;
- recuperación de nombres por hash;
- un servidor MCP con su skill, para que agentes como Claude Code exploren paquetes NPK.

## 2. Diagnóstico del estado actual

| # | Hallazgo | Dónde | Impacto |
|---|---|---|---|
| D1 | No hay CLI: `run_cli()` lanza `NotImplementedError` y `args.py` solo define `gui`. | `main.py:9`, `core/args.py` | No se puede automatizar ni integrar con scripts o agentes. |
| D2 | El núcleo depende de Qt: `core/logger.py` importa PySide6 e instala un handler al importarse; `core/images.py` importa `QImage`. | `core/logger.py:7,45`, `core/images.py:10` | No existe un modo headless limpio (CLI o MCP sin pantalla). |
| D3 | La canalización de decodificación es implícita y está duplicada entre NPK e IDX/WPK. Solo `unwrap_layers` registra una parte de los pasos. | `core/npk/npk_file.py:289-388`, `core/wpk/wpk_file.py:331-475` | No se sabe *por qué* una entrada salió así; depurar juegos nuevos es lento. |
| D4 | `crc` y `zcrc` se leen pero nunca se verifican; solo se usan como material de clave XOR. | `core/npk/npk_file.py:187-188` | La corrupción y las claves incorrectas pasan desapercibidas. |
| D5 | Los errores son flags más logs, con `except Exception` amplios. NPK no guarda el motivo; WPK sí, en `format_metadata["load_error"]`. | `core/npk/npk_file.py:316,323,356`, `core/formats/registry.py:132,158` | El usuario solo ve "ERROR", sin remedio. |
| D6 | La extracción vive en el widget de la GUI, aplana rutas con `os.path.basename` y sobrescribe en silencio cuando hay colisiones. No genera manifiesto. | `gui/widgets/npk_file_list.py:232,271-283` | Se pierde la estructura NXFN y los archivos homónimos se pisan. |
| D7 | Al abrir, se cargan en memoria todas las entradas completas. | `gui/windows/main_window.py:504-517` | Paquetes de varios GB implican mucha RAM y una apertura lenta. |
| D8 | La detección es una cadena de `if` sin id de regla ni confianza; `dat` y `unknown1` mezclan "desconocido" con tipos reales. | `core/npk/detection.py:78,290` | No se distingue lo observado de lo adivinado. |
| D9 | `npkhash_v1.mesh_hash` está implementado pero no se usa en ninguna parte. | `core/npk/npkhash_v1.py` | Se pierde la oportunidad de recuperar nombres a partir de rutas. |
| D10 | No hay tests; la CI solo compila con PyInstaller en `push` a `main`. | `.github/workflows/` | Cualquier refactor de la canalización es arriesgado. |
| D11 | Las claves XOR deben conocerse de antemano y escribirse a mano en un config. | `README.md`, `configs/*.json` | Mucha fricción con juegos nuevos. |

## 3. Correspondencia REA → NeoXtractor

| Idea de REA | Dónde está en REA | Adaptación en NeoXtractor | Fase |
|---|---|---|---|
| Registro de evidencia (`observed` / `derived` / `inferred`, `limitations`, `locations`) | `src/domain/evidence.ts` | `EntryEvidence` por entrada: pasos de decodificación, rangos de bytes, hashes, confianza del nombre y del tipo, limitaciones. | 1 |
| Desconocidos explícitos | `src/domain/residualUnknown.ts` | Campo `unknowns` (p. ej. `encrypt_flag=UNKNOWN(7)` o tipo no reconocido) en vez de un `dat` silencioso. | 1 |
| Grafo de artefactos (nodos, ocurrencias, aristas, `hash_status`) | `src/domain/artifactGraph.ts` | Inventario del paquete: paquete → entrada (offset, tamaños, flags) → derivados (descifrado, descomprimido, BXML→XML, PNG, glTF). | 3 |
| Integridad *fail-closed* / *record-and-continue* | `docs/mcp-contracts.md` (Integrity) | Política de CRC `strict` / `record` (por defecto) / `off`. | 1 |
| Taxonomía de errores con remedio | `src/contracts/errorSchemas.ts`, `docs/user-visible-errors.csv` | `ErrorCode` más `remediation` (p. ej. `key_required` → "ejecuta `probe-key`"). | 1 |
| CLI y MCP sobre la misma capa de aplicación | `src/application/` | `core/services/`, usado por la GUI, la CLI y el MCP. | 2, 5 |
| `--json` y códigos de salida 0/1 | `docs/cli.md` | Igual: 0 = completado (aunque haya entradas parciales), 1 = no se pudo completar. | 2 |
| `doctor` (comprobación del entorno) | `src/application/Doctor.ts` | `neoxtractor doctor`: dependencias, plugins, configs, backend gráfico, permisos de salida. | 2 |
| Snapshots ligados a una identidad exacta | `src/domain/analysisSnapshot.ts` | Caché del inventario por identidad del archivo + digest del config + versión de la canalización. | 3 |
| `compare_artifacts` (added / removed / changed / unknown / contradiction) | `src/domain/artifactComparison.ts` | `compare` entre dos versiones de un paquete (parches del juego). | 3 |
| Formas de herramienta `inspect` / `search` / `trace` / `compare` | `docs/tool-design.md` | `inspect`, `list`/`search`, `refs` (referencias entre recursos), `compare`. | 2–4 |
| Servidor MCP y skill para agentes | `src/server/`, `skill-src/` | `neoxtractor mcp` (stdio) y `skills/neoxtractor/SKILL.md`. | 5 |
| Efectos declarados (solo lectura vs escritura) | `ToolAnnotations` de MCP | `readOnlyHint` en la inspección; `extract` y `convert` declaran escritura y nunca sobrescriben sin `overwrite`. | 5 |
| Carriles de verificación (real vs fixtures) | `docs/testing.md` | Fixtures sintéticos en la CI y un carril opcional con muestras reales locales (`NEOX_SAMPLES_DIR`). | 0 |

### Lo que no adoptamos

- **La pila Node/TypeScript** (Zod, turbo, release-please): seguimos en Python. El núcleo usa `dataclasses`;
  el SDK MCP genera los esquemas a partir de las anotaciones de tipo.
- **Los proveedores externos** (Hopper, Ghidra, IDA, CDP, Playwright, captura de procesos) quedan fuera del
  producto. Sí pueden usarse *junto a* NeoXtractor (ver fase 6).
- **Un `setup` que edite la configuración de los agentes**: basta con documentar el registro manual del MCP.
- **IDs criptográficos para todo y cadenas de revisiones de desconocidos**: usamos SHA-256 del contenido y un
  `manifest_id`, sin libro de revisiones.
- **Un límite de respuesta fijo por diseño**: solo paginamos donde hay una restricción real, como listas de
  más de 100 000 entradas en MCP.

## 4. Arquitectura objetivo

```text
main.py ── despachador: gui | <comando cli> | mcp
│
├── gui/          (PySide6)  consume core.services; añade un panel "Detalles" con la evidencia
├── cli/          (argparse) adaptador fino → core.services; salida de texto o --json
├── mcp_server/   (SDK mcp, FastMCP, stdio) adaptador fino → core.services
│
└── core/         (SIN imports de Qt)
    ├── domain/     evidence.py, errors.py, inventory.py, comparison.py   (puro, sin E/S)
    ├── pipeline/   context.py, stages.py, runner.py                      (pasos trazables)
    ├── npk/, wpk/  lectores existentes; implementan el protocolo ArchiveReader
    ├── formats/    reglas de detección y procesadores (BXML, plugins v1/v2)
    ├── names/      resolución de nombres: NXFN, config, hash (npkhash_v1), diccionarios
    ├── services/   ArchiveSession, inspect, extract, search, refs, compare, probe_key, doctor
    └── images, mesh_loader, mesh_converter, bindict…  (existentes, ya sin Qt)
```

Reglas, adaptadas del `AGENTS.md` de REA:

- Las dependencias apuntan hacia dentro: `domain` no importa `npk`, `wpk` ni `services`.
- La lógica de cada formato vive en `npk/`, `wpk/` o `formats/`, nunca en `domain/` ni en `services/`.
- La CLI y el MCP no contienen lógica: solo traducen argumentos y resultados.
- `core/npk/` y `core/wpk/` se quedan donde están para no romper imports ni plugins.

## 5. Modelo de datos (borrador)

```python
# core/domain/evidence.py
class Confidence(StrEnum):
    OBSERVED = "observed"  # leído tal cual del archivo (cabecera, índice, NXFN)
    DERIVED = "derived"    # cálculo determinista (descompresión, CRC verificado)
    INFERRED = "inferred"  # heurística (tipo por contenido, nombre por hash o diccionario)


class HashStatus(StrEnum):
    VERIFIED = "verified"
    MISMATCHED = "mismatched"
    NOT_HASHED = "not-hashed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ByteRange:
    source: str  # "archive" | "wpk:<pkg_id>" | "slot_file:<ruta>"
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class DecodeStep:
    operation: str  # "read", "expk_decrypt", "xor_basic", "zlib", "lz4", "zstd", "rotor",
                    # "nxs3", "strip_none", "lz4_like", "wpk_stage1", "dtsz", "cobl", "processor:bxml"…
    parameters: dict[str, Any]  # {"key": 150}, {"declared_size": 1234}
    input_size: int
    output_size: int
    output_sha256: str | None  # None en modo de evidencia "summary"
    status: Literal["applied", "skipped", "failed"]
    error: "NeoxError | None" = None


@dataclass(frozen=True, slots=True)
class Finding:
    value: str
    confidence: Confidence
    rule: str  # "nxfn", "config_map", "hash:npkhash_v1", "magic:DDS", "text:<Material", "fallback"
    location: ByteRange | None = None


@dataclass(slots=True)
class EntryEvidence:
    archive_id: str  # identidad del paquete (ver §8)
    index: int
    signature: str   # hexadecimal
    source: ByteRange
    declared: dict[str, int]  # compressed_size, original_size, zip_flag, encrypt_flag, crc, zcrc
    integrity: dict[str, HashStatus]  # {"crc": ..., "zcrc": ...}
    steps: list[DecodeStep]
    name: Finding
    file_type: Finding
    category: str
    output_size: int
    output_sha256: str | None
    limitations: list[str]
    unknowns: list[str]
    errors: list["NeoxError"]
```

```python
# core/domain/errors.py
class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"              # ruta o argumento mal formado
    UNSUPPORTED_FORMAT = "unsupported_format"    # magic desconocido, zip_flag/encrypt_flag UNKNOWN(n)
    KEY_REQUIRED = "key_required"                # BASIC_XOR sin clave en el config
    WRONG_KEY_SUSPECTED = "wrong_key_suspected"  # la descompresión falla tras aplicar la clave dada
    DECOMPRESSION_FAILED = "decompression_failed"
    INTEGRITY_MISMATCH = "integrity_mismatch"    # el CRC no coincide
    MISSING_RESOURCE = "missing_resource"        # WPK o slot_file ausente
    TRUNCATED = "truncated"
    PROCESSOR_FAILED = "processor_failed"        # un plugin o BXML lanzó una excepción
    OUTPUT_EXISTS = "output_exists"
    PERMISSION_DENIED = "permission_denied"


@dataclass(frozen=True, slots=True)
class NeoxError:
    code: ErrorCode
    message: str             # qué restricción falló y en qué entrada
    remediation: str | None  # acción concreta y disponible
    details: dict[str, Any]
```

**Compatibilidad.** `NPKEntry` gana un atributo `evidence: EntryEvidence | None`. Los flags existentes
(`ERROR`, `ENCRYPTED`, `TEXT`…) se derivan de la evidencia, así que la GUI y los plugins siguen funcionando
sin cambios.

## 6. Fases

Tamaños orientativos: S (días), M (1–2 semanas), L (más de 2 semanas), para una persona.

### Fase 0 — Cimientos (S–M)

Objetivo: refactorizar sin miedo y ejecutar el núcleo sin Qt.

1. **Desacoplar Qt del núcleo**
   - `core/logger.py`: quitar el import de PySide6 y mover `custom_logging_handler` y
     `qInstallMessageHandler` a `gui/__init__.py:run()`.
   - `core/images.py`: devolver `PIL.Image` y mover `image_to_qimage` y `bgra_to_rgba` a
     `gui/utils/images.py`.
   - Mover `gui/config_manager.py` a `core/config_manager.py` (no usa Qt) y dejar un re-export.
   - Test: importar `core.*` no carga PySide6 (`"PySide6" not in sys.modules`).
2. **Infraestructura de tests**
   - Añadir `pytest` a un grupo de dependencias de desarrollo (`[dependency-groups] dev`) y crear `tests/`.
   - Crear `tests/fixtures/npk_builder.py`, un escritor *solo para tests* que genera NXPK y EXPK sintéticos con:
     entradas zlib, lz4 y zstd; XOR BASIC, ADVANCED e INCREMENTAL (el XOR es simétrico, así que el
     decodificador sirve también de codificador); NXFN (`hash_mode=2`); `var1=1`; envoltorios `NONE`.
   - Escribir tests de caracterización del comportamiento **actual** (extensión, categoría, bytes de salida,
     flags) antes de tocar la canalización.
   - Carril opcional `NEOX_SAMPLES_DIR`: si la variable existe, se recorren paquetes reales locales y se guarda
     un resumen dorado en `tests/golden/` (ignorado por git) para comparar antes y después del refactor.
     **Nunca se suben assets de juegos.**
3. **CI de pull requests**: nuevo `.github/workflows/test.yml` que se ejecuta en `pull_request` y `push` y
   corre `uv sync --locked`, `pylint` (ya existe `pylintrc`; al principio, sin bloquear) y `pytest`.

**Criterios de aceptación**

- `uv run pytest` pasa en la CI.
- Los tests y `python main.py --version` funcionan sin pantalla.

### Fase 1 — Canalización trazable y evidencia (M–L)

Objetivo: que cada entrada sepa cómo se decodificó, con qué confianza y por qué falló.

1. Crear `core/domain/evidence.py` y `core/domain/errors.py` (ver §5).
2. **Canalización por etapas en `core/pipeline/`.** Convertir `NPKFile._load_entry_data` e
   `IDXWPKFile._load_entry_data` en etapas componibles:

   ```python
   class Stage(Protocol):
       name: str

       def applies(self, ctx: DecodeContext) -> bool: ...

       def run(self, ctx: DecodeContext) -> DecodeStep: ...
   ```

   - **Etapas NPK:** `read` → `expk_decrypt` → `entry_decrypt` (`xor_*`) → `arc4_lz4` (`hash_mode=3`) →
     `decompress` → bucle de desenvoltura (`strip_none`, `lz4_like`, `rotor`, `nxs3`) → procesadores →
     detección.
   - **Etapas WPK:** `read` (WPK o slot) → separación cabecera/payload → `stage1` → desenvoltura
     (`dtsz`, `enon`, `cobl`) → procesadores → detección.
   - El bucle de desenvoltura conserva el límite `max_layers=32` y la detección de ciclos, y **reporta** una
     limitación si se alcanza el límite.
   - Dos modos de evidencia: `full` (SHA-256 en cada paso) y `summary` (tamaños más el hash final). Antes de
     fijar el valor por defecto, medir el coste en un paquete grande.
3. **Integridad CRC**
   - Spike con muestras reales: determinar qué bytes cubren `crc` y `zcrc` (¿descomprimidos?
     ¿comprimidos o cifrados?) y con qué algoritmo (`zlib.crc32` o una variante). Documentarlo en
     `docs/formats/npk.md`.
   - Si se confirma, `integrity` pasa a `verified` o `mismatched`. Si no, se marca `unavailable` con una
     limitación explícita. Nunca se marca `mismatched` sin un oráculo verificado.
   - Política `--integrity strict|record|off`, con `record` por defecto:
     - `strict` marca el error y no procesa capas anidadas de esa entrada.
     - `record` continúa y lo anota (equivale al *record-and-continue* de REA).
   - Un `zcrc` que no cuadra tras aplicar el XOR es una señal fuerte de clave incorrecta y se reporta como
     `WRONG_KEY_SUSPECTED`.
4. **Detección por reglas.** Crear `core/formats/detection_rules.py` con una tabla
   `DetectionRule(id, kind, predicate, extension, confidence)` construida a partir de los `if` actuales, en el
   mismo orden para obtener los mismos resultados. `get_ext()` se mantiene como envoltorio. Si ninguna regla
   aplica, se devuelve `Finding("dat", INFERRED, "fallback")` más el desconocido "tipo no reconocido".
5. **Errores.** Reemplazar los `except Exception` por capturas concretas que produzcan un `NeoxError` con
   remedio. Por ejemplo:
   - `KEY_REQUIRED`: "Selecciona un config con `decryption_key` o ejecuta `neoxtractor probe-key <npk>`".
   - `MISSING_RESOURCE` (WPK): indicar el `pkg_id` y las rutas candidatas probadas (ya las genera
     `WPKPathResolver`).
   - `UNSUPPORTED_FORMAT` con `UNKNOWN(n)`: pedir que se abra un issue adjuntando la salida de
     `neoxtractor inspect --json`.
6. **API de plugins v2, compatible con v1.**
   - `FormatDecodeResult` gana `confidence` y `limitations`.
   - El registro envuelve cada `decode` en un `DecodeStep("processor:<nombre>")`.
   - Los fallos quedan como `PROCESSOR_FAILED` en la evidencia, no solo en el log.
7. **GUI.**
   - Panel "Detalles" en `PreviewWidget` con la evidencia: pasos, hashes, CRC, nombre y tipo con la regla que
     los produjo, y errores con su remedio.
   - Icono de aviso en la lista para las entradas con estado `mismatched` o con errores.

**Criterios de aceptación**

- Los tests de caracterización de la fase 0 siguen pasando (mismas salidas).
- Toda entrada cargada tiene `evidence`.
- Todo fallo tiene un `ErrorCode`.

### Fase 2 — Capa de aplicación, CLI y extracción correcta (M)

1. **`core/services/session.py` — `ArchiveSession`**
   - Abre NPK, EXPK, IDX o WPK según el magic (no solo por la extensión) y lee el índice de inmediato.
   - Carga perezosa con una caché LRU de datos decodificados, limitada en bytes y configurable, en lugar de
     retener todo en memoria (D7).
   - La GUI mantiene la pasada de clasificación en segundo plano, pero libera `data` tras extraer los
     metadatos.
2. **`core/services/extract.py`**
   - Conserva las rutas NXFN, saneadas: sin `..`, sin rutas absolutas y sin caracteres inválidos en Windows.
   - Colisiones: política `skip|rename|overwrite`, con `rename` por defecto. Nunca se sobrescribe en silencio.
   - Escribe un `manifest.json` en la carpeta de destino con, por archivo: `index`, `signature`, ruta,
     `sha256`, tamaño, si es decodificado o crudo y un resumen de la evidencia. Incluye también un
     `manifest_id`.
   - La GUI (`npk_file_list.extract_entries`) pasa a usar este servicio, lo que corrige D6.
3. **`cli/` y `main.py`**: subcomandos con salida legible y opción `--json`.

   ```text
   neoxtractor info     <archivo> [--config NOMBRE|RUTA]
   neoxtractor list     <archivo> [--category …] [--ext …] [--name GLOB] [--json]
   neoxtractor inspect  <archivo> <entrada>            # índice | 0xfirma | nombre
   neoxtractor extract  <archivo> -o DIR [--filter …] [--raw] [--on-conflict rename|skip|overwrite]
   neoxtractor search   <archivo> <texto> [--in names|content]
   neoxtractor convert  <archivo> <entrada> --to png|gltf|obj|fbx|smd|iqe|pmx -o RUTA
   neoxtractor doctor   [--json]
   neoxtractor gui      # comando por defecto si no se indica ninguno
   ```

   - Códigos de salida, como en REA:
     - `0`: la operación se completó, aunque haya entradas parciales, avisos o desconocidos.
     - `1`: la operación no se pudo completar (entrada inválida, permisos, cancelación).
   - `--config` reutiliza `ConfigManager` y acepta tanto un nombre de los configs incluidos como la ruta a un
     JSON.
4. **`doctor`**
   - Comprueba las versiones de las dependencias (lz4, zstandard, texture2ddecoder, Pillow, PySide6), los
     plugins cargados y sus errores, la validez de `configs/*.json`, el backend gráfico disponible y el
     permiso de escritura en el directorio de salida.
   - Separa requisitos obligatorios y opcionales: como en REA, que falte algo opcional no hace fallar tareas
     que no lo necesitan.

**Criterios de aceptación**

- Extraer el mismo fixture por CLI y por GUI genera manifiestos idénticos.
- `neoxtractor list --json` funciona en Linux, Windows y macOS sin abrir ninguna ventana.

### Fase 3 — Inventario, caché y comparación de versiones (M)

1. **`core/domain/inventory.py`**: grafo del paquete, adaptado del `artifactGraph` de REA.
   - **Nodos:** el paquete, cada entrada (bytes crudos) y cada derivado (descifrado, descomprimido,
     decodificado, exportado), con `sha256`, tamaño y tipo.
   - **Ocurrencias:** dónde aparece cada entrada (`ByteRange`, tamaños declarados, `encrypted`,
     `hash_status`).
   - **Aristas:** `contains`, `derived-from` y `converts-to` (`references` llega en la fase 4).
   - `manifest_id = sha256(JSON canónico)`: el mismo paquete con el mismo config y la misma versión de la
     canalización produce siempre el mismo id.
2. **Caché del inventario** (equivale a los snapshots de REA)
   - Se guarda en `~/.cache/neoxtractor/<clave>.json`, donde la clave combina la identidad del paquete, el
     digest del config y `PIPELINE_VERSION`.
   - Reabrir un paquete grande ya analizado es inmediato.
   - Se invalida automáticamente si cambia cualquier componente de la clave; `--no-cache` la desactiva.
3. **`compare`** (adaptado de `compare_artifacts`)

   ```text
   neoxtractor compare <viejo.npk> <nuevo.npk> [--config …] [--json] [--extract-changed DIR]
   ```

   - Empareja por `file_signature` (estable porque es un hash de la ruta) y, como respaldo, por nombre.
   - Clasifica cada entrada como:
     - `unchanged`, `added`, `removed` o `changed`;
     - `unknown`, si un lado no se pudo decodificar;
     - `contradiction`, si el CRC no cuadra.
   - Indica qué dimensiones cambiaron: `content`, `size`, `type`, `flags` o `name`.
   - Caso de uso principal: ver qué cambió en un parche del juego y extraer solo lo nuevo.
   - GUI: "Archivo → Comparar con…" con una lista filtrable por estado.

### Fase 4 — Inferencia asistida con evidencia (M–L)

1. **`probe-key`** (resuelve D11)
   - Para las entradas `BASIC_XOR`, probar las claves 0–255 sobre una muestra de entradas comprimidas.
   - Una clave es candidata si, tras el XOR, la descompresión produce exactamente `file_original_length`
     bytes y, si se confirmó en la fase 1, el CRC coincide.
   - El resultado lista las candidatas ordenadas, con su evidencia: cuántas entradas se validaron y qué
     oráculo se usó (tamaño, CRC o magic). Su confianza es `derived`. Si ninguna clave sirve, se devuelve
     `unknown` junto con lo que se probó.
   - GUI: botón "Detectar clave" en el gestor de configs y, al abrir un paquete con entradas `KEY_REQUIRED`,
     una propuesta del tipo "¿Crear un config con la clave 150?".
2. **Recuperación de nombres por hash** (resuelve D9)
   - Spike: confirmar que `npkhash_v1.mesh_hash(ruta)` reproduce la `file_signature` de 32 bits en paquetes
     con NXFN conocido. Es un oráculo perfecto, porque ahí tenemos a la vez el nombre y la firma.
   - Fuentes de rutas candidatas:
     - el NXFN de otros paquetes del mismo juego;
     - las cadenas con forma de ruta dentro de entradas decodificadas (XML, BXML, mtg, gim, json);
     - listas del usuario (`data/namelists/*.txt`).
   - El nombre resultante lleva `Finding(confidence=INFERRED, rule="hash:npkhash_v1")` y nunca se mezcla con
     los nombres observados. Se puede guardar en el `entry_signature_name_map` del config.
3. **`refs`: grafo de referencias** (la forma `trace` de REA)
   - Extraer referencias con forma de ruta de las entradas de texto o BXML (`.gim` → `.mesh`,
     `.mtg` → texturas, `.sfx` → partículas…), hashearlas y resolverlas a entradas.
   - Cada referencia es una arista `references` con evidencia: entrada de origen, offset de la cadena y regla.
     Las referencias no resueltas se listan como tales.
   - CLI: `neoxtractor refs <npk> <entrada> [--depth N]`.
   - GUI: "Extraer con dependencias" (modelo, texturas y materiales).

### Fase 5 — Servidor MCP y skill para agentes (M)

1. **Dependencia.** El SDK oficial `mcp` de Python (FastMCP) se declara como extra opcional
   (`[project.optional-dependencies] mcp`) y se incluye en el build de PyInstaller.
2. **Servidor.** `mcp_server/server.py` usa transporte stdio y se arranca con `neoxtractor mcp`. Como en REA,
   cada herramienta recibe la ruta explícita del paquete. Las sesiones (`ArchiveSession`) se reutilizan desde
   una caché del proceso.

   | Herramienta | Forma | Efectos | Devuelve |
   |---|---|---|---|
   | `archive_overview` | inspect | solo lectura | Cabecera, conteos por categoría y tipo, config usado, problemas. |
   | `list_entries` | list | solo lectura | Una página de entradas (filtros y `cursor`); la paginación se justifica por paquetes de más de 100 000 entradas. |
   | `inspect_entry` | inspect | solo lectura | `EntryEvidence` completa. |
   | `read_entry_text` | inspect | solo lectura | Texto o BXML decodificado, con `max_bytes` y aviso explícito de truncado. |
   | `search_entries` | search | solo lectura | Coincidencias por nombre o contenido, con offset. |
   | `trace_references` | trace | solo lectura | Aristas `references` y referencias no resueltas. |
   | `compare_archives` | compare | solo lectura | Diferencias clasificadas. |
   | `probe_decryption_key` | workflow | solo lectura | Claves candidatas con evidencia. |
   | `extract_entries` | — | escribe en el directorio elegido | Manifiesto; nunca sobrescribe sin `overwrite: true`. |
   | `convert_entry` | — | escribe | Ruta y hash del PNG, glTF, OBJ… generado. |
   | `doctor` | — | solo lectura | Estado del entorno. |

   - Las `ToolAnnotations` (`readOnlyHint`, `destructiveHint`) describen los efectos reales.
   - Los resultados incluyen `result` con la evidencia en línea; los errores incluyen `code` y `remediation`.
   - Si un resultado no cabe en la respuesta, se devuelve un error `resource_constraint` con una sugerencia
     (paginar o usar `extract_entries`). Nunca se trunca en silencio.
3. **Skill.** `skills/neoxtractor/SKILL.md`, al estilo del `skill-src/` de REA, indica:
   - cuándo usar NeoXtractor y cómo elegir la ruta según el tipo (`.npk`/`.expk` frente a `.idx`/`.wpk`);
   - que conviene empezar por `archive_overview`;
   - que no hay que adivinar claves, sino usar `probe_decryption_key`;
   - que al responder hay que separar lo observado de lo inferido;
   - que nunca se deben sobrescribir salidas.
4. **Documentación.** `docs/mcp.md` explica el registro manual en cada cliente, por ejemplo
   `claude mcp add neoxtractor -- /ruta/neoxtractor mcp`.
5. **Tests.** Un cliente MCP en el mismo proceso se ejecuta contra los fixtures y valida los JSON Schema
   publicados.

### Fase 6 — Flujo de investigación con REA (documentación, opcional) (S)

REA no se integra en el código, pero es la herramienta natural para lo que NeoXtractor no puede hacer solo:
descubrir **nuevos esquemas** de cifrado o compresión en los clientes del juego, como las tablas de
`eggyparty_codes.py` o las variantes nuevas de NXS.

- **Plantilla.** `docs/research/TEMPLATE.md` recoge:
  - la identidad del binario (juego, versión, sha256);
  - las funciones y direcciones relevantes;
  - la evidencia (pseudocódigo resumido), las inferencias y los desconocidos;
  - el fixture sintético que lo prueba.
- **Flujo.**
  1. Analizar el ejecutable o la `.so` del cliente con REA y Ghidra para localizar la rutina.
  2. Implementarla como una `Stage` o un plugin.
  3. Añadir un fixture y un test.
  4. Documentar el hallazgo.
- **Límites.** Respetar el aviso legal del README: uso educativo, sin redistribuir assets ni binarios.

## 7. Orden recomendado y primeros PRs

1. **PR 1 (fase 0):** desacoplar Qt de `core/logger.py` y `core/images.py`; añadir `pytest`, `npk_builder` y
   la CI de tests.
2. **PR 2 (fase 0):** tests de caracterización y carril `NEOX_SAMPLES_DIR`.
3. **PR 3 (fase 1):** `domain/evidence.py`, `domain/errors.py` y reglas de detección en tabla, sin cambiar
   la salida.
4. **PR 4 y PR 5 (fase 1):** canalización por etapas con evidencia; primero NPK y luego IDX/WPK.
5. **PR 6 (fase 1):** spike y verificación de CRC; panel "Detalles" en la GUI.
6. **PR 7 y PR 8 (fase 2):** `ArchiveSession` perezosa; servicio de extracción con manifiesto (también en la
   GUI); CLI y `doctor`.
7. **Después:** fases 3, 4 y 5, en ese orden. `probe-key` (fase 4) puede adelantarse justo después de la
   fase 2 si es lo que más piden los usuarios.

Dependencias: 0 → 1 → 2 → {3, 4} → 5. La fase 6 es independiente.

## 8. Riesgos y preguntas abiertas

| Riesgo o pregunta | Mitigación |
|---|---|
| El refactor de la canalización cambia salidas en juegos que los fixtures no cubren. | Tests de caracterización y carril con muestras reales locales; comparar los resúmenes dorados antes de fusionar. |
| No se sabe qué cubren `crc`/`zcrc` ni con qué algoritmo, y puede variar según el juego. | Spike en la fase 1. Si no se confirma, `unavailable` con una limitación. Nunca `mismatched` sin un oráculo verificado. |
| Calcular SHA-256 en cada paso es caro en paquetes de varios GB. | Modo `summary` por defecto si la medición lo justifica; `full` bajo demanda (`inspect`). |
| Hashear un paquete enorme completo solo para la clave de la caché es lento. | Para la clave, usar una identidad rápida (tamaño + mtime + hash de la cabecera y el índice), declarada como tal; el SHA-256 completo queda opcional. |
| `mesh_hash` puede no ser el hash de firma en todos los juegos. | Spike con NXFN como oráculo; declarar en cada config qué algoritmo aplica (`name_hash: "npkhash_v1"`). |
| El SDK MCP (pydantic, anyio…) aumenta el tamaño del ejecutable. | Extra opcional; medir el tamaño del build y, si pesa demasiado, publicar un build aparte `neoxtractor-mcp`. |
| `probe-key` puede dar falsos positivos con muestras pequeñas. | Exigir al menos N entradas validadas con el oráculo de tamaño exacto (y CRC si existe), e informar de N y del oráculo usado. |
| Hay plugins externos escritos para la API v1. | La API v2 es compatible: el registro adapta los resultados v1 añadiéndoles una evidencia mínima. |

Preguntas para el equipo:

1. ¿Va antes `probe-key` (fase 4) que la comparación de versiones (fase 3)?
2. ¿El ejecutable de las releases debe incluir el servidor MCP o se publica aparte?
3. ¿Hay paquetes NPK de referencia por juego que el equipo pueda usar localmente para el carril real?

## 9. Definición de "hecho"

- `core/` no importa Qt. La GUI, la CLI y el MCP llaman a los mismos servicios y obtienen los mismos
  resultados.
- Toda entrada tiene evidencia (pasos, rangos, hashes, confianza del nombre y del tipo) y todo fallo tiene un
  `ErrorCode` con su remedio.
- La extracción conserva las rutas, nunca sobrescribe en silencio y deja un manifiesto verificable.
- Se pueden comparar dos versiones de un paquete y detectar la clave XOR sin conocerla de antemano.
- Un agente con el MCP registrado puede responder, citando evidencia, preguntas como "¿qué texturas usa el
  modelo X y qué cambió en el último parche?".
