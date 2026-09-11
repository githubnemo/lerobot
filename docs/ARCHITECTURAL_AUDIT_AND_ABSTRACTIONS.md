# Architektur- und Code-Audit: lerobot-video-vam

**Status:** Umfassender Korrektheits- und Architektur-Audit
**Datum:** September 2026
**Ziel:** Vollständige Dokumentation von Fehlern, Data Leakage, VAE-Bypasses, Format-Diskrepanzen und modularer Soll-Architektur für `lerobot-video-vam`.

---

## 1. Executive Summary & Audit-Befund

Das Projekt `lerobot-video-vam` koppelt generative Videodiffusions- und Flow-Matching-Modelle (Cosmos 2B/3/7B/14B, FLUX.2 [klein], LTX-2.5) mit SmolExpert-Aktionsdecodern für Physical AI (Benchmark: `hubnemo/cube_out_of_box_dataset`, 6.536 Frames, 40 Episoden, 10 FPS, Revision `243370c3c08bcbd860133c4a0d658ea7c1d2e77e`).

### Kernergebnisse des Audits:

1. **Gravierendes Frame-Level Data Leakage bei 7B / 14B / FLUX.2:**
   Frühere Skripte (`train_smolexpert_on_cosmos7b.py`, `14b.py`, `flux2_klein.py`) nutzten `torch.utils.data.random_split(dataset, [85%, 15%])` auf sequentiellen Extraktions-Frames (Stride 3). Dadurch wurden zeitlich benachbarte Frames ($t$ und $t+3$, 300 ms Abstand) aus derselben Trajektorie über Train und Val verteilt. Die berichteten Werte (**9.46°**, **9.97°**, **10.33°**) sind **ungültig und methodisch unbrauchbar**.
2. **VAE-Bypass durch Pseudo-Latent-Mocks:**
   In den 7B-, 14B- und FLUX.2-Extraktoren wurden Frames nicht durch echte VAEs kodiert, sondern via `F.interpolate` bilinear auf $32 \times 32$ herunterskaliert und mit Nullen auf 16 Kanäle aufgefüllt. Die Modelle verarbeiteten willkürliche Pixelmittelwerte statt strukturierter VAE-Latents.
3. **Cosmos 3 Edge Mismatch-Problematik:**
   - **Gen-Tower vs. Und-Tower Mismatch:** Video-LoRA wurde auf dem Generierungs-Tower (`gen_seq`) mit Flow-Matching-Loss trainiert, während die Aktions-Features aus dem un-verrauschten Verstehens-Tower (`und_seq`) extrahiert wurden. Dieser Repräsentations-Shift bleibt trotz dualer LoRA bestehen.
   - **Latent-Normalisierung & Posterior Mode:** Fehlende kanalweise Latent-Normalisierung und fehlender deterministischer Posterior Mode (`posterior.mode()`).
   - **FPS-Mismatch:** 10 FPS Roboter-Video gegen 24 FPS Pretraining-Annahme.
   - **Patchify-Reihenfolge:** Frühere manuelle Implementierungen nutzten `(C, p, p)` statt der nativen räumlichen Reihenfolge `(p, p, C)`.
   - **Modellkonfiguration:** Gemessen: 2048 Hidden Dim, 16 Heads, 28 Layer, 3.37B Parameter (Marketing-Label: 4B).
4. **Cosmos-1.0 EDM vs. RF Scheduler-Mismatch:**
   Cosmos-1.0 basiert auf EDM (Karras VP) Diffusion, während Trainingsskripte lineare Rectified-Flow-Gleichungen ohne Rausch-Skalen-Abgleich anwandten.
5. **Abstraktion allein garantiert keine Korrektheit:**
   Die Bereitstellung von `BaseVAMExtractor` und `split_guard.py` ist notwendig, erzwingt jedoch erst dann korrekte Ergebnisse, wenn die abgeleiteten Extraktoren nachweislich echte VAE-Aufrufe, native Patch-Reihenfolgen und disjunkte Splits ausführen.

---

## 2. Detaillierte Fehleranalyse nach Modellfamilien

### 2.1 Cosmos 3 Edge

- **Gen vs. Und Tower:** Der Dual-Pathway MoT trennt visuelles Verstehen (`und_seq`, 600 Tokens) von generativer Diffusion (`gen_seq`). Das Training der Video-LoRA auf Rausch-Vorhersage optimiert die Attention-Gewichte für Rausch-Denoising, nicht für deterministische Merkmalsextraktion auf `und_seq`.
- **Patchify-Bug:** Das Packing von Latents `[B, C, T, H, W]` in Transformer-Tokens erfordert:
  ```python
  x = latents.view(b, c, t, patch_h, p, patch_w, p).permute(0, 2, 3, 5, 4, 6, 1) # -> (b, t, patch_h, patch_w, p, p, c)
  ```
  Alte Skripte vertauschten räumliche und Kanal-Dimensionen, wodurch die räumliche Struktur zerstört wurde.

### 2.2 Cosmos 7B / 14B / FLUX.2

- **Data Leakage:** Unter Protocol 1.0 (disjunkte Episoden 0–31 Train, 32–39 Val) muss der Generalisierungsfehler über den Operator- und Setup-Changepoint hinweg gemessen werden. Random Frame-Splits hebeln diesen Test vollständig aus.
- **Pseudo-Latents:** Der Code `torch.nn.functional.pad(p, (0, 0, 0, 0, 0, 4))` täuschte einen 16-Kanal-Latent-Tensor vor, umging jedoch den Diffusers-VAE vollständig.

---

## 3. Evaluation- und Validierungs-Invariante

1. **Per-Sample Determinismus:** Jedes Evaluierungs-Sample erhält einen deterministischen Seed via stabiler Hash-Funktion `int(hashlib.sha256(f"{evaluation_seed}:{sample_id}".encode()).hexdigest()[:8], 16) % (2**31 - 1)`, unabhängig von Batch Size oder Batch-Reihenfolge.
2. **Optimizer Step Alignment:** Schedulers und Logging synchronisieren strikt auf echte Optimizer-Schritte nach Gradient Accumulation.
3. **Fail-Closed Validierung:** Manifest-Prüfung auf Schema-Version, Dataset-Commit (`243370c3c08bcbd860133c4a0d658ea7c1d2e77e`) und strikt disjunkte Episoden-Sets.
4. **Mixed-Unit Reporting:** Einheitliche Ausweisung der 5 Gelenke in Grad ($^\circ$) und des Greifers in $[0, 100]$ (Prozent) sowie separate Erfassung von H1 (Offset 0), First-5 (Offsets $0..4$) und Full-30 (Offsets $0..29$).

---

## 4. Status der Implementierung & Verifikation

- **Aktueller Stand:** Korrekte, leakage-freie Extraktoren und Trainingsskripte werden aktuell von Agenten implementiert und durchlaufen die Code-Review.
- **Aktive Jobs:** **Aktuell laufen keine Trainings- oder Evaluierungs-Jobs auf `abakus`.** Keine neuen Performance-Zahlen erfunden.
- **Unit Tests:** Tests für Basisabstraktionen und Komponenten sind vorhanden; Hauptverifikation steht noch aus (keine erfundenen Pass-Zahlen).
- **Nächste Schritte:** Nach Abschluss der Code-Verifikation wird ein bereinigter Re-Benchmarking-Queue unter Protocol 1.0 gestartet.
