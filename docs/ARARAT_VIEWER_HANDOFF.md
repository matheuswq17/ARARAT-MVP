# ARARAT Viewer - Handoff & Documentação Técnica

**Status:** MVP v1.0.0 (Fevereiro 2026)  
**Suporte de Implementação:** TRAE (Agente de Desenvolvimento)

Esta documentação serve como referência completa para a manutenção, uso e evolução do ARARAT Viewer. O projeto combina visualização DICOM avançada (MPR) com um pipeline de inferência de Machine Learning para classificação de risco de câncer de próstata (ProstateX).

---

## 1. Visão Geral do Produto e Fluxo

O ARARAT Viewer é uma aplicação desktop Python que permite:
1.  **Carregar Séries DICOM:** Navegação rápida por casos (pacientes) e séries (T2, ADC, DWI, Ktrans).
2.  **Visualização MPR:** Multi-Planar Reconstruction (Axial, Coronal, Sagittal) com sincronização de crosshair.
3.  **Anotação de ROIs:** Desenho de Regiões de Interesse (esferas 3D) nas lesões suspeitas.
4.  **Pipeline de Inferência:**
    *   **Export:** A ROI é exportada como máscara NIfTI (`mask_L1.nii.gz`).
    *   **Radiomics:** O Viewer extrai features radiômicas (PyRadiomics) da imagem + máscara.
    *   **Inferência:** Um subprocesso isolado carrega o modelo ML (`scikit-learn 1.8.0`) e gera a predição.
    *   **Resultado:** A probabilidade de malignidade e categoria de risco são exibidas no Viewer.
5.  **Validação (Ground Truth):** Sobreposição de gabarito (lesões reais do dataset ProstateX) para validação visual.

### Fluxo de Dados
`DICOM (Input)` → `Viewer (MPR)` → `ROI (User)` → `Export (NIfTI/JSON)` → `Radiomics (Feature Extraction)` → `CSV (1 linha)` → `Inference CLI (.venv_infer)` → `Predição (JSON)` → `Viewer (Display)`

---

## 2. Ambientes Python e Dependências

O projeto utiliza **dois ambientes virtuais distintos** para garantir compatibilidade entre o código moderno do Viewer e o modelo ML legado/fixo.

### 2.1. Por que dois ambientes?
1.  **`.venv39` (Viewer):** Ambiente principal onde roda a GUI. Usa Python 3.9+ (ou 3.10/3.12) com bibliotecas modernas (`matplotlib`, `SimpleITK`, `PyRadiomics`).
2.  **`.venv_infer` (Inferência):** Ambiente isolado e estrito. O modelo `model.joblib` foi treinado com `scikit-learn==1.8.0`. Versões mais novas do sklearn não conseguem carregar esse arquivo. Portanto, este ambiente é mantido congelado com as dependências exatas do treino.

### 2.2. Como criar os ambientes

#### A. Ambiente de Inferência (`.venv_infer`)
O script PowerShell automatiza a criação e instalação das dependências fixas.
```powershell
# Na raiz do repositório (PowerShell)
.\scripts\setup_inference_env.ps1
```
*Isso cria a pasta `.venv_infer` e instala `scikit-learn==1.8.0`, `joblib`, `pandas`, etc., conforme `requirements/inference.txt`.*

#### B. Ambiente do Viewer (`.venv39`)
Este é o ambiente onde você roda o `viewer_app.py`.
```bash
# Exemplo de criação manual
python -m venv .venv39
.\.venv39\Scripts\Activate
pip install -r requirements.txt  # (Se existir, ou instalar manualmente as libs abaixo)
```
**Principais dependências do Viewer:**
*   `numpy`
*   `matplotlib` (Backend TkAgg ou Qt5Agg)
*   `SimpleITK`
*   `pydicom`
*   `pyradiomics` (Crítico: a extração de features acontece aqui!)
*   `Pillow` (para ícones/assets)
*   `pandas`

---

## 3. Como Rodar

### 3.1. Pré-requisitos
*   Dados do PROSTATEx organizados (ver seção Estrutura de Pastas).
*   Ambientes configurados (`.venv_infer` criado).

### 3.2. Executando o Viewer
Ative o ambiente do viewer e execute o módulo principal:
```powershell
# Ativar venv do viewer
.\.venv39\Scripts\Activate

# Rodar apontando para a pasta de dados
python -m viewer.viewer_app --data_root "C:\Caminho\Para\PROSTATEx"
```
*Se você já rodou antes, ele lembra o último `data_root` salvo em `viewer/config_local.json`.*

### 3.3. Validando Ground Truth (Script Auxiliar)
Para verificar se as coordenadas do gabarito batem com as imagens sem abrir a GUI:
```powershell
python scripts/validate_gt.py --data_root "C:\Dados\PROSTATEx" --case "ProstateX-0000" --dicom_dir "C:\Dados\PROSTATEx\SAMPLES\ProstateX-0000\..."
```

---

## 4. Estrutura de Pastas e Paths

### Estrutura de Dados Esperada (`data_root`)
O Viewer espera uma estrutura similar ao dataset ProstateX:
```text
PROSTATEx/
├── LABELS/                     # CSVs com gabarito (opcional, para modo GT)
│   ├── ProstateX-Findings-Train.csv
│   └── ProstateX-Images-Train.csv
├── SAMPLES/                    # Pastas dos pacientes
│   ├── ProstateX-0000/
│   │   └── ... (séries DICOM)
│   ├── ProstateX-0001/
│   └── ...
└── sample_case_map.json        # (Opcional) Mapeia "case1" -> "ProstateX-0000"
```

### Estrutura do Repositório
*   `inference/models/v1_prostatex/`: Contém `model.joblib`, `meta.json` (lista de features) e `thresholds.json`.
*   `viewer/`: Código fonte da GUI.
*   `exports/`: Saída padrão do sistema.
    *   `roi_manifest.csv`: Log histórico de todas as ROIs criadas.
    *   `roi_images/`: Snapshots PNG das ROIs.
    *   `<PatientID>/<Timestamp>/`: Pasta de exportação por sessão.
        *   `rois.json`: Metadados das ROIs.
        *   `mask_L1.nii.gz`: Máscara binária da lesão.
        *   `features_mask_L1.csv`: Features radiômicas extraídas.
        *   `pred_mask_L1.json`: Resultado da inferência.

---

## 5. Interface (UI) e MPR

### Layout
A tela é dividida em 3 slots principais (Viewports):
*   **MAIN (Grande):** Viewport principal interativo.
*   **BL (Bottom-Left) / BR (Bottom-Right):** Viewports secundários.

### Navegação MPR
*   **Alternar Painel Principal:**
    *   `A` -> Axial no Main.
    *   `K` -> Coronal no Main.
    *   `S` -> Sagittal no Main.
*   **Crosshair Sincronizado:** O centro (voxel atual) é compartilhado. Mover o slice no Axial (Scroll/Setas) atualiza a linha de referência nos outros planos.
*   **Zoom/Pan:**
    *   Scroll: Zoom in/out.
    *   Botão do Meio (arrastar): Pan.
    *   `R`: Reset view (ativo).
    *   `Shift+R`: Reset all views.

---

## 6. ROIs e Inferência (O Coração do Sistema)

### 6.1. Workflow de Criação de ROI
1.  **Navegue** até a lesão suspeita.
2.  **Clique Esquerdo:** "Trava" um candidato a ROI (cruz vermelha tracejada).
3.  **Ajuste o Raio:** Teclas `+` e `-` ajustam o tamanho da esfera (em mm).
4.  **Confirme:** Pressione `Enter`.
    *   A ROI fica **Verde/Lime**.
    *   Um ID é atribuído (ex: L1).
    *   Snapshot PNG e entrada no `roi_manifest.csv` são salvos automaticamente.

### 6.2. Pipeline de Inferência (Tecla `E`)
Ao pressionar `E` (Export), o `viewer_app.py` dispara o processo:
1.  **Salva JSON e NIfTI:** Gera `rois.json` e arquivos `.nii.gz` para cada ROI na pasta `exports/<Patient>/<Time>/`.
2.  **Chama Bridge (`viewer/inference_bridge.py`):**
    *   O bridge roda dentro do Viewer (`.venv39`).
    *   Para cada máscara, ele usa `pyradiomics` para extrair ~100 features da imagem original DICOM.
    *   Filtra as features necessárias (listadas em `inference/models/v1_prostatex/meta.json`).
    *   Salva um CSV temporário (ex: `features_mask_L1.csv`).
3.  **Chama Inferência CLI:**
    *   O bridge executa um comando de terminal (`subprocess`) invocando o Python do `.venv_infer`.
    *   Comando: `python -m inference.infer_cli --features_csv ... --model_dir ...`
    *   O CLI carrega o `model.joblib`, faz a predição e salva `pred_mask_L1.json`.
4.  **Exibe Resultado:**
    *   O Viewer lê o JSON de volta.
    *   Exibe no painel lateral: Probabilidade (%) e Categoria de Risco (Baixo/Int/Alto).
    *   A cor da ROI muda conforme o risco (Verde=Baixo, Amarelo=Int, Vermelho=Alto).

---

## 7. Modo Dev / Ground Truth (GT)

O sistema permite validar a precisão visual comparando com gabaritos oficiais.

### Como funciona
1.  O arquivo `viewer/gt_labels.py` busca arquivos CSV na pasta `LABELS`.
    *   Prioridade: `labels_merged.csv` (Fonte unificada), `ProstateX-Findings-Train.csv`, etc.
2.  Ele tenta cruzar o ID do caso atual (ex: "ProstateX-0000") com as entradas do CSV.
3.  Se encontrar, carrega as coordenadas (x, y, z) das lesões reais.

### GT Labels e ISUP/GGG
*   **Fonte Unificada:** Foi criado o arquivo `data/PROSTATEx/LABELS/labels_merged.csv` para centralizar o Ground Truth.
*   **Campos:** Além de `ProxID`, `fid`, `pos`, `zone`, `ClinSig`, ele suporta `ISUP` e `GGG` (Gleason Grade Group).
*   **Estado Atual:** Como os dados originais de ISUP/GGG não estavam disponíveis no repositório, esses campos estão vazios por padrão. Para adicionar, edite diretamente o `labels_merged.csv` mantendo a estrutura.
*   **Fallback:** Se `labels_merged.csv` for removido, o sistema volta a ler os CSVs originais do ProstateX (sem ISUP/GGG).

### Controles GT
*   `g` (minúsculo): Liga/Desliga a sobreposição visual (X Magenta).
*   `Shift+G`: Pula automaticamente para o slice da lesão GT mais próxima.
*   **Painel de Info:** Mostra detalhes da lesão GT (ClinSig, Zone, Gleason/ISUP) e a distância em mm para a sua ROI desenhada.

### Troubleshooting GT
*   **"GT Indisponível":** O viewer não achou o CSV em `LABELS` ou o nome da pasta do paciente não bate com a coluna `PatientID` do CSV.
*   **Correção:** Verifique se `sample_case_map.json` existe em `SAMPLES` para mapear nomes de pasta (ex: "case1") para IDs oficiais ("ProstateX-0000").

---

## 8. Lista Completa de Atalhos

| Tecla | Função | Contexto |
| :--- | :--- | :--- |
| **Navegação** | | |
| `Scroll` | Zoom In/Out | Painel sob o mouse |
| `Botão Meio` | Pan (Arrastar) | Painel sob o mouse |
| `Botão Dir` | Window/Level (Brilho/Contraste) | Painel sob o mouse |
| `Setas` | Mover slice (Up/Down/Left/Right) | Painel Ativo |
| `[` / `]` | Paginar lista de séries (Sidebar) | Geral |
| `1`..`9` | Carregar série N da lista atual | Geral |
| `Ctrl+G` | Ir para série número N (input) | Geral |
| `Ctrl+Up/Dn`| Próximo/Anterior Paciente | Geral |
| `C` | Ir para Paciente número N (input) | Geral |
| `A` / `K` / `S` | Definir painel principal (Axial/Cor/Sag) | Geral |
| `O` | Abrir nova pasta raiz (Open) | Geral |
| **ROI / Edição** | | |
| `Clique Esq` | Marcar centro (Candidata) | MPR |
| `+` / `=` | Aumentar Raio (+0.5mm) | ROI Candidata |
| `-` / `_` | Diminuir Raio (-0.5mm) | ROI Candidata |
| `Enter` | Confirmar ROI | ROI Candidata |
| `X` | Cancelar seleção | ROI Candidata |
| `Del` | Apagar última ROI criada | Geral |
| **Ações** | | |
| `E` | **EXPORTAR & INFERIR** (Pipeline completo) | Geral |
| `F` | Abrir pasta do último export | Geral |
| `J` / `Ctrl+S`| Salvar apenas JSON (Rápido) | Geral |
| `V` | Validar ROIs (Log no terminal) | Geral |
| **Visualização** | | |
| `R` | Reset view (Zoom/Pan) | Painel Ativo |
| `Shift+R` | Reset all views | Todos |
| `Z` | Toggle Zoom/Crop Mode | Todos |
| `I` | Toggle Interpolação (Bilinear/Nearest) | Todos |
| `g` | Toggle GT Overlay (Gabarito) | Todos |
| `Shift+G` | Ir para lesão GT | Todos |
| `P` | Toggle Painel de Predições | Sidebar |
| `D` | Toggle Debug Mode (Layout info) | Geral |
| `H` | Toggle Help Overlay | Geral |
| `Q` | Sair | Geral |

---

## 9. Troubleshooting Comum

### 1. "GT Indisponível"
*   **Causa:** Arquivo CSV de labels não encontrado em `data_root/LABELS` ou ID do paciente não bate.
*   **Solução:** Crie a pasta `LABELS` na raiz dos dados e coloque o `ProstateX-Findings-Train.csv`. Se usar nomes de pasta customizados, crie o `sample_case_map.json`.

### 2. Erro de Inferência (scikit-learn mismatch)
*   **Sintoma:** Log mostra erro ao carregar `model.joblib` (versão incompatível).
*   **Causa:** O `viewer_app.py` está tentando rodar inferência usando o Python errado ou o `.venv_infer` não foi criado corretamente.
*   **Solução:** Rode `scripts/setup_inference_env.ps1` novamente. Verifique se `viewer/inference_bridge.py` está encontrando o python correto em `.venv_infer/Scripts/python.exe`.

### 3. PowerShell Execution Policy
*   **Sintoma:** Erro ao rodar `.ps1`.
*   **Solução:** Abra o PowerShell como Admin e rode: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

### 4. ROI fora do volume
*   **Sintoma:** Ao validar (`V`), mensagem "ROI fora do volume".
*   **Causa:** A ROI foi desenhada numa série, mas você trocou para outra série com geometria diferente onde aquela coordenada física não existe.
*   **Solução:** Volte para a série original (T2) ou apague a ROI.

---

## 10. Status do MVP e Futuro

### Checklist MVP (Entregue)
- [x] Leitura robusta de DICOM (SimpleITK)
- [x] Visualização MPR (Axial/Cor/Sag) sincronizada
- [x] Criação e exportação de ROIs (NIfTI + JSON)
- [x] Integração com PyRadiomics (extração de features)
- [x] Pipeline de inferência ML (ponte para ambiente legado)
- [x] Visualização de resultados de risco
- [x] Módulo de validação com Ground Truth (GT)

### Pendências / Futuro
- [ ] **Empacotamento .EXE:** Atualmente roda via script. Para distribuir, recomenda-se usar PyInstaller no `.venv39` e incluir a pasta `.venv_infer` inteira como um recurso (asset) distribuído junto, para manter o isolamento.
- [ ] **Suporte a DICOM SR:** Exportar resultados como DICOM Structured Report.
- [ ] **Refinamento UI:** Melhorar performance de renderização (migrar de Matplotlib para VTK/PyQtGraph se a performance for crítica).

---

**Fim do Documento de Handoff.**
