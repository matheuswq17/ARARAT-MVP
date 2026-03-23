# Manual de Uso — ARARAT Viewer

## 1) Como abrir

### Opção A: Executável (.exe)

1. Vá até a pasta `dist`.
2. Dê duplo clique em `ARARAT_Viewer.exe`.
3. Aguarde a janela principal abrir.

### Opção B: Python (modo desenvolvimento)

```powershell
python viewer/viewer_app.py
```

Também é possível iniciar já apontando dataset:

```powershell
python viewer/viewer_app.py --data_root "C:\CAMINHO\PROSTATEx"
```

## 2) O que acontece ao abrir

1. O app inicializa logs e cria pastas de trabalho (`exports`, `logs`, `data`) se necessário.
2. O app tenta carregar o último dataset salvo em `config_local.json`.
3. Se não houver dataset válido, abre automaticamente o seletor de pasta.
4. Ao selecionar a raiz correta, o app detecta o modo:
   - **Samples mode**: raiz com vários casos (`PROSTATEx` / `SAMPLES`).
   - **Single-case mode**: pasta de um caso específico.
5. A interface carrega os painéis AX/COR/SAG e entra em estado **Pronto**.

## 3) Como escolher a pasta certa

Use a tecla **O** para abrir novamente o seletor de pasta a qualquer momento.

A pasta selecionada pode ser:

- A raiz do dataset (exemplo: `...\PROSTATEx`)
- A pasta `SAMPLES`
- A pasta de um caso único com DICOM válido

Se a pasta não for válida, o app mantém a anterior e mostra aviso na interface.

## 4) Fluxo recomendado de uso (do zero ao laudo)

1. **Navegue no caso/série**
   - Scroll ou setas para percorrer slices
   - `[` e `]` para trocar série
2. **Marque ROIs**
   - Clique esquerdo para travar centro
   - `+` / `-` para ajustar raio
   - `Enter` para confirmar ROI
3. **Exporte e rode inferência**
   - Pressione **E**
   - O app gera JSON + máscaras e roda predição
4. **Revise resultado**
   - Use **P** para painel de predições
   - Use **V** para validações
5. **Gere PDF**
   - Pressione **Ctrl+P** ou botão “Gerar Relatório (PDF)”
6. **Abra pasta de saída**
   - Pressione **F**

## 5) Atalhos principais

- **Navegação**
  - `Arrows`: trocar slice
  - `Scroll / Drag`: zoom/pan no painel ativo
  - `A / K / S`: alternar painel ativo (axial/coronal/sagittal)
  - `Ctrl+Up / Ctrl+Down`: trocar paciente
  - `1..9`: atalho direto de série
  - `Ctrl+G`: ir para série N
  - `C`: ir para paciente N
  - `O`: abrir data_root
- **ROI**
  - Clique esquerdo: travar centro
  - `Enter`: confirmar ROI
  - `X`: limpar seleção
  - `Del`: apagar última ROI
  - `+` / `-`: ajustar raio
- **Geral**
  - `E`: exportar + inferir
  - `J` ou `Ctrl+S`: salvar JSON
  - `Ctrl+P`: gerar PDF
  - `F`: abrir pasta de export
  - `G`: mostrar/ocultar GT
  - `Shift+G`: ir para lesão GT
  - `D`: alternar modo DEV
  - `H`: mostrar/ocultar ajuda
  - `Q`: sair

## 6) Onde ficam os arquivos gerados

Estrutura de saída padrão:

- `exports/<case>/<timestamp>/rois.json`
- `exports/<case>/<timestamp>/mask_*.nii.gz`
- `exports/<case>/<timestamp>/pred_mask_*.json`
- `exports/<case>/<timestamp>/ARARAT_CDS_Report.pdf`

Configuração e logs:

- `config_local.json` (último dataset usado)
- `logs/ararat_viewer.log` (log principal)

Observação: em ambiente sem permissão de escrita na pasta do executável, o app usa fallback em `%APPDATA%\ARARAT`.

## 7) Mensagens comuns na abertura

- **“Dataset não encontrado. Selecione a pasta de cases.”**
  - Não há raiz válida salva; selecione com **O**.
- **“Modelo não encontrado...”**
  - O arquivo `model.joblib` não foi encontrado no bundle.
- **“Erro - ver logs.”**
  - O app capturou exceção crítica; abra `logs/ararat_viewer.log`.

## 8) Checklist rápido de operação

1. Abriu sem erro.
2. Carregou caso/série.
3. ROI criada com `Enter`.
4. Export + inferência com `E`.
5. PDF gerado com `Ctrl+P`.
6. Pasta abriu com `F`.
