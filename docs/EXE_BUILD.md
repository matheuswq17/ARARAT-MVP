# Build do ARARAT Viewer (.exe)

## Pré-requisitos

- Windows 10/11
- Python instalado e disponível no PATH
- Dependências instaláveis via `pip`

## Build

```powershell
.\scripts\build_exe.ps1 -Clean
```

Para instalar dependências automaticamente antes do build:

```powershell
.\scripts\build_exe.ps1 -Clean -InstallDeps
```

Artefato esperado:

- `dist/ARARAT_Viewer.exe`

## O que é incluído no bundle

- `viewer/assets/*` para ícone e branding
- `inference/models/v1_prostatex/*` para inferência
- `radiomics_params.yaml` para extração radiômica
- Bibliotecas usadas no runtime do viewer e inferência

## Paths no runtime

- Leitura de recursos: raiz do app (`repo` em dev, `_MEIPASS` no exe)
- Escrita: diretório executável (quando permissível) ou `%APPDATA%\ARARAT`
- Pastas auto-criadas: `exports/`, `logs/`, `data/`
- Log principal: `logs/ararat_viewer.log`

## Checklist de validação

- Abrir `ARARAT_Viewer.exe`
- Carregar dataset pela UI
- Navegar slices e alternar views
- Criar ROI e confirmar
- Exportar JSON/NIfTI
- Rodar inferência
- Gerar PDF
- Validar GT/DEV e atalhos

## Manual operacional

- Consulte `docs/MANUAL_USO_VIEWER.md` para o passo a passo completo de abertura e uso.
- Consulte `docs/MANUAL_DISTRIBUICAO_EXE.md` para envio e operação com apenas o `.exe`.
