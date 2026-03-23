# Manual rápido — distribuição somente com .exe

Este guia é para usuários finais que vão receber **apenas** o executável.

## Posso enviar só o `.exe`?

**Sim.** A versão atual foi empacotada em modo **onefile**, então o usuário não precisa instalar Python nem dependências do projeto.

## O que você deve enviar

- `ARARAT_Viewer.exe`

Opcional:

- uma pasta com casos DICOM de exemplo (para facilitar o primeiro uso)

## Requisitos da máquina destino

- Windows 10 ou 11
- Permissão para executar `.exe` local
- Espaço em disco para dados exportados e logs

## Passo a passo para o usuário final

1. Copie `ARARAT_Viewer.exe` para uma pasta local, por exemplo `C:\ARARAT`.
2. Dê duplo clique no arquivo.
3. Na primeira execução, se aparecer aviso do Windows Defender/SmartScreen:
   - clique em **Mais informações**
   - clique em **Executar assim mesmo**
4. Se o app abrir sem dataset, ele vai pedir para escolher a pasta:
   - selecione a raiz `PROSTATEx`, ou `SAMPLES`, ou um caso único DICOM.
5. Use normalmente:
   - `E` exporta e roda inferência
   - `Ctrl+P` gera PDF
   - `F` abre pasta de exportação

## O que o app cria automaticamente

Na mesma pasta do `.exe` (quando houver permissão):

- `exports\`
- `logs\`
- `data\`
- `config_local.json`

Se não houver permissão de escrita, o app usa fallback em:

- `%APPDATA%\ARARAT`

## Limitações e observações

- O `.exe` funciona sozinho para executar o app.
- O usuário ainda precisa ter acesso aos dados DICOM que deseja abrir.
- Primeira abertura pode demorar alguns segundos.
- Antivírus corporativo pode bloquear execução de binário novo; nesse caso, liberar o arquivo na política local.

## Modo gabarito (GT) no EXE

- O EXE já inclui um pacote de labels GT padrão (`labels_merged.csv`).
- Se o caso estiver com nome oficial `ProstateX-XXXX`, o GT funciona direto.
- Se os nomes dos casos forem customizados (`case1`, `pacienteA`), inclua um mapeamento:
  - `<data_root>\SAMPLES\sample_case_map.json`
  - Exemplo: `{ "case1": "ProstateX-0000" }`
- Também é possível sobrescrever labels criando uma pasta `LABELS`:
  - dentro do dataset selecionado, ou
  - na mesma pasta do `.exe`
- Formatos aceitos: `labels_merged.csv`, `ProstateX-Findings-Train.csv`, `labels.csv`, `labels.json`.

## Checklist de validação para quem recebeu o EXE

1. O app abre.
2. Consegue selecionar dataset.
3. Consegue criar ROI e confirmar com `Enter`.
4. Consegue exportar/inferir com `E`.
5. Consegue gerar PDF com `Ctrl+P`.
6. Existe log em `logs\ararat_viewer.log` (ou `%APPDATA%\ARARAT\logs\ararat_viewer.log`).
