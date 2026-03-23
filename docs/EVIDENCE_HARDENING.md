# Evidência de Hardening do Viewer (MVP)

## 1. Verificação do Flag de Debug
O arquivo `viewer/viewer_app.py` foi auditado e o valor padrão de `dev_layout_debug` foi confirmado como `False`.

**Trecho de Código (viewer/viewer_app.py:139):**
```python
139:         self.dev_layout_debug = False
```

## 2. Fidelidade do Help Text (H)
O método `_get_help_text` foi auditado linha a linha contra os handlers de eventos (`on_key`, `on_scroll`, `on_mouse`).
Todos os atalhos funcionais estão documentados.

**Tabela de Auditoria:**
| Tecla | Função | Status no Help |
| :--- | :--- | :--- |
| `Scroll` | Zoom/Pan | ✅ Presente |
| `Arrows` | Slice | ✅ Presente |
| `A/K/S` | Views | ✅ Presente |
| `1..9` | Atalhos | ✅ Presente |
| `Ctrl+G` | Go to Series | ✅ Presente |
| `Ctrl+Up/Dn` | Nav Paciente | ✅ Presente |
| `R / Shift+R` | Reset | ✅ Presente |
| `Click` | Marcar | ✅ Presente |
| `Enter` | Confirmar | ✅ Presente |
| `X` | Cancelar | ✅ Presente |
| `Del` | Deletar | ✅ Presente |
| `E` | Exportar | ✅ Presente |
| `F` | Abrir Pasta | ✅ Presente |
| `V` | Validar | ✅ Presente |
| `G` | Toggle GT | ✅ Presente |
| `Shift+G` | Jump GT | ✅ Presente |
| `P` | Predições | ✅ Presente |
| `H` | Help | ✅ Presente |
| `D` | Debug | ✅ Presente |
| `Q` | Sair | ✅ Presente |

## 3. Atualização do README
O `README.md` foi atualizado com uma tabela "Cheatsheet" completa, espelhando a fidelidade do Help interno.

## 4. Script de Prova Automatizada
Foi criado o script `scripts/prove_viewer_hardening.py` para validação em CI/CD ou ambiente local.

**Como rodar:**
```powershell
python scripts/prove_viewer_hardening.py
```
*Requer ambiente com matplotlib instalado.*
