# ARARAT MVP – Visualizador e Inferência ProstateX

Este é o repositório do MVP do Projeto ARARAT, focado na visualização e análise de risco de câncer de próstata.

## 🚀 Quickstart (Início Rápido)

1.  **Prepare o ambiente de inferência** (Executar uma única vez no PowerShell):
    ```powershell
    .\scripts\setup_inference_env.ps1
    ```
    *Isso cria a pasta `.venv_infer` necessária para rodar o modelo de IA.*

2.  **Execute o Viewer**:
    Certifique-se de estar no seu ambiente Python principal (ex: `.venv39`) e rode:
    ```powershell
    python -m viewer.viewer_app --data_root "C:\Caminho\Para\Seus\Dados_PROSTATEx"
    ```
    *Dica: Use `--series_hint t2` se suas séries T2 tiverem nomes diferentes.*

3.  **No Viewer (Cheatsheet de Atalhos)**:
    
    | Categoria | Tecla | Ação |
    | :--- | :--- | :--- |
    | **Navegação** | `Scroll` / `Arrastar` | Zoom / Pan (Mover) |
    | | `Setas` | Trocar Slice (Fatia) |
    | | `A` / `K` / `S` | Visão Axial / Coronal / Sagittal |
    | | `[ ` / ` ]` | Paginar lista de séries |
    | | `1`..`9` | Ir para série N (atalho rápido) |
    | | `Ctrl` + `G` | Ir para série específica (digitar número) |
    | | `Ctrl` + `↑`/`↓` | Próximo / Anterior Paciente |
    | | `R` | Resetar visão atual |
    | | `Shift` + `R` | Resetar todas as visões |
    | **ROI (Lesão)** | `Clique Esq.` | Marcar ponto (travar centro) |
    | | `+` / `-` | Aumentar / Diminuir raio da lesão |
    | | `Enter` | **CONFIRMAR** lesão (salva memória) |
    | | `X` | Cancelar seleção atual |
    | | `Del` | Apagar última lesão confirmada |
    | **Geral** | `E` | **EXPORTAR & INFERIR** (Roda IA) |
    | | `F` | Abrir pasta de exports no Windows |
    | | `V` | Validar se ROIs estão dentro do volume |
    | | `G` | Ligar/Desligar Gabarito (GT) |
    | | `Shift` + `G` | Pular para fatia da lesão GT |
    | | `P` | Ligar/Desligar Painel de Predições |
    | | `H` | Mostrar/Ocultar Ajuda na tela |
    | | `D` | Modo Debug (Layout) |
    | | `Q` | Sair |

    *Os resultados da inferência aparecem na barra lateral e são salvos em `exports/`.*

---

## 📚 Documentação Completa

Para detalhes técnicos profundos, arquitetura, lista completa de atalhos e guia de manutenção, consulte o documento de handoff:

👉 **[Documentação Técnica e Handoff (docs/ARARAT_VIEWER_HANDOFF.md)](docs/ARARAT_VIEWER_HANDOFF.md)**

---

## Estrutura Resumida
*   `viewer/`: Código da aplicação gráfica.
*   `inference/`: Modelos e scripts de ML.
*   `scripts/`: Utilitários de setup e validação.
*   `exports/`: Saída de dados (ROIs, Máscaras, JSONs de predição).

## Contato / Manutenção
Este projeto foi desenvolvido com suporte do agente TRAE. Consulte o histórico de commits e a documentação em `docs/` para manter o contexto.
