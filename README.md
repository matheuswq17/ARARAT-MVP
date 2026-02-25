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

3.  **No Viewer**:
    *   Use **A / K / S** para alternar as visões (Axial, Coronal, Sagittal).
    *   **Clique** para marcar uma lesão e **Enter** para confirmar a ROI.
    *   Pressione **E** para exportar e rodar a inferência de risco.
    *   Os resultados aparecem na tela e são salvos em `exports/`.

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
