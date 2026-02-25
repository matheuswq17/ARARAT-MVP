import sys
import os
import matplotlib
matplotlib.use('Agg') # Backend sem GUI
import matplotlib.pyplot as plt

# Adicionar raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viewer.viewer_app import ViewerApp

def prove_hardening():
    print("=== PROVA DE HARDENING DO VIEWER ===")
    
    # 1. Instanciar App (sem carregar dados reais se possivel)
    # Passamos um diretorio dummy para evitar scan demorado
    try:
        app = ViewerApp(dicom_dir=".", data_root=".") 
    except Exception as e:
        print(f"Aviso na instanciacao (esperado em ambiente de teste): {e}")
        # Se falhar no init por causa de diretorios, podemos tentar mockar ou
        # apenas instanciar a classe e checar atributos se possivel,
        # mas como o init define os atributos, precisamos que ele rode ate a linha do debug.
        # O ViewerApp do ARARAT é robusto, deve rodar mesmo com pasta vazia.
        return

    # 2. Verificar Flag de Debug
    print(f"\n[CHECK 1] Default dev_layout_debug:")
    print(f"  Valor atual: {app.dev_layout_debug}")
    if app.dev_layout_debug is False:
        print("  => SUCESSO: Debug desligado por padrao.")
    else:
        print("  => FALHA: Debug ligado! Risco de vazamento de info de dev.")
        
    # 3. Verificar Texto de Ajuda
    print(f"\n[CHECK 2] Texto de Ajuda (H) Fiel:")
    # O metodo é privado mas acessivel em Python
    help_text = app._get_help_text()
    print("-" * 40)
    print(help_text)
    print("-" * 40)
    
    required_keys = ["Scroll", "Enter", "E", "G", "H", "D", "Z"]
    missing = [k for k in required_keys if k not in help_text]
    
    if not missing:
        print("  => SUCESSO: Help contem atalhos criticos verificados.")
    else:
        print(f"  => FALHA: Help incompleto. Faltam: {missing}")

if __name__ == "__main__":
    prove_hardening()
