
import os
import json
import time
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.units import cm, mm

def _safe_get(d, key, default="N/D"):
    return d.get(key, default) if d.get(key) is not None else default

def _format_prob(p):
    try:
        val = float(p)
        return f"{val*100:.1f}%"
    except:
        return "N/D"

def _get_risk_category(prob):
    try:
        p = float(prob)
        if p < 0.30: return "Baixo risco"
        if p < 0.60: return "Risco intermediário"
        return "Alto risco"
    except:
        return "N/D"

def _get_side_from_x(x_mm):
    try:
        x = float(x_mm)
        if x < 0: return "Direita" # DICOM LPS: +X é Esquerda (L), -X é Direita (R). 
        # Wait, standard DICOM coords: 
        # X increases to the Patient's Left. So X > 0 is Left, X < 0 is Right.
        # Let's assume standard DICOM unless proven otherwise.
        if x > 0: return "Esquerda"
        return "Direita"
    except:
        return "N/D"

def generate_report(case_name, export_dir, output_path, patient_id_real=None, series_name="T2 Axial"):
    """
    Gera o relatório PDF ARARAT CDS.
    """
    print(f"[PDF] Gerando relatorio para {case_name} em {output_path}...")
    
    # 1. Carregar Dados
    rois_path = os.path.join(export_dir, "rois.json")
    if not os.path.exists(rois_path):
        rois_path = os.path.join(export_dir, "rois_latest.json")
    
    rois_data = []
    if os.path.exists(rois_path):
        try:
            with open(rois_path, 'r') as f:
                data = json.load(f)
                rois_data = data.get("rois", [])
        except Exception as e:
            print(f"[PDF] Erro ao ler rois.json: {e}")

    # Carregar predições
    lesions_info = []
    for i, roi in enumerate(rois_data):
        lid = roi.get("id", f"L{i+1}")
        
        # Tentar achar pred_mask_L*.json
        # O padrão pode ser pred_mask_{lid}.json ou algo similar
        # Vamos procurar na pasta
        pred_data = {}
        for fname in os.listdir(export_dir):
            if fname.startswith("pred_mask_") and fname.endswith(".json") and (f"_{lid}." in fname or f"_{lid}_" in fname):
                try:
                    with open(os.path.join(export_dir, fname), 'r') as f:
                        pred_data = json.load(f)
                    break
                except:
                    pass
        
        # Se não achou arquivo específico, tentar achar no case_summary.json se existir (opcional)
        # Mas vamos focar no pred_mask individual que é garantido pelo pipeline
        
        prob = pred_data.get("prob_pos", pred_data.get("risk_percent", None))
        if prob is not None and prob > 1.0: prob /= 100.0 # Normalizar se vier 0-100
        
        risk_cat = pred_data.get("risk_category", _get_risk_category(prob))
        
        # Dados Geométricos
        center = roi.get("center_xyz_mm", [0,0,0])
        radius = roi.get("radius_mm", 0)
        side = _get_side_from_x(center[0]) if center else "N/D"
        
        # GT Info (Zone) - Será preenchido pelo caller se possível, ou N/D aqui
        # O viewer passa essas infos? O viewer tem acesso ao GT. 
        # Vamos tentar inferir ou deixar N/D. 
        # Melhor: O viewer deve passar o objeto de GT se tiver.
        # Como estamos lendo do disco, não temos o GT em memória a menos que salvemos.
        # Vamos assumir N/D por enquanto para Zone, a menos que esteja no pred (as vezes o modelo prediz zona)
        zone = "N/D" 

        lesions_info.append({
            "id": lid,
            "prob": prob,
            "risk": risk_cat,
            "side": side,
            "radius": radius,
            "zone": zone, # Placeholder
            "center": center
        })

    # 2. Configurar PDF
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(A4),
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleH = styles["Heading1"]
    styleH.alignment = 1 # Center
    
    elements = []
    
    # --- HEADER ---
    header_text = Paragraph("<b>ARARAT® — Relatório de Suporte à Decisão Clínica</b>", styleH)
    elements.append(header_text)
    elements.append(Spacer(1, 1.0*cm))
    
    # --- BLOCO SUPERIOR (2 Colunas) ---
    # Coluna 1: Metadados
    data_meta = [
        ["Modalidade:", "RM de Próstata (Multiparamétrica)"],
        ["Sequência:", series_name],
        ["Análise:", "Radiômica por Lesão (T2W)"],
        ["Finalidade:", "Suporte à Decisão Clínica (CDS)"],
        ["Paciente:", f"{patient_id_real or 'N/D'} ({case_name})"],
        ["Data/Hora:", datetime.now().strftime("%d/%m/%Y %H:%M")]
    ]
    
    t_meta = Table(data_meta, colWidths=[3.5*cm, 6.0*cm])
    t_meta.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    
    # Coluna 2: Texto Sugerido
    text_content = "<b>Texto sugerido para o laudo radiológico (CDS):</b><br/><br/>"
    text_content += f"A análise ARARAT identificou {len(lesions_info)} lesões na próstata.<br/>"
    
    for l in lesions_info:
        p_str = _format_prob(l['prob'])
        text_content += (f"• <b>{l['id']}</b> (zona {l['zone']}, {l['side']}): "
                         f"Probabilidade de <b>{p_str}</b> p/ ISUP ≥ 3 ({l['risk']}).<br/>")
    
    text_content += ("<br/><i>Estes achados devem ser interpretados em conjunto com dados clínicos, "
                     "PSA, histopatologia e fatores específicos do paciente. "
                     "Esta ferramenta não substitui o laudo do radiologista.</i>")
    
    p_text = Paragraph(text_content, styleN)
    
    # Layout Tabela Principal (Topo)
    data_top = [[t_meta, p_text]]
    t_top = Table(data_top, colWidths=[10*cm, 16*cm])
    t_top.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(t_top)
    elements.append(Spacer(1, 1.0*cm))
    
    # --- LESÕES (Parte de Baixo) ---
    # Vamos criar tabelas lado a lado se houver > 1 lesão
    
    lesion_tables = []
    
    for l in lesions_info:
        # Tabela da Lesão
        title = Paragraph(f"<b>{l['id']}</b>", styles["Heading3"])
        
        rows = [
            ["Categoria", "Achado", "Significado Clínico"],
            ["Visão Geral", f"ROI R={l['radius']:.1f}mm", "Região de extração radiômica"],
            ["Prob. ISUP≥3", _format_prob(l['prob']), "Estimativa de CaP sig."],
            ["Faixa de Risco", l['risk'], "Estratificação CDS"],
            ["Zona", l['zone'], "Localização anatômica"],
            ["Lado", l['side'], "Lateralidade"],
            ["PI-RADS", "N/D", "Classificação convencional"],
            ["Intensidade T2", "N/D", "Avaliação qualitativa"],
        ]
        
        t_les = Table(rows, colWidths=[3.0*cm, 4.0*cm, 5.5*cm])
        t_les.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        
        # Wrapper para manter titulo e tabela juntos
        lesion_tables.append([title, t_les])

    # Organizar tabelas de lesão em grid (2 colunas)
    if not lesion_tables:
        elements.append(Paragraph("Nenhuma lesão detectada/exportada.", styleN))
    else:
        # Agrupar em pares
        grid_data = []
        curr_row = []
        for item in lesion_tables:
            # item é [Title, Table]
            # Vamos colocar numa celula interna
            sub_t = Table([[item[0]], [item[1]]], colWidths=[12.5*cm])
            sub_t.setStyle(TableStyle([('LEFTPADDING', (0,0), (-1,-1), 0)]))
            
            curr_row.append(sub_t)
            if len(curr_row) == 2:
                grid_data.append(curr_row)
                curr_row = []
        
        if curr_row:
            curr_row.append("") # Spacer se impar
            grid_data.append(curr_row)
            
        t_grid = Table(grid_data, colWidths=[13*cm, 13*cm])
        t_grid.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ]))
        elements.append(t_grid)

    # --- FOOTER ---
    elements.append(Spacer(1, 1.0*cm))
    footer_text = f"Export Folder: {os.path.basename(export_dir)} | Gerado por ARARAT Viewer MVP"
    elements.append(Paragraph(f"<font size=8 color=grey>{footer_text}</font>", styleN))

    # Build
    try:
        doc.build(elements)
        print(f"[PDF] Sucesso: {output_path}")
        return True
    except Exception as e:
        print(f"[PDF] Falha no build: {e}")
        return False

if __name__ == "__main__":
    # Teste isolado
    print("Teste de geracao de PDF...")
    # Mock data
    # generate_report("TestCase", ".", "test_report.pdf")
