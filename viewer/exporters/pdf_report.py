
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
        if x < 0: return "Direita"  
        if x > 0: return "Esquerda"
        return "Direita"
    except:
        return "N/D"

def generate_report(case_name, export_dir, output_path, patient_id_real=None, series_name="T2 Axial"):
    """
    Gera o relatório PDF ARARAT CDS.
    """
    print(f"[PDF] Gerando relatorio para {case_name} em {output_path}...")
    
    # carrega dados
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

    # carrega predições
    lesions_info = []
    for i, roi in enumerate(rois_data):
        lid = roi.get("id", f"L{i+1}")
        
        pred_data = {}
        for fname in os.listdir(export_dir):
            if fname.startswith("pred_mask_") and fname.endswith(".json") and (f"_{lid}." in fname or f"_{lid}_" in fname):
                try:
                    with open(os.path.join(export_dir, fname), 'r') as f:
                        pred_data = json.load(f)
                    break
                except:
                    pass
        
        prob = pred_data.get("prob_pos", pred_data.get("risk_percent", None))
        if prob is not None and prob > 1.0: prob /= 100.0
        
        risk_cat = pred_data.get("risk_category", _get_risk_category(prob))

        center = roi.get("center_xyz_mm", [0,0,0])
        radius = roi.get("radius_mm", 0)
        side = _get_side_from_x(center[0]) if center else "N/D"
        
        zone = "N/D" 

        lesions_info.append({
            "id": lid,
            "prob": prob,
            "risk": risk_cat,
            "side": side,
            "radius": radius,
            "zone": zone,
            "center": center
        })

    # configura pdf
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.5*cm, leftMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    
    styles = getSampleStyleSheet()
    styleN = styles["Normal"]
    styleH = styles["Heading1"]
    styleH.alignment = 1 # Center
    
    elements = []
    

    header_text = Paragraph("<b>ARARAT® — Relatório de Suporte à Decisão Clínica</b>", styleH)
    elements.append(header_text)
    elements.append(Spacer(1, 0.8*cm))
    
    data_meta = [
        ["Modalidade:", "RM de Próstata (MP)"],
        ["Sequência:", series_name[:20]],
        ["Análise:", "Radiômica (T2W)"],
        ["Finalidade:", "Suporte à Decisão (CDS)"],
        ["Paciente:", f"{patient_id_real or 'N/D'} ({case_name})"],
        ["Data/Hora:", datetime.now().strftime("%d/%m/%Y %H:%M")]
    ]
    
    # Ajuste para Portrait (Largura Total ~18cm)
    # Meta Table: 7.5cm
    t_meta = Table(data_meta, colWidths=[2.5*cm, 5.0*cm])
    t_meta.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (0,-1), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 4),
    ]))
    

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
    
    data_top = [[t_meta, p_text]]
    t_top = Table(data_top, colWidths=[8.0*cm, 10.0*cm])
    t_top.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    elements.append(t_top)
    elements.append(Spacer(1, 1.0*cm))
    
    lesion_tables = []
    
    for l in lesions_info:
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
        
        t_les = Table(rows, colWidths=[4.0*cm, 5.0*cm, 9.0*cm])
        t_les.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('PADDING', (0,0), (-1,-1), 6),
        ]))
        
        # mantem titulo e tabela juntos
        lesion_tables.append([title, t_les])

    if not lesion_tables:
        elements.append(Paragraph("Nenhuma lesão detectada/exportada.", styleN))
    else:
        for item in lesion_tables:
            elements.append(item[0]) # titulo (L1)
            elements.append(item[1]) # tabela
            elements.append(Spacer(1, 0.8*cm))

    # --- FOOTER ---
    elements.append(Spacer(1, 1.0*cm))
    footer_text = f"Export Folder: {os.path.basename(export_dir)} | Gerado por ARARAT Viewer MVP"
    elements.append(Paragraph(f"<font size=8 color=grey>{footer_text}</font>", styleN))

    # build
    try:
        doc.build(elements)
        print(f"[PDF] Sucesso: {output_path}")
        return True
    except Exception as e:
        print(f"[PDF] Falha no build: {e}")
        return False

if __name__ == "__main__":
    print("Teste de geracao de PDF...")