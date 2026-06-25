from html import escape
from urllib.parse import quote_plus

import streamlit as st


def normalize_tips(tips):
    """Ubah tips dari list atau string menjadi string yang aman untuk ditampilkan."""
    if isinstance(tips, list):
        return " ".join(str(tip) for tip in tips)
    return str(tips)


def render_tone_box(tone, color):
    """Kotak warna skin tone untuk bagian 'Skin tone categories'."""
    txt = "#2C1A1D" if tone in ["Fair", "Medium"] else "#FFF8F8"
    st.markdown(
        f'<div class="tone-box" style="background:{color};color:{txt};">{tone}</div>',
        unsafe_allow_html=True,
    )


def render_confidence_bars(probabilities, final_tone):
    """Bar chart horizontal untuk menampilkan confidence tiap skin tone."""
    for label, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        fill_color = "#C2748A" if label == final_tone else "rgba(194,116,138,0.35)"
        weight     = "700"    if label == final_tone else "400"
        st.markdown(f"""
        <div class="conf-row">
            <div class="conf-label-row">
                <span style="font-weight:{weight};">{label}</span>
                <span style="opacity:0.65;">{prob:.1%}</span>
            </div>
            <div class="conf-track">
                <div class="conf-fill" style="width:{prob*100:.1f}%;background:{fill_color};"></div>
            </div>
        </div>""", unsafe_allow_html=True)


def get_search_url(prod):
    """Buat URL Google Search untuk produk foundation tertentu."""
    q = f"{prod.get('brand','')} {prod.get('product','')} {prod.get('shade_name','')} {prod.get('shade_code','')} foundation"
    return "https://www.google.com/search?q=" + quote_plus(q.strip())


def render_product_cards(products, final_tone, rec):
    """Render grid card untuk tiap produk foundation yang direkomendasikan."""
    cards = []
    for i, prod in enumerate(products, start=1):
        shade_name  = str(prod.get("shade_name", prod.get("product", "")))
        shade_code  = str(prod.get("shade_code", ""))
        shade_text  = f"{shade_name} · {shade_code}" if shade_code and shade_code not in ("-", "nan", "None") else shade_name
        hex_color   = str(prod.get("hex", "#cccccc"))
        brand       = str(prod.get("brand", "-")).upper()
        product     = str(prod.get("product", "-"))
        # Produk direkomendasikan berdasarkan kecocokan warna/lightness.
        # Saran finish/coverage ditampilkan di summary, bukan sebagai atribut produk.
        finish      = str(rec.get("suggested_finish", ""))
        coverage    = str(rec.get("suggested_coverage", ""))
        search_url  = get_search_url(prod)
        dataset_url = str(prod.get("url", "")).strip()
        has_url     = dataset_url.startswith("http")

        # Tombol aksi: prioritas ke URL dataset, fallback ke Google Search
        if has_url:
            btn_html = (
                f'<a href="{dataset_url}" target="_blank" rel="noopener" class="prod-btn prod-btn-primary">View Product</a>'
                f'<a href="{search_url}" target="_blank" rel="noopener" class="prod-btn prod-btn-secondary">Search Online</a>'
                f'<div class="prod-url-note">Some product links may no longer be active.</div>'
            )
        else:
            btn_html = (
                f'<a href="{search_url}" target="_blank" rel="noopener" class="prod-btn prod-btn-secondary">Search Online</a>'
                f'<div class="prod-url-note">No direct URL in dataset</div>'
            )

        cards.append(f"""
        <div class="prod-card">
            <div class="prod-swatch" style="background:{hex_color};"></div>
            <div class="prod-body">
                <span class="prod-rank">#{i} Match</span>
                <div class="prod-brand">{escape(brand)}</div>
                <div class="prod-name">{escape(product)}</div>
                <div class="prod-shade">{escape(shade_text)}</div>
                <div class="prod-pills">
                    <span class="prod-pill prod-pill-tone">{escape(str(final_tone))}</span>
                    <span class="prod-pill prod-pill-match">Color match</span>
                </div>
            </div>
            <div class="prod-actions">{btn_html}</div>
        </div>""")

    st.markdown('<div class="prod-grid">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_match_summary(final_tone, skin_type_key, finish, coverage, skin_tone_colors, skin_type_labels_short):
    """Render grid 4-kolom berisi ringkasan hasil analisis.

    Parameter skin_tone_colors dan skin_type_labels_short di-pass dari
    app.py agar modul ini tidak perlu mengimpor konfigurasi global secara langsung.
    """
    tone_bg = skin_tone_colors.get(final_tone, "var(--card-bg)")
    items = [
        ("Skin Tone", final_tone,                            "summary-tone-card", f'<div class="summary-tone-accent" style="background:{tone_bg};"></div>'),
        ("Skin Type", skin_type_labels_short[skin_type_key], "",                  ""),
        ("Suggested Finish", finish,                          "",                  ""),
        ("Suggested Coverage", coverage,                      "",                  ""),
    ]
    cards = []
    for label, value, extra_class, accent_html in items:
        cards.append(
            f'<div class="summary-item-html {extra_class}">'
            f'{accent_html}'
            f'<div class="summary-item-label">{escape(str(label))}</div>'
            f'<div class="summary-item-value">{escape(str(value))}</div>'
            f'</div>'
        )
    st.markdown('<div class="summary-grid-html">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_faq(items):
    """Render FAQ menggunakan native Streamlit expander (theme-aware, no iframe)."""
    for q, a in items:
        with st.expander(q):
            st.markdown(
                f'<div class="faq-answer-body">{escape(a)}</div>',
                unsafe_allow_html=True,
            )