from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageColor

# Warna representasi tiap skin tone untuk swatch pada shade card.
SKIN_TONE_COLORS = {
    "Fair"  : "#F3D7C2",
    "Medium": "#C98F68",
    "Tan"   : "#8D5524",
    "Deep"  : "#4B2E1E",
}

# Label skin type pendek untuk ditampilkan pada chip profil di shade card.
SKIN_TYPE_LABELS_SHORT = {
    "oily"       : "Oily",
    "dry"        : "Dry",
    "combination": "Combination",
    "normal"     : "Normal",
    "sensitive"  : "Sensitive",
}


# ============================================================
# FONT & COLOR HELPERS
# ============================================================

def _get_font(size, bold=False):
    """Ambil font yang aman untuk local dan Streamlit Cloud."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    ]
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _safe_color(hex_color, fallback="#C98F68"):
    """Validasi warna hex dari dataset agar aman dipakai di card."""
    try:
        return ImageColor.getrgb(str(hex_color).strip())
    except Exception:
        return ImageColor.getrgb(fallback)


# ============================================================
# TEXT HELPERS
# ============================================================

def _fit_text(draw, text, font, max_width):
    """Potong text panjang dengan ellipsis agar tidak keluar card."""
    text = str(text).strip()
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return text.strip() + "…" if text else "…"


def _wrap_lines(draw, text, font, max_width, max_lines=2):
    """Wrap text pendek untuk nama produk/shade."""
    words = str(text).replace("\n", " ").split()
    lines, line = [], ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textlength(test, font=font) <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
        if len(lines) == max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    if lines and draw.textlength(lines[-1], font=font) > max_width:
        lines[-1] = _fit_text(draw, lines[-1], font, max_width)
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = _fit_text(draw, lines[-1] + "…", font, max_width)
    return lines


def _draw_centered(draw, xy, text, font, fill, max_width=None):
    """Draw text centered pada titik x tertentu."""
    x, y = xy
    text = str(text)
    if max_width:
        text = _fit_text(draw, text, font, max_width)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((x - w / 2, y), text, font=font, fill=fill)


def _product_display_text(prod):
    """Gabungkan nama shade agar card tetap ringkas."""
    shade_name = str(prod.get("shade_name", "")).strip()
    shade_code = str(prod.get("shade_code", "")).strip()
    invalid = {"", "-", "nan", "None", "none", "NaN"}
    if shade_name in invalid:
        shade_name = str(prod.get("product", "Shade match")).strip()
    if shade_code not in invalid and shade_code.lower() not in shade_name.lower():
        return f"{shade_name} · {shade_code}"
    return shade_name


# ============================================================
# DECORATIVE DRAWING HELPERS
# ============================================================

def _draw_sparkle(draw, cx, cy, size, fill):
    """Gambar bintang 4-titik kecil (sparkle) sebagai dekorasi playful.
    Digambar sebagai dua diamond bertumpuk (vertikal panjang + horizontal pendek)
    agar terlihat seperti ikon sparkle tanpa butuh font emoji.
    """
    draw.polygon(
        [(cx, cy - size), (cx + size * 0.22, cy), (cx, cy + size), (cx - size * 0.22, cy)],
        fill=fill,
    )
    draw.polygon(
        [(cx - size * 0.6, cy), (cx, cy - size * 0.22), (cx + size * 0.6, cy), (cx, cy + size * 0.22)],
        fill=fill,
    )


def _lerp_color(c1, c2, t):
    """Interpolasi linear antara dua warna RGB."""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _draw_gradient_strip(img, draw, box, colors, radius):
    """Gambar strip horizontal dengan gradient blend antar warna produk."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    n = len(colors)

    if n == 1:
        draw.rounded_rectangle(box, radius=radius, fill=colors[0])
        return

    # Buat gradient di layer terpisah lalu mask rounded-rect agar sudut tetap halus
    grad = Image.new("RGB", (width, height))
    seg = width / (n - 1)
    for x in range(width):
        pos = x / seg
        idx = min(int(pos), n - 2)
        t = pos - idx
        col = _lerp_color(colors[idx], colors[idx + 1], t)
        for y in range(height):
            grad.putpixel((x, y), col)

    mask = Image.new("L", (width, height), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    img.paste(grad, (int(x1), int(y1)), mask)


# ============================================================
# MAIN CARD BUILDER
# ============================================================

def create_shade_card_image(final_result, skin_type_key):
    """Buat image result card yang cute, compact, dan shareable.

    Versi ini dibuat sebagai PNG share card:
    - Top 5 shade matches saja
    - tanpa foto wajah, confidence, lightness, distance, dan link produk
    """
    rec = final_result["recommendation"]
    final_tone = final_result["final_skin_tone"]
    products = rec.get("recommended_products", [])[:5]

    # Canvas dibuat lebih besar supaya hasil download lebih tajam.
    W, H = 1400, 1780

    # Palette dasar
    rose = "#BC4F72"
    rose_dark = "#7E334B"
    rose_mid = "#A94667"
    rose_soft = "#FAE7EE"
    rose_line = "#E8BACA"
    ink = "#28171C"
    muted = "#7B656D"
    soft_card = "#FFF7FA"
    bg = "#FFF7FA"

    tone_rgb = _safe_color(SKIN_TONE_COLORS.get(final_tone, "#C98F68"))

    TONE_VIBES = {
        "Fair": "Soft & Fresh",
        "Medium": "Warm & Glowing",
        "Tan": "Golden & Confident",
        "Deep": "Bold & Radiant",
    }
    tone_vibe = TONE_VIBES.get(final_tone, "Beauty Match")

    # Fonts
    f_brand = _get_font(44, True)
    f_kicker = _get_font(24, True)
    f_title = _get_font(92, True)
    f_sub = _get_font(32, False)
    f_vibe = _get_font(26, True)
    f_chip_label = _get_font(19, True)
    f_chip_value = _get_font(27, True)
    f_h2 = _get_font(46, True)
    f_small = _get_font(22, False)
    f_small_b = _get_font(22, True)
    f_rank = _get_font(24, True)
    f_card_title = _get_font(27, True)
    f_card_sub = _get_font(21, False)
    f_note = _get_font(22, True)
    f_tiny = _get_font(18, False)
    f_tiny_b = _get_font(18, True)

    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Decorative background blobs
    draw.ellipse((-210, -205, 500, 390), fill="#FFE0EA")
    draw.ellipse((1020, -170, 1600, 300), fill="#F6C7D6")
    draw.ellipse((-260, H - 320, 330, H + 120), fill="#FFF0F4")
    draw.ellipse((985, H - 300, 1560, H + 130), fill="#FFE3EB")

    # Outer card
    margin = 70
    outer = (margin, 60, W - margin, H - 78)
    draw.rounded_rectangle(outer, radius=62, fill="#FFFFFF", outline=rose_line, width=3)

    content_x1 = outer[0] + 66
    content_x2 = outer[2] - 66
    content_w = content_x2 - content_x1
    cx = W / 2

    # ── Header
    y = 116
    _draw_centered(draw, (cx, y), "MatchMyShade", f_brand, rose_dark)
    y += 62

    kicker = "MY SHADE CARD"
    kicker_w = draw.textlength(kicker, font=f_kicker)
    pill_h = 50
    pill_w = kicker_w + 64
    draw.rounded_rectangle(
        (cx - pill_w / 2, y, cx + pill_w / 2, y + pill_h),
        radius=pill_h / 2,
        fill=rose_soft,
    )
    _draw_centered(draw, (cx, y + 11), kicker, f_kicker, rose)
    y += 92

    # ── Main result
    _draw_centered(draw, (cx, y), f"I got {final_tone}!", f_title, ink, max_width=content_w - 40)
    y += 106
    _draw_centered(draw, (cx, y), "my foundation shade vibe", f_sub, muted)
    y += 78

    # Main tone swatch
    sw_cx, sw_cy = int(cx), y + 74
    outer_r, inner_r = 72, 48
    draw.ellipse(
        (sw_cx - outer_r, sw_cy - outer_r, sw_cx + outer_r, sw_cy + outer_r),
        fill="#FFF7FA",
        outline="#F7CEDB",
        width=8,
    )
    draw.ellipse(
        (sw_cx - inner_r, sw_cy - inner_r, sw_cx + inner_r, sw_cy + inner_r),
        fill=tone_rgb,
        outline="#FFFFFF",
        width=6,
    )
    _draw_sparkle(draw, sw_cx + 102, sw_cy - 50, 34, rose)
    _draw_sparkle(draw, sw_cx - 100, sw_cy + 66, 22, "#E8A8BE")
    y += 166

    # Vibe badge
    vibe_w = draw.textlength(tone_vibe, font=f_vibe) + 66
    vibe_h = 62
    draw.rounded_rectangle((cx - vibe_w / 2, y, cx + vibe_w / 2, y + vibe_h), radius=31, fill=tone_rgb)
    vibe_text_color = "#2C1A1D" if final_tone in ["Fair", "Medium"] else "#FFF8F8"
    _draw_centered(draw, (cx, y + 17), tone_vibe, f_vibe, vibe_text_color)
    y += 102

    # ── Profile chips
    chip_data = [
        ("Skin type", SKIN_TYPE_LABELS_SHORT.get(skin_type_key, str(skin_type_key).title())),
        ("Best finish", rec.get("suggested_finish", "-")),
        ("Coverage", rec.get("suggested_coverage", "-")),
    ]
    chip_gap = 24
    chip_w = (content_w - chip_gap * 2) / 3
    chip_h = 100

    for i, (label, value) in enumerate(chip_data):
        x1 = content_x1 + i * (chip_w + chip_gap)
        x2 = x1 + chip_w
        draw.rounded_rectangle((x1, y, x2, y + chip_h), radius=28, fill=soft_card, outline=rose_line, width=2)
        _draw_centered(draw, ((x1 + x2) / 2, y + 22), label.upper(), f_chip_label, rose_mid, max_width=chip_w - 30)
        _draw_centered(draw, ((x1 + x2) / 2, y + 58), value, f_chip_value, ink, max_width=chip_w - 34)
    y += chip_h + 78

    # ── Palette section
    _draw_centered(draw, (cx, y), "My Top 5 Shade Palette", f_h2, rose_dark)
    y += 58
    _draw_centered(draw, (cx, y), "save these colors for your next foundation hunt", f_small, muted)
    y += 58

    # Palette circles
    if products:
        pal_r = 55
        overlap = 26
        n = len(products)
        total_w = pal_r * 2 + (n - 1) * (pal_r * 2 - overlap)
        start_x = cx - total_w / 2 + pal_r
        offsets = [0, -10, 12, -8, 10]
        pal_y = y + pal_r + 10

        positions = []
        for idx, prod in enumerate(products):
            pcx = start_x + idx * (pal_r * 2 - overlap)
            pcy = pal_y + offsets[idx % len(offsets)]
            positions.append((pcx, pcy))
            prod_rgb = _safe_color(prod.get("hex", "#C98F68"))
            draw.ellipse((pcx - pal_r - 5, pcy - pal_r - 5, pcx + pal_r + 5, pcy + pal_r + 5), fill="#FFFFFF")
            draw.ellipse((pcx - pal_r, pcy - pal_r, pcx + pal_r, pcy + pal_r), fill=prod_rgb)

        num_r = 22
        for idx, (pcx, pcy) in enumerate(positions):
            ncx, ncy = pcx - pal_r + 12, pcy - pal_r + 12
            draw.ellipse((ncx - num_r, ncy - num_r, ncx + num_r, ncy + num_r), fill="#FFFFFF", outline=rose_line, width=2)
            _draw_centered(draw, (ncx, ncy - 11), str(idx + 1), f_small_b, rose_dark)

    y += 152

    # ── Compact top matches
    _draw_centered(draw, (cx, y), "Top matches", f_h2, rose_dark)
    y += 62

    card_gap_x = 26
    card_gap_y = 18
    card_w = (content_w - card_gap_x) / 2
    card_h = 92

    positions = [
        (content_x1, y),
        (content_x1 + card_w + card_gap_x, y),
        (content_x1, y + card_h + card_gap_y),
        (content_x1 + card_w + card_gap_x, y + card_h + card_gap_y),
        (cx - card_w / 2, y + (card_h + card_gap_y) * 2),
    ]

    for idx, prod in enumerate(products[:5]):
        x1, y1 = positions[idx]
        x2, y2 = x1 + card_w, y1 + card_h

        draw.rounded_rectangle((x1, y1, x2, y2), radius=26, fill="#FFF9FB", outline=rose_line, width=2)

        # Rank badge
        rank_r = 28
        rb_cx, rb_cy = x1 + 44, y1 + card_h / 2
        draw.ellipse((rb_cx - rank_r, rb_cy - rank_r, rb_cx + rank_r, rb_cy + rank_r), fill=rose)
        _draw_centered(draw, (rb_cx, rb_cy - 13), str(idx + 1), f_rank, "#FFFFFF")

        # Swatch
        sw_size = 44
        sw_x = x1 + 92
        sw_y = y1 + (card_h - sw_size) / 2
        draw.rounded_rectangle(
            (sw_x, sw_y, sw_x + sw_size, sw_y + sw_size),
            radius=12,
            fill=_safe_color(prod.get("hex", "#C98F68")),
        )

        brand = str(prod.get("brand", "-")).strip().title()
        shade = _product_display_text(prod)
        text_x = sw_x + sw_size + 20
        max_text_w = card_w - (text_x - x1) - 20

        draw.text((text_x, y1 + 16), _fit_text(draw, brand, f_card_title, max_text_w), font=f_card_title, fill=ink)
        draw.text((text_x, y1 + 51), _fit_text(draw, shade, f_card_sub, max_text_w), font=f_card_sub, fill=muted)

    if products:
        y = y + (card_h + card_gap_y) * 3 + 48
    else:
        _draw_centered(draw, (cx, y + 25), "No product recommendation available", f_sub, muted)
        y += 95

    # ── Beauty note
    note_h = 92
    draw.rounded_rectangle((content_x1, y, content_x2, y + note_h), radius=28, fill=rose_soft, outline=rose_line, width=2)
    _draw_centered(draw, (cx, y + 21), "Beauty note: try these shades under natural light.", f_note, rose_dark, max_width=content_w - 60)
    _draw_centered(draw, (cx, y + 55), "Made with MatchMyShade", f_tiny, muted, max_width=content_w - 60)

    y += note_h + 58

    return img


def create_shade_card_png(final_result, skin_type_key):
    """Return PNG bytes untuk shareable shade card."""
    card_img = create_shade_card_image(final_result, skin_type_key)
    png_buffer = BytesIO()
    card_img.save(png_buffer, format="PNG")
    return png_buffer.getvalue()