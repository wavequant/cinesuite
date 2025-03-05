import os
import threading
import random
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageColor, ImageTk, ImageDraw, ImageFilter, ImageChops
import pillow_avif
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser
from concurrent.futures import ThreadPoolExecutor

# ========= Helper Functions =========

def pil_to_cv2(pil_image):
    """Convert a PIL image to an OpenCV image (BGR format)."""
    np_img = np.array(pil_image)
    if np_img.ndim == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    return np_img

def cv2_to_pil(cv2_img):
    """Convert an OpenCV image (BGR) to a PIL image (RGB)."""
    if cv2_img.ndim == 3:
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

# ========= Image Effects =========

class ImageEffects:
    @staticmethod
    def apply_halation(np_img, highlight_range, intensity, blur, strength, halation_color, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        img = np_img.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        threshold = 0.4 + 0.4 * highlight_range
        luminance = 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        mask = np.clip((luminance - threshold) * (5.0 / (1.0 + highlight_range * 2)), 0, 1)
        if progress_callback:
            progress_callback(30)
        base_blur = max(h, w) * (0.003 + 0.03 * blur)
        sigma = base_blur * (0.8 + intensity * 0.5)
        kernel_size = int(np.clip(2 * np.ceil(sigma) + 1, 3, min(h, w) // 2))
        blurred_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
        blurred_mask = cv2.GaussianBlur(blurred_mask, (kernel_size+2, kernel_size+2), sigma)
        if progress_callback:
            progress_callback(60)
        color_rgb = np.array(ImageColor.getrgb(halation_color), dtype=np.float32) / 255.0
        color_norm = color_rgb[::-1]
        glow = blurred_mask[..., np.newaxis] * (color_norm * (0.5 + 1.2 * strength))
        blended = img * (1 - intensity) + (1 - (1 - img) * (1 - glow)) * intensity
        blended = np.clip(blended, 0, 1)
        if progress_callback:
            progress_callback(100)
        return (blended * 255).astype(np.uint8)

    @staticmethod
    def apply_grain(pil_image, intensity, size, chroma, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        img_array = np.array(pil_image).astype(np.float32)
        h, w = img_array.shape[:2]
        scale_factor = 1 + 4 * ((size - 0.1) / 0.9)
        noise_h = max(1, int(h / scale_factor))
        noise_w = max(1, int(w / scale_factor))
        mono_noise = cv2.resize(np.random.normal(0, 1, (noise_h, noise_w)), (w, h), interpolation=cv2.INTER_LINEAR)
        mono_noise = np.repeat(mono_noise[:, :, np.newaxis], 3, axis=2)
        if progress_callback:
            progress_callback(30)
        noise_color = np.random.normal(0, 1, (noise_h, noise_w, 3)).astype(np.float32)
        noise_color *= np.random.uniform(0.99, 1.01, (1, 1, 3))
        noise_color = cv2.resize(noise_color, (w, h), interpolation=cv2.INTER_LINEAR)
        final_noise = mono_noise * (1 - chroma) + noise_color * chroma
        final_noise = final_noise * intensity * 30
        cluster_mask = cv2.resize(
            cv2.GaussianBlur(np.random.uniform(0.5, 1.5, (noise_h, noise_w)).astype(np.float32), (7, 7), 0),
            (w, h), interpolation=cv2.INTER_LINEAR)
        final_noise *= cluster_mask[..., np.newaxis]
        final_noise = cv2.GaussianBlur(final_noise, (3, 3), 0)
        if progress_callback:
            progress_callback(60)
        gray = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
        multiplier = 1.0 - np.power(np.clip(gray / 255.0, 0, 1), 2)
        final_noise *= multiplier[..., np.newaxis]
        result = img_array + final_noise
        result = np.clip(result, 0, 255).astype(np.uint8)
        if progress_callback:
            progress_callback(100)
        return Image.fromarray(result)

    @staticmethod
    def apply_color_fringing(pil_image, intensity):
        r, g, b = pil_image.split()
        shift = int(5 * intensity)
        r = ImageChops.offset(r, -shift, 0)
        b = ImageChops.offset(b, shift, 0)
        return Image.merge("RGB", (r, g, b))

    @staticmethod
    def apply_vignette(pil_image, intensity, radius, feather, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        width, height = pil_image.size
        x_center, y_center = width / 2, height / 2
        max_dist = np.sqrt(x_center**2 + y_center**2) * radius
        y_indices, x_indices = np.ogrid[:height, :width]
        distance = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2) / max_dist
        if progress_callback:
            progress_callback(40)
        exponent = 1 + (intensity * feather)
        mask = np.clip(1 - distance, 0, 1) ** exponent
        mask = np.dstack([mask]*3)
        img_array = np.array(pil_image).astype(np.float32)
        blended = img_array * mask + img_array * (1 - 0.3*intensity) * (1 - mask)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        if progress_callback:
            progress_callback(100)
        return Image.fromarray(blended)

    @staticmethod
    def apply_border(pil_image, border_thickness, border_color, framing):
        w, h = pil_image.size
        min_dim = min(w, h)
        border_px = int((border_thickness / 100) * min_dim)
        if framing == "Original":
            return ImageOps.expand(pil_image, border=border_px, fill=border_color)
        target_ratios = {
            "4:5": 4/5, "5:4": 5/4,
            "16:9": 16/9, "9:16": 9/16,
            "3:4": 3/4, "4:3": 4/3,
            "1:1": 1,
            "2:3": 2/3, "3:2": 3/2,
            "2:1": 2, "1:2": 0.5,
            "16:10": 16/10, "10:16": 10/16,
            "21:9": 21/9, "9:21": 9/21,
            "Custom": "Custom"
        }
        target_ratio = target_ratios.get(framing, w/h)
        new_w = w + 2*border_px
        new_h = h + 2*border_px
        new_ratio = new_w / new_h
        add_left = add_top = add_right = add_bottom = 0
        if isinstance(target_ratio, (int, float)) and abs(new_ratio - target_ratio) >= 1e-2:
            if new_ratio < target_ratio:
                desired_w = target_ratio * new_h
                extra = desired_w - new_w
                add_left = add_right = extra / 2
            else:
                desired_h = new_w / target_ratio
                extra = desired_h - new_h
                add_top = add_bottom = extra / 2
        left = int(border_px + add_left)
        right = int(border_px + add_right)
        top = int(border_px + add_top)
        bottom = int(border_px + add_bottom)
        return ImageOps.expand(pil_image, border=(left, top, right, bottom), fill=border_color)

    @staticmethod
    def generate_dust_overlay(base_image, dust_params, progress_callback=None):
        amount, size, opacity = dust_params
        w, h = base_image.size
        overlay = Image.new("RGBA", base_image.size, (0,0,0,0))
        blur_groups = {
            'small': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'base_radius': 10, 'radius': 2.0},
            'medium': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'base_radius': 12, 'radius': 3.0},
            'large': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'base_radius': 15, 'radius': 4.0},
        }
        hair_blur_groups = {
            'fine': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'radius': 1.0},
            'normal': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'radius': 2.0},
            'thick': {'overlay': Image.new("RGBA", base_image.size, (0,0,0,0)), 'radius': 3.0},
        }
        count = int(10 + amount * 20)
        for i in range(count):
            if random.random() < 0.8:
                temp = Image.new("RGBA", base_image.size, (0,0,0,0))
                draw = ImageDraw.Draw(temp)
                x = random.randint(0, w)
                y = random.randint(0, h)
                num_points = random.randint(8, 14)
                angles = np.linspace(0, 2*np.pi, num_points, endpoint=False) + np.random.uniform(-0.3, 0.3, num_points)
                rs = (random.uniform(0.2,1.8)) * (random.uniform(8,15) * size) * np.random.uniform(0.8,1.2, num_points)
                xs = x + rs * np.cos(angles)
                ys = y + rs * np.sin(angles)
                points = list(zip(xs, ys))
                poly_opacity = int(random.randint(80,240) * opacity)
                draw.polygon(points, fill=(40,40,40,poly_opacity))
                if rs.mean() > blur_groups['large']['base_radius']:
                    group_key = 'large'
                elif rs.mean() > blur_groups['medium']['base_radius']:
                    group_key = 'medium'
                else:
                    group_key = 'small'
                rand_factor = random.uniform(0.5,1.5)
                blur_radius = blur_groups[group_key]['radius'] * rand_factor
                blurred = temp.filter(ImageFilter.GaussianBlur(blur_radius))
                group_overlay = blur_groups[group_key]['overlay']
                blur_groups[group_key]['overlay'] = Image.alpha_composite(group_overlay, blurred)
            else:
                temp = Image.new("RGBA", base_image.size, (0,0,0,0))
                draw = ImageDraw.Draw(temp)
                x = random.randint(0, w)
                y = random.randint(0, h)
                chain_length = random.randint(4,8)
                segment_length = random.uniform(7,20)*size
                current_x, current_y = x, y
                angle = random.uniform(0, 2*np.pi)
                if segment_length < 10:
                    hair_group = 'fine'
                elif segment_length > 15:
                    hair_group = 'thick'
                else:
                    hair_group = 'normal'
                for k in range(chain_length):
                    seg_angle = angle + random.uniform(-0.5,0.5)
                    next_x = current_x + segment_length * np.cos(seg_angle)
                    next_y = current_y + segment_length * np.sin(seg_angle)
                    width_line = random.uniform(2,4)
                    offset_x = width_line * np.sin(seg_angle)
                    offset_y = width_line * np.cos(seg_angle)
                    polygon = [
                        (current_x - offset_x, current_y + offset_y),
                        (current_x + offset_x, current_y - offset_y),
                        (next_x + offset_x, next_y - offset_y),
                        (next_x - offset_x, next_y + offset_y)
                    ]
                    poly_opacity = int(random.randint(80,180) * opacity)
                    draw.polygon(polygon, fill=(30,30,30,poly_opacity))
                    current_x, current_y = next_x, next_y
                    angle = seg_angle
                rand_factor = random.uniform(0.5,1.5)
                blur_radius = hair_blur_groups[hair_group]['radius'] * rand_factor
                blurred = temp.filter(ImageFilter.GaussianBlur(blur_radius))
                group_overlay = hair_blur_groups[hair_group]['overlay']
                hair_blur_groups[hair_group]['overlay'] = Image.alpha_composite(group_overlay, blurred)
            if progress_callback:
                progress_callback((i+1)/count * 100)
        for group in blur_groups.values():
            overlay = Image.alpha_composite(overlay, group['overlay'])
        for group in hair_blur_groups.values():
            overlay = Image.alpha_composite(overlay, group['overlay'])
        return overlay

    @staticmethod
    def apply_light_leak(pil_image, leak_image, opacity, progress_callback=None):
        if progress_callback:
            progress_callback(10)
        if leak_image is None:
            return pil_image
        
        if pil_image.mode != "RGBA":
            pil_image = pil_image.convert("RGBA")
        
        img_w, img_h = pil_image.size
        leak_w, leak_h = leak_image.size
        
        photo_portrait = img_h > img_w
        leak_portrait = leak_h > leak_w
        if photo_portrait != leak_portrait:
            leak_image = leak_image.rotate(90, expand=True)
            leak_w, leak_h = leak_image.size
        if progress_callback:
            progress_callback(30)
        scale = max(img_w / leak_w, img_h / leak_h)
        new_leak_w = int(leak_w * scale)
        new_leak_h = int(leak_h * scale)
        leak_resized = leak_image.resize((new_leak_w, new_leak_h), Image.LANCZOS)
        
        offset_x = (new_leak_w - img_w) // 2
        offset_y = (new_leak_h - img_h) // 2
        leak_cropped = leak_resized.crop((offset_x, offset_y, offset_x+img_w, offset_y+img_h))
        
        if leak_cropped.mode != "RGBA":
            leak_cropped = leak_cropped.convert("RGBA")
        
        if progress_callback:
            progress_callback(50)
        img_array = np.array(pil_image).astype(np.float32)
        leak_array = np.array(leak_cropped).astype(np.float32)
        
        img_rgb = img_array[..., :3]
        img_alpha = img_array[..., 3:4] if img_array.shape[2] == 4 else np.ones((img_h, img_w, 1), dtype=np.float32) * 255
        
        leak_rgb = leak_array[..., :3]
        leak_alpha = leak_array[..., 3:4] if leak_array.shape[2] == 4 else np.ones((img_h, img_w, 1), dtype=np.float32) * 255
        
        enhanced_opacity = opacity * 1.8
        leak_alpha = leak_alpha * enhanced_opacity / 255.0
        
        result_rgb = 255.0 - (255.0 - img_rgb) * (255.0 - leak_rgb * min(enhanced_opacity, 1.5)) / 255.0
        
        luminance = img_rgb[..., 0] * 0.299 + img_rgb[..., 1] * 0.587 + img_rgb[..., 2] * 0.114
        highlight_mask = np.clip(luminance / 200.0 + 0.2, 0.4, 1.0).reshape(img_h, img_w, 1)
        blend_factor = highlight_mask * min(enhanced_opacity, 1.0)
        
        max_vals = np.max(leak_rgb, axis=2, keepdims=True)
        min_vals = np.min(leak_rgb, axis=2, keepdims=True)
        
        r_is_max = (leak_rgb[..., 0:1] == max_vals)
        g_is_max = (leak_rgb[..., 1:2] == max_vals)
        b_is_max = (leak_rgb[..., 2:3] == max_vals)
        
        r_boost = np.where(r_is_max, 
                        np.dstack([
                            leak_rgb[..., 0:1],
                            leak_rgb[..., 1:2] + (leak_rgb[..., 1:2] - min_vals) * 0.3,
                            leak_rgb[..., 2:3] + (leak_rgb[..., 2:3] - min_vals) * 0.3
                        ]), 0)
        
        g_boost = np.where(g_is_max, 
                        np.dstack([
                            leak_rgb[..., 0:1] + (leak_rgb[..., 0:1] - min_vals) * 0.3,
                            leak_rgb[..., 1:2],
                            leak_rgb[..., 2:3] + (leak_rgb[..., 2:3] - min_vals) * 0.3
                        ]), 0)
        
        b_boost = np.where(b_is_max, 
                        np.dstack([
                            leak_rgb[..., 0:1] + (leak_rgb[..., 0:1] - min_vals) * 0.3,
                            leak_rgb[..., 1:2] + (leak_rgb[..., 1:2] - min_vals) * 0.3,
                            leak_rgb[..., 2:3]
                        ]), 0)
        
        hsv_leak = r_boost + g_boost + b_boost
        
        zero_mask = (max_vals == 0)
        hsv_leak = np.where(zero_mask, leak_rgb, hsv_leak)
        
        enhanced_leak_rgb = np.clip(leak_rgb * 0.7 + hsv_leak * 0.3, 0, 255)
        
        adjusted_rgb = img_rgb * (1.0 - blend_factor) + result_rgb * blend_factor
        
        color_intensity = min(enhanced_opacity * 0.6, 0.6)
        color_blend = img_rgb * (1.0 - color_intensity) + enhanced_leak_rgb * color_intensity
        
        final_rgb = (adjusted_rgb * 0.6 + color_blend * 0.4)
        final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
        
        final_alpha = np.clip(img_alpha, 0, 255).astype(np.uint8)
        final_array = np.concatenate([final_rgb, final_alpha], axis=2)
        
        if progress_callback:
            progress_callback(100)
        result = Image.fromarray(final_array, mode="RGBA")
        return result.convert("RGB")
    
    @staticmethod
    def apply_custom_border(photo_img, border_img):
        photo_portrait = photo_img.height > photo_img.width
        border_portrait = border_img.height > border_img.width
        if photo_portrait != border_portrait:
            border_img = border_img.rotate(90, expand=True)
        photo_w, photo_h = photo_img.size
        border_w, border_h = border_img.size
        scale = max(border_w / photo_w, border_h / photo_h)
        new_photo_w = int(photo_w * scale)
        new_photo_h = int(photo_h * scale)
        resized_photo = photo_img.resize((new_photo_w, new_photo_h), Image.LANCZOS)
        offset_x = (new_photo_w - border_w) // 2
        offset_y = (new_photo_h - border_h) // 2
        cropped_photo = resized_photo.crop((offset_x, offset_y, offset_x+border_w, offset_y+border_h))
        return Image.alpha_composite(cropped_photo.convert("RGBA"), border_img.convert("RGBA")).convert("RGB")

# ========= Main Application =========

class ModernPhotoEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("CineSuite")
        self.root.state('zoomed')
        self.root.minsize(1024,768)
        self.root.configure(bg="#0F0F0F")
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()

        # Default effect values
        self.halation_enabled = tk.BooleanVar(value=True)
        self.halation_intensity = tk.DoubleVar(value=2)
        self.halation_ranges = tk.DoubleVar(value=0.75)
        self.halation_blur = tk.DoubleVar(value=0.15)
        self.halation_strength = tk.DoubleVar(value=1.75)
        self.halation_color = tk.StringVar(value="#FF4655")

        self.grain_enabled = tk.BooleanVar(value=True)
        self.grain_intensity = tk.DoubleVar(value=0.5)
        self.grain_size = tk.DoubleVar(value=0.8)
        self.grain_chroma = tk.DoubleVar(value=0.3)

        self.border_enabled = tk.BooleanVar(value=False)
        self.border_color = tk.StringVar(value="#000000")
        self.border_thickness = tk.DoubleVar(value=5)
        self.border_framing = tk.StringVar(value="Original")
        self.custom_border_image = None

        self.light_leak_opacity = tk.DoubleVar(value=0.5)
        self.light_leak_enabled_var = tk.BooleanVar(value=False)

        self.fringe_enabled = tk.BooleanVar(value=True)
        self.fringe_intensity = tk.DoubleVar(value=0.5)

        self.vignette_enabled = tk.BooleanVar(value=True)
        self.vignette_intensity = tk.DoubleVar(value=0.9)
        self.vignette_radius = tk.DoubleVar(value=2)
        self.vignette_feather = tk.DoubleVar(value=1.4)

        self.dust_enabled = tk.BooleanVar(value=False)
        self.dust_amount = tk.DoubleVar(value=1.3)
        self.dust_size = tk.DoubleVar(value=0.6)
        self.dust_opacity = tk.DoubleVar(value=0.95)
        # ---------------------------------------------------------------------

        self.input_files = []
        self.current_image_index = 0
        self.original_images = {}  # Original images per index
        self.current_image = None   # Base image used for further processing
        self.current_display_image = None  # Image shown in preview (rotated)
        self.preview_image = None
        self.is_processing = False
        self.update_job = None
        self.individual_settings = {}  # Store per-image settings; here we keep the rotation count

        self.setup_ui()
        self.executor = ThreadPoolExecutor(max_workers=1)

    # ------- UI Setup -------
    def configure_styles(self):
        bg_color = "#0F0F0F"
        panel_bg = "#1A1A1A"
        accent = "#FF4655"
        text_color = "#FFFFFF"
        slider_trough = "#404040"
        self.style.configure('.', background=bg_color, foreground=text_color)
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('Panel.TFrame', background=panel_bg, relief=tk.FLAT, borderwidth=0)
        self.style.configure('Control.TFrame', background=panel_bg, relief=tk.FLAT, borderwidth=0)
        self.style.configure('Primary.TButton', background=accent, foreground=text_color,
                             borderwidth=0, focusthickness=0, font=('Helvetica', 10, 'bold'),
                             padding=10, relief='flat')
        self.style.map('Primary.TButton', background=[('active', '#E53E4B'), ('disabled', '#4A4A4A')],
                       foreground=[('disabled', '#7A7A7A')])
        self.style.configure('Rotate.TButton', background="#121212", foreground="#CCCCCC",
                             font=('Helvetica', 16, 'bold'), relief='flat', padding=8)
        self.style.map('Rotate.TButton', background=[('active', "#1F1F1F")])
        self.style.configure('Modern.Horizontal.TScale', background=panel_bg, troughcolor=slider_trough,
                             bordercolor=panel_bg, lightcolor=panel_bg, darkcolor=panel_bg,
                             sliderthickness=14, sliderrelief='flat', troughrelief='flat')
        self.style.configure('Modern.TCheckbutton', background=panel_bg, indicatorbackground=panel_bg,
                             indicatormargin=5, indicatordiameter=14, font=('Helvetica', 10))
        self.style.map('Modern.TCheckbutton', indicatorbackground=[('selected', accent), ('active', '#404040')],
                       background=[('active', panel_bg)])
        self.style.configure('Section.TLabel', font=('Helvetica', 14, 'bold'),
                             background=panel_bg, foreground=accent)
        self.style.configure("TCombobox",
                             fieldbackground=panel_bg,
                             background=panel_bg,
                             foreground=text_color)
        self.style.map("TCombobox",
                       fieldbackground=[('readonly', panel_bg)],
                       background=[('readonly', panel_bg)])
        self.style.configure("Red.Horizontal.TProgressbar", troughcolor=panel_bg, bordercolor=panel_bg,
                             background=accent, lightcolor=accent, darkcolor=accent)
        self.style.configure("Gray.Horizontal.TProgressbar", troughcolor=panel_bg, bordercolor=panel_bg,
                             background=panel_bg, lightcolor=panel_bg, darkcolor=panel_bg)
        self.style.configure("Vertical.TScrollbar", gripcount=0,
                             background=panel_bg, darkcolor=panel_bg, lightcolor=panel_bg,
                             troughcolor=bg_color, bordercolor=bg_color, arrowcolor=text_color)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        
        self.fixed_file_section = ttk.Frame(left_container, style='Panel.TFrame')
        self.fixed_file_section.pack(fill=tk.X)
        self.create_file_controls(self.fixed_file_section)
        
        scroll_holder = ttk.Frame(left_container)
        scroll_holder.pack(fill=tk.BOTH, expand=True)
        
        self.scroll_canvas = tk.Canvas(scroll_holder, bg="#1A1A1A", width=350, highlightthickness=0)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.left_scroll = ttk.Scrollbar(scroll_holder, orient=tk.VERTICAL, command=self.scroll_canvas.yview, style="Vertical.TScrollbar")
        self.left_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_canvas.configure(yscrollcommand=self.left_scroll.set)
        
        self.control_panel = ttk.Frame(self.scroll_canvas, style='Panel.TFrame')
        self.scroll_canvas.create_window((0,0), window=self.control_panel, anchor="nw", width=350)
        self.control_panel.bind("<Configure>", lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")))
        
        self.create_halation_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_grain_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_border_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_light_leak_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_fringe_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_vignette_controls(self.control_panel)
        self.add_separator(self.control_panel)
        self.create_dust_controls(self.control_panel)
        
        preview_panel = ttk.Frame(main_frame, style='Panel.TFrame')
        preview_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(preview_panel, bg="#121212", bd=0, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas_text = self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, 
                                                    text="Upload photos to begin", fill="#404040",
                                                    font=('Helvetica', 14), anchor="center")
        self.rotate_btn = ttk.Button(preview_panel, text="⟳", style='Rotate.TButton', command=self.rotate_image)
        self.rotate_btn.place(relx=0.98, rely=0.98, anchor="se")
        
        self.progress_bar = ttk.Progressbar(preview_panel, orient=tk.HORIZONTAL, mode="determinate",
                                            style="Gray.Horizontal.TProgressbar")
        self.progress_bar.place(relx=0.5, rely=0.95, anchor="center", relwidth=0.2)
        self.progress_bar.lower()

    def add_separator(self, parent):
        sep = ttk.Separator(parent, orient='horizontal')
        sep.pack(fill=tk.X, padx=20, pady=10)
    
    def create_file_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        title = ttk.Label(frame, text="Files", style='Section.TLabel')
        title.pack(anchor="w")
        btn = ttk.Button(frame, text="Upload Photos", style='Primary.TButton', command=self.open_files)
        btn.pack(fill=tk.X, pady=5)
        nav_frame = ttk.Frame(frame, style='Control.TFrame')
        nav_frame.pack(fill=tk.X, pady=5)
        self.prev_btn = ttk.Button(nav_frame, text="◀", style='Primary.TButton', command=self.prev_image,
                                    state=tk.DISABLED, width=3)
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        self.next_btn = ttk.Button(nav_frame, text="▶", style='Primary.TButton', command=self.next_image,
                                    state=tk.DISABLED, width=3)
        self.next_btn.pack(side=tk.RIGHT, padx=2)
        self.image_counter = ttk.Label(nav_frame, text="0/0", style='TLabel')
        self.image_counter.pack(pady=5)
        export_btn = ttk.Button(frame, text="Export Processed Photos", style='Primary.TButton', command=self.export_photos)
        export_btn.pack(fill=tk.X, pady=5)
    
    def create_halation_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Halation Effect", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.halation_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        self._create_slider(frame, "Intensity:", self.halation_intensity, 0.1, 3.0)
        self._create_slider(frame, "Highlight Ranges:", self.halation_ranges, 0.3, 1.2, individual_key='halation_ranges')
        self._create_slider(frame, "Glow Blur:", self.halation_blur, 0.0, 1.2)
        self._create_slider(frame, "Effect Strength:", self.halation_strength, 0.5, 2.0)
        color_frame = ttk.Frame(frame, style='Control.TFrame')
        color_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(color_frame, text="Halation Color:", style='TLabel').pack(side=tk.LEFT)
        color_btn = ttk.Button(color_frame, text="Pick", style='Primary.TButton', command=self.choose_halation_color)
        color_btn.pack(side=tk.RIGHT)
    
    def create_grain_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Film Grain", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.grain_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        self._create_slider(frame, "Grain Intensity:", self.grain_intensity, 0.1, 1.0)
        self._create_slider(frame, "Grain Size:", self.grain_size, 0.1, 1.0)
        self._create_slider(frame, "Grain Chroma:", self.grain_chroma, 0.0, 1.0)
    
    def create_border_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Film Border", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.border_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        color_frame = ttk.Frame(frame, style='Control.TFrame')
        color_frame.pack(fill=tk.X, pady=(5,0))
        ttk.Label(color_frame, text="Border Color:", style='TLabel').pack(side=tk.LEFT)
        border_color_btn = ttk.Button(color_frame, text="Pick", style='Primary.TButton', command=self.choose_border_color)
        border_color_btn.pack(side=tk.RIGHT)
        self._create_slider(frame, "Border Thickness (%):", self.border_thickness, 0, 35)
        aspect_options = [
            "Original",
            "4:5", "5:4",
            "16:9", "9:16",
            "3:4", "4:3",
            "1:1",
            "2:3", "3:2",
            "2:1", "1:2",
            "16:10", "10:16",
            "21:9", "9:21",
            "Custom"
        ]
        self.border_framing.set(aspect_options[0])
        combo = ttk.Combobox(frame, textvariable=self.border_framing, values=aspect_options, state="readonly")
        combo.pack(fill=tk.X, pady=5)
        combo.bind("<<ComboboxSelected>>", lambda e: self.toggle_custom_border())
        self.custom_border_btn = ttk.Button(frame, text="Upload Custom Border", style='Primary.TButton', command=self.upload_custom_border)
        self.custom_border_btn.pack(fill=tk.X, pady=5)
        self.custom_border_btn.pack_forget()
    
    def toggle_custom_border(self):
        if self.border_framing.get() == "Custom":
            self.custom_border_btn.pack(fill=tk.X, pady=5)
        else:
            self.custom_border_btn.pack_forget()
        self.schedule_update()
    
    def create_light_leak_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Light Leaks", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.light_leak_enabled_var,
                              style='Modern.TCheckbutton', command=self.update_light_leak_setting)
        chk.pack(side=tk.RIGHT)
        leak_frame = ttk.Frame(frame, style='Control.TFrame')
        leak_frame.pack(fill=tk.X, pady=5)
        ttk.Label(leak_frame, text="Pick PNG/AVIF Overlay:", style='TLabel').pack(side=tk.LEFT)
        leak_btn = ttk.Button(leak_frame, text="Pick", style='Primary.TButton', command=self.upload_light_leak)
        leak_btn.pack(side=tk.RIGHT)
        self._create_slider(frame, "Leak Opacity:", self.light_leak_opacity, 0.0, 1.0, individual_key='light_leak_opacity')
    
    def create_fringe_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Color Fringing", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.fringe_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        self._create_slider(frame, "Fringing Intensity:", self.fringe_intensity, 0.0, 1.0)
    
    def create_vignette_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Vignetting", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.vignette_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        self._create_slider(frame, "Vignette Intensity:", self.vignette_intensity, 0.0, 3.0)
        self._create_slider(frame, "Vignette Radius:", self.vignette_radius, 0.5, 4.0)
        self._create_slider(frame, "Vignette Feather:", self.vignette_feather, 0.0, 2.0)
    
    def create_dust_controls(self, parent):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=10, padx=20)
        header = ttk.Frame(frame, style='Control.TFrame')
        header.pack(fill=tk.X)
        ttk.Label(header, text="Dust", style='Section.TLabel').pack(side=tk.LEFT)
        chk = ttk.Checkbutton(header, text="", variable=self.dust_enabled,
                              style='Modern.TCheckbutton', command=self.schedule_update)
        chk.pack(side=tk.RIGHT)
        self._create_slider(frame, "Dust Amount:", self.dust_amount, 0.0, 2.0)
        self._create_slider(frame, "Dust Size:", self.dust_size, 0.1, 2.5)
        self._create_slider(frame, "Dust Opacity:", self.dust_opacity, 0.1, 1.0)
    
    def _create_slider(self, parent, label_text, variable, min_val, max_val, individual_key=None):
        frame = ttk.Frame(parent, style='Control.TFrame')
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label_text, style='TLabel').pack(side=tk.LEFT)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, variable=variable, orient=tk.HORIZONTAL,
                           style='Modern.Horizontal.TScale',
                           command=lambda val, var=variable, key=individual_key, fr=frame: self._slider_callback(var, key, fr))
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        val_label = ttk.Label(frame, text=f"{variable.get():.2f}", style='TLabel')
        val_label.pack(side=tk.RIGHT)
        frame.value_label = val_label
    
    def _slider_callback(self, variable, individual_key, frame):
        frame.value_label.config(text=f"{variable.get():.2f}")
        idx = self.current_image_index
        if individual_key:
            if idx not in self.individual_settings:
                self.individual_settings[idx] = { 'rotation': 0, individual_key: variable.get() }
            else:
                self.individual_settings[idx][individual_key] = variable.get()
        self.schedule_update()
    
    def choose_halation_color(self):
        color = colorchooser.askcolor(initialcolor=self.halation_color.get())
        if color[1]:
            self.halation_color.set(color[1])
            self.schedule_update()
    
    def choose_border_color(self):
        color = colorchooser.askcolor(initialcolor=self.border_color.get())
        if color[1]:
            self.border_color.set(color[1])
            self.schedule_update()
    
    def upload_custom_border(self):
        path = filedialog.askopenfilename(title="Select Custom Border", filetypes=[("PNG/AVIF files", "*.png *.avif")])
        if path:
            self.custom_border_image = Image.open(path).convert("RGBA")
            self.border_framing.set("Custom")
            self.toggle_custom_border()
            self.schedule_update()
    
    def upload_light_leak(self):
        path = filedialog.askopenfilename(title="Select Light Leak", filetypes=[("PNG/AVIF files", "*.png *.avif")])
        if path:
            leak_img = Image.open(path).convert("RGBA")
            idx = self.current_image_index
            self.individual_settings.setdefault(idx, {})['light_leak'] = leak_img
            self.schedule_update()
    
    def update_light_leak_setting(self):
        idx = self.current_image_index
        if idx not in self.individual_settings:
            self.individual_settings[idx] = {}
        self.individual_settings[idx]['light_leak_enabled'] = self.light_leak_enabled_var.get()
        self.schedule_update()
    
    def open_files(self):
        files = filedialog.askopenfilenames(title="Select Photos",
                                            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp")])
        if not files:
            return
        self.input_files = list(files)
        self.current_image_index = 0
        self.image_counter.config(text=f"1/{len(self.input_files)}")
        self.update_nav_buttons()
        for idx in range(len(self.input_files)):
            if idx not in self.individual_settings:
                self.individual_settings[idx] = {'rotation': 0,
                                                 'halation_ranges': self.halation_ranges.get(),
                                                 'light_leak_opacity': self.light_leak_opacity.get(),
                                                 'light_leak_enabled': False}
        self.load_current_image()
    
    def load_current_image(self):
        if not self.input_files:
            return
        self.start_progress()
        try:
            img_path = self.input_files[self.current_image_index]
            original = Image.open(img_path)
            self.original_images[self.current_image_index] = original.copy()
            idx = self.current_image_index
            rotation = self.individual_settings[idx].get('rotation', 0)
            # Quickly rotate the original by the stored rotation
            rotated = original.rotate(-rotation, expand=True) if rotation else original.copy()
            self.current_image = rotated.copy()
            self.current_display_image = rotated.copy()
            self.update_preview_image()
            self.halation_ranges.set(self.individual_settings[idx].get('halation_ranges', self.halation_ranges.get()))
            self.light_leak_opacity.set(self.individual_settings[idx].get('light_leak_opacity', self.light_leak_opacity.get()))
            self.light_leak_enabled_var.set(self.individual_settings[idx].get('light_leak_enabled', False))
            self.process_preview()
        except Exception as e:
            self.stop_progress()
    
    def process_preview(self):
        if self.current_image is None or self.is_processing:
            return
        self.start_progress()
        self.is_processing = True

        total_effects = 0
        if self.halation_enabled.get() and self.current_image is not None:
            total_effects += 1
        if self.grain_enabled.get():
            total_effects += 1
        if self.dust_enabled.get():
            total_effects += 1
        if self.vignette_enabled.get():
            total_effects += 1
        idx = self.current_image_index
        leak = self.individual_settings.get(idx, {}).get('light_leak', None)
        light_leak_enabled = self.individual_settings.get(idx, {}).get('light_leak_enabled', False)
        if light_leak_enabled and leak is not None:
            total_effects += 1
        effect_index = [0]
        def cumulative_callback(sub_progress):
            overall = (effect_index[0] / total_effects)*100 + (sub_progress/100)*(100/total_effects)
            self.root.after(0, lambda: self.progress_bar.configure(value=overall))
        def next_effect():
            effect_index[0] += 1

        self.executor.submit(self._process_and_update, total_effects, cumulative_callback, next_effect)
    
    def _process_and_update(self, total_effects, cumulative_callback, next_effect):
        try:
            processed_img = self.current_image.copy()
            np_img = np.array(processed_img)
            if np_img.ndim == 3 and np_img.shape[2] == 3 and self.halation_enabled.get():
                idx = self.current_image_index
                current_params = (self.halation_intensity.get(), self.halation_blur.get(),
                                  self.halation_strength.get(), self.halation_color.get(),
                                  self.individual_settings[idx].get('halation_ranges', self.halation_ranges.get()))
                # Apply halation without caching
                halation_np_img = ImageEffects.apply_halation(
                        pil_to_cv2(processed_img), current_params[4],
                        self.halation_intensity.get(), self.halation_blur.get(),
                        self.halation_strength.get(), self.halation_color.get(),
                        progress_callback=cumulative_callback
                    )
                next_effect()
                np_img = cv2.cvtColor(halation_np_img, cv2.COLOR_BGR2RGB)
            processed_img = Image.fromarray(np_img)
            if self.grain_enabled.get():
                processed_img = ImageEffects.apply_grain(
                    processed_img, self.grain_intensity.get(),
                    self.grain_size.get(), self.grain_chroma.get(),
                    progress_callback=cumulative_callback
                )
                next_effect()
            if self.dust_enabled.get():
                idx = self.current_image_index
                # Use original image size for caching so that rotation doesn't affect the cache key
                orig_size = self.original_images[idx].size
                cache_key = (self.dust_amount.get(), self.dust_size.get(), self.dust_opacity.get(), orig_size)
                cached = self.individual_settings.get(idx, {}).get('dust_overlay')
                if cached is None or self.individual_settings[idx].get('dust_params') != cache_key:
                    # Generate dust overlay using the original (unrotated) image
                    overlay = ImageEffects.generate_dust_overlay(self.original_images[idx],
                                (self.dust_amount.get(), self.dust_size.get(), self.dust_opacity.get()),
                                progress_callback=cumulative_callback)
                    self.individual_settings.setdefault(idx, {})['dust_overlay'] = overlay
                    self.individual_settings[idx]['dust_params'] = cache_key
                else:
                    overlay = cached
                # Rotate the cached overlay to match the current rotation if needed
                rot = self.individual_settings[idx].get('rotation', 0)
                if rot != 0:
                    overlay = overlay.rotate(-rot, expand=True)
                next_effect()
                processed_img = Image.alpha_composite(processed_img.convert("RGBA"), overlay).convert("RGB")
            if self.fringe_enabled.get():
                processed_img = ImageEffects.apply_color_fringing(processed_img, self.fringe_intensity.get())
            if self.vignette_enabled.get():
                processed_img = ImageEffects.apply_vignette(
                    processed_img,
                    self.vignette_intensity.get(),
                    self.vignette_radius.get(),
                    self.vignette_feather.get(),
                    progress_callback=cumulative_callback
                )
                next_effect()
            if self.border_enabled.get():
                if self.border_framing.get() == "Custom" and self.custom_border_image:
                    processed_img = ImageEffects.apply_custom_border(processed_img, self.custom_border_image)
                else:
                    processed_img = ImageEffects.apply_border(processed_img, self.border_thickness.get(),
                                                              self.border_color.get(), self.border_framing.get())
            idx = self.current_image_index
            leak = self.individual_settings.get(idx, {}).get('light_leak', None)
            light_leak_enabled = self.individual_settings.get(idx, {}).get('light_leak_enabled', False)
            per_image_leak_opacity = self.individual_settings[idx].get('light_leak_opacity', self.light_leak_opacity.get())
            if light_leak_enabled and leak is not None:
                processed_img = ImageEffects.apply_light_leak(
                    processed_img, leak, per_image_leak_opacity,
                    progress_callback=cumulative_callback
                )
                next_effect()
            self.current_display_image = processed_img
            self.root.after(0, self._update_canvas)
        except Exception as e:
            pass
        finally:
            self.root.after(0, self._reset_processing)
            self.root.after(0, self.stop_progress)
    
    def _update_canvas(self):
        if self.current_display_image:
            self.update_preview_image()
    
    def _reset_processing(self):
        self.is_processing = False
    
    def update_preview_image(self):
        if self.current_display_image is None:
            self.canvas.delete("all")
            self.canvas.itemconfigure(self.canvas_text, text="Upload photos to begin")
            self.canvas.coords(self.canvas_text, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2)
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            return
        img_width, img_height = self.current_display_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        self.preview_image = self.current_display_image.resize((new_width, new_height), Image.LANCZOS)
        tk_image = ImageTk.PhotoImage(self.preview_image)
        self.canvas.delete("all")
        x_pos = (canvas_width - new_width) // 2
        y_pos = (canvas_height - new_height) // 2
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image
    
    def on_canvas_resize(self, event):
        if self.current_display_image is None:
            self.canvas.coords(self.canvas_text, self.canvas.winfo_width()/2, self.canvas.winfo_height()/2)
        else:
            self.update_preview_image()
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_counter.config(text=f"{self.current_image_index+1}/{len(self.input_files)}")
            self.update_nav_buttons()
            self.load_current_image()
    
    def next_image(self):
        if self.current_image_index < len(self.input_files) - 1:
            self.current_image_index += 1
            self.image_counter.config(text=f"{self.current_image_index+1}/{len(self.input_files)}")
            self.update_nav_buttons()
            self.load_current_image()
    
    def update_nav_buttons(self):
        self.prev_btn.config(state=tk.NORMAL if self.current_image_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_image_index < len(self.input_files) - 1 else tk.DISABLED)
    
    def rotate_image(self):
        idx = self.current_image_index
        # Increment rotation count by 90 degrees (modulo 360)
        self.individual_settings[idx]['rotation'] = (self.individual_settings[idx].get('rotation', 0) + 90) % 360
        # Rotate both the base and display images by 90 degrees clockwise
        self.current_image = self.current_image.rotate(-90, expand=True)
        self.current_display_image = self.current_display_image.rotate(-90, expand=True)
        self.update_preview_image()
    
    def schedule_update(self):
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(80, self.process_preview)
    
    def export_photos(self):
        if not self.input_files:
            return
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        self.start_progress(maximum=len(self.input_files))
        export_thread = threading.Thread(target=self._export_photos_thread, args=(export_dir,))
        export_thread.daemon = True
        export_thread.start()
    
    def _export_photos_thread(self, export_dir):
        try:
            for i, file_path in enumerate(self.input_files):
                original = Image.open(file_path)
                idx = i
                settings = self.individual_settings.get(idx, {'rotation': 0, 'halation_ranges': self.halation_ranges.get()})
                proc_img = original.rotate(-settings['rotation'], expand=True) if settings['rotation'] else original.copy()
                np_img = np.array(proc_img)
                if np_img.ndim == 3 and np_img.shape[2] == 3 and self.halation_enabled.get():
                    np_img = ImageEffects.apply_halation(
                        pil_to_cv2(proc_img), settings['halation_ranges'],
                        self.halation_intensity.get(), self.halation_blur.get(),
                        self.halation_strength.get(), self.halation_color.get()
                    )
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                proc_img = Image.fromarray(np_img)
                if self.grain_enabled.get():
                    proc_img = ImageEffects.apply_grain(proc_img, self.grain_intensity.get(),
                                                         self.grain_size.get(), self.grain_chroma.get())
                if self.dust_enabled.get():
                    idx = i
                    orig_size = self.original_images[idx].size
                    cache_key = (self.dust_amount.get(), self.dust_size.get(), self.dust_opacity.get(), orig_size)
                    cached = self.individual_settings.get(idx, {}).get('dust_overlay')
                    if cached is None or self.individual_settings[idx].get('dust_params') != cache_key:
                        overlay = ImageEffects.generate_dust_overlay(self.original_images[idx],
                                    (self.dust_amount.get(), self.dust_size.get(), self.dust_opacity.get()))
                        self.individual_settings.setdefault(idx, {})['dust_overlay'] = overlay
                        self.individual_settings[idx]['dust_params'] = cache_key
                    else:
                        overlay = cached
                    rot = self.individual_settings[idx].get('rotation', 0)
                    if rot != 0:
                        overlay = overlay.rotate(-rot, expand=True)
                    proc_img = Image.alpha_composite(proc_img.convert("RGBA"), overlay).convert("RGB")
                # Apply fringing effect (present in preview)
                if self.fringe_enabled.get():
                    proc_img = ImageEffects.apply_color_fringing(proc_img, self.fringe_intensity.get())
                # Apply vignette effect (present in preview)
                if self.vignette_enabled.get():
                    proc_img = ImageEffects.apply_vignette(proc_img, self.vignette_intensity.get(),
                                                          self.vignette_radius.get(), self.vignette_feather.get())
                # Apply border effect if enabled
                if self.border_enabled.get():
                    if self.border_framing.get() == "Custom" and self.custom_border_image:
                        proc_img = ImageEffects.apply_custom_border(proc_img, self.custom_border_image)
                    else:
                        proc_img = ImageEffects.apply_border(proc_img, self.border_thickness.get(),
                                                              self.border_color.get(), self.border_framing.get())
                idx = i
                leak = self.individual_settings.get(idx, {}).get('light_leak', None)
                light_leak_enabled = self.individual_settings.get(idx, {}).get('light_leak_enabled', False)
                if light_leak_enabled and leak is not None:
                    proc_img = ImageEffects.apply_light_leak(proc_img, leak, self.individual_settings[idx].get('light_leak_opacity', self.light_leak_opacity.get()))
                base_name = os.path.basename(file_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(export_dir, f"{name}_cinesuite{ext}")
                proc_img.save(output_path, quality=100)
                self.root.after(0, lambda val=i+1: self._update_progress(val))
            self.root.after(0, self._export_complete)
        except Exception as e:
            self.root.after(0, lambda e=e: self._export_error(str(e)))
    
    def _update_progress(self, value):
        self.progress_bar["value"] = value
    
    def _export_complete(self):
        self.stop_progress()
    
    def _export_error(self, error_msg):
        self.stop_progress()
    
    def start_progress(self, maximum=0):
        self.progress_bar.lift()
        if maximum > 0:
            self.progress_bar.configure(mode="determinate", maximum=maximum, value=0, style="Red.Horizontal.TProgressbar")
        else:
            self.progress_bar.configure(mode="determinate", maximum=100, style="Red.Horizontal.TProgressbar")
            self.progress_bar["value"] = 0
    
    def stop_progress(self):
        self.progress_bar["value"] = 0
        self.root.after(500, lambda: self.progress_bar.lower())

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernPhotoEditor(root)
    root.mainloop()
