# NLSQ GUI Assets

This directory contains visual assets for the NLSQ desktop application.

## Required Assets

### Application Icon

The application requires an icon in the following formats:

- **icon.png** - PNG format, recommended size 256x256 pixels or larger
- **icon.ico** - ICO format for Windows (multi-resolution: 16x16, 32x32, 48x48, 256x256)

The icon should visually represent curve fitting or scientific data analysis.
Consider using elements like:
- A curve through data points
- A graph or chart symbol
- Mathematical notation

### Generating Icons

You can generate icon files from the included SVG template:

```bash
# Using ImageMagick (install: apt install imagemagick)
convert icon.svg -resize 256x256 icon.png

# For Windows ICO with multiple resolutions
convert icon.svg -define icon:auto-resize=256,128,64,48,32,16 icon.ico

# Using Inkscape (install: apt install inkscape)
inkscape icon.svg --export-type=png --export-filename=icon.png -w 256 -h 256
```

## Optional Assets

### Splash Screen

- **splash.png** - Splash screen image displayed during application startup
- Recommended size: 600x400 pixels
- Should include the NLSQ logo and/or name

## Asset Guidelines

1. **Format**: Use PNG for transparency support, ICO for Windows compatibility
2. **Size**: Icons should be at least 256x256 for high-DPI displays
3. **Design**: Keep icons simple and recognizable at small sizes
4. **Colors**: Use colors consistent with the application theme
   - Primary: #1f77b4 (blue)
   - Secondary: #d62728 (red)
   - Background: transparent or white

## Current Status

The included `icon.svg` is a placeholder that can be customized or replaced
with a professionally designed icon for production use.
