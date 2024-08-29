# This is a patch for CairoMakie to allow the font of a marker to be specified.
import CairoMakie
# First, allow markers that are tuples of (char, font) to be used and passed through to the backends.
CairoMakie.Makie.to_spritemarker(marker::Tuple{Char, CairoMakie.Makie.FTFont}) = marker
# Then, when this is encountered in CairoMakie, simply forward the font to the `char` drawing method.
function CairoMakie.draw_marker(ctx, marker::Tuple{Char, CairoMakie.Makie.FTFont}, pos, scale, strokecolor, strokewidth, marker_offset, rotation)
    CairoMakie.draw_marker(ctx, marker[1], marker[2], pos, scale, strokecolor, strokewidth, marker_offset, rotation) # just forward the font
end
