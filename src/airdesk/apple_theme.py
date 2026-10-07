"""Shared adaptive macOS glass styling for AirDesk's native utility panels."""

from __future__ import annotations


def make_glass_container(frame, corner_radius: float = 24.0):
    """Return (outer glass view, content view), with a pre-Tahoe fallback."""
    import AppKit

    content = AppKit.NSView.alloc().initWithFrame_(frame)
    glass_class = getattr(AppKit, "NSGlassEffectView", None)
    if glass_class is not None:
        try:
            glass = glass_class.alloc().initWithFrame_(frame)
            glass.setContentView_(content)
            glass.setCornerRadius_(corner_radius)
            return glass, content
        except Exception:
            # NSGlassEffectView is SDK/runtime dependent. The visual-effect
            # fallback preserves translucency on older supported macOS builds.
            pass

    glass = AppKit.NSVisualEffectView.alloc().initWithFrame_(frame)
    glass.setMaterial_(AppKit.NSVisualEffectMaterialPopover)
    glass.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
    glass.setState_(AppKit.NSVisualEffectStateActive)
    glass.setEmphasized_(True)
    glass.setWantsLayer_(True)
    glass.layer().setCornerRadius_(corner_radius)
    glass.layer().setMasksToBounds_(True)
    glass.layer().setBorderWidth_(0.75)
    glass.layer().setBorderColor_(
        AppKit.NSColor.separatorColor().colorWithAlphaComponent_(0.55).CGColor()
    )
    return glass, glass


def make_tinted_card(frame, corner_radius: float = 14.0, alpha: float = 0.12):
    """Create a quiet hierarchy card that sits inside, rather than over, glass."""
    import AppKit

    card = AppKit.NSView.alloc().initWithFrame_(frame)
    card.setWantsLayer_(True)
    card.layer().setCornerRadius_(corner_radius)
    card.layer().setBackgroundColor_(
        AppKit.NSColor.labelColor().colorWithAlphaComponent_(alpha).CGColor()
    )
    card.layer().setBorderWidth_(0.5)
    card.layer().setBorderColor_(
        AppKit.NSColor.separatorColor().colorWithAlphaComponent_(0.38).CGColor()
    )
    return card


def configure_floating_panel(panel) -> None:
    """Give a borderless panel the system floating-surface behavior."""
    import AppKit

    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(True)
    panel.setHidesOnDeactivate_(False)
    panel.setMovableByWindowBackground_(True)
