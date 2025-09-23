from typing import List

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .utils import stack_to_rgb


class MplCanvas(FigureCanvas):
    def __init__(self, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)

    def show_image(self, img: np.ndarray, title: str = "", grayscale: bool = False, show_colorbar: bool = True, raw_img: np.ndarray = None, custom_min: float = None, custom_max: float = None):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

        im = None
        vmin, vmax = None, None

        if img.ndim == 2:
            if custom_min is not None and custom_max is not None:
                vmin, vmax = custom_min, custom_max
            else:
                vmin, vmax = np.min(img), np.max(img)

            if grayscale:
                im = self.ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                im = self.ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
        elif img.ndim == 3:
            if grayscale:
                if custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = self.ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
            else:
                if img.shape[-1] <= 3:
                    im = self.ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = self.ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
        else:
            self.ax.text(0.5, 0.5, "Unsupported image shape", ha="center", va="center")

        self.ax.set_title(title)
        self.ax.axis("off")

        if show_colorbar and im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
            cbar = self.fig.colorbar(im, ax=self.ax, shrink=0.8, aspect=20)
            cbar.set_ticks([vmin, vmax])
            cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])

        self.draw()

    def show_grid(self, images: List[np.ndarray], titles: List[str], grayscale: bool = False, raw_images: List[np.ndarray] = None, custom_min: float = None, custom_max: float = None, channel_names: List[str] = None, channel_scaling: dict = None, custom_scaling_enabled: bool = False):
        n_images = len(images)
        if n_images == 0:
            return

        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols

        self.fig.clear()
        
        # Store axes for synchronized zooming
        self.grid_axes = []

        for i, (img, title) in enumerate(zip(images, titles)):
            ax = self.fig.add_subplot(rows, cols, i + 1)
            self.grid_axes.append(ax)
            im = None
            vmin, vmax = None, None

            channel_min = None
            channel_max = None
            if custom_scaling_enabled and channel_names and i < len(channel_names) and channel_scaling:
                channel_name = channel_names[i]
                if channel_name in channel_scaling:
                    channel_min = channel_scaling[channel_name]['min']
                    channel_max = channel_scaling[channel_name]['max']

            if img.ndim == 2:
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img), np.max(img)

                if grayscale:
                    im = ax.imshow(img, interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)
                else:
                    im = ax.imshow(img, interpolation="nearest", cmap='viridis', vmin=vmin, vmax=vmax)
            elif img.ndim == 3 and not grayscale:
                if img.shape[-1] <= 3:
                    im = ax.imshow(stack_to_rgb(img), interpolation="nearest")
                else:
                    im = ax.imshow(stack_to_rgb(img[..., :3]), interpolation="nearest")
            else:
                if channel_min is not None and channel_max is not None:
                    vmin, vmax = channel_min, channel_max
                elif custom_min is not None and custom_max is not None:
                    vmin, vmax = custom_min, custom_max
                else:
                    vmin, vmax = np.min(img[..., 0]), np.max(img[..., 0])
                im = ax.imshow(img[..., 0], interpolation="nearest", cmap='gray', vmin=vmin, vmax=vmax)

            ax.set_title(title, fontsize=10)
            ax.axis("off")

            if im is not None and img.ndim == 2 and vmin is not None and vmax is not None:
                cbar = self.fig.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.set_ticks([vmin, vmax])
                cbar.set_ticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])

        # Set up synchronized zooming for grid view
        self._setup_synchronized_zoom()
        
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw()

    def _setup_synchronized_zoom(self):
        """Set up synchronized zooming for grid view."""
        if not hasattr(self, 'grid_axes') or len(self.grid_axes) <= 1:
            return
        
        # Disconnect any existing zoom callbacks to avoid duplicates
        for ax in self.grid_axes:
            if hasattr(ax, '_synchronized_zoom_x_callback'):
                ax.callbacks.disconnect(ax._synchronized_zoom_x_callback)
            if hasattr(ax, '_synchronized_zoom_y_callback'):
                ax.callbacks.disconnect(ax._synchronized_zoom_y_callback)
        
        # Create synchronized zoom callback
        def on_zoom(ax):
            # The callback receives the axes object that changed
            if ax not in self.grid_axes:
                return
            
            # Prevent recursive callbacks by checking if we're already in a callback
            if hasattr(self, '_in_zoom_callback') and self._in_zoom_callback:
                return
            
            # Set flag to prevent recursive callbacks
            self._in_zoom_callback = True
            
            try:
                # Get the new view limits from the zoomed axis
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                # Apply the same limits to all other axes
                for other_ax in self.grid_axes:
                    if other_ax != ax:
                        # Temporarily disconnect callbacks to prevent recursion
                        x_callback = getattr(other_ax, '_synchronized_zoom_x_callback', None)
                        y_callback = getattr(other_ax, '_synchronized_zoom_y_callback', None)
                        
                        if x_callback is not None:
                            other_ax.callbacks.disconnect(x_callback)
                        if y_callback is not None:
                            other_ax.callbacks.disconnect(y_callback)
                        
                        # Set the limits
                        other_ax.set_xlim(xlim)
                        other_ax.set_ylim(ylim)
                        
                        # Reconnect callbacks
                        if x_callback is not None:
                            other_ax._synchronized_zoom_x_callback = other_ax.callbacks.connect('xlim_changed', on_zoom)
                        if y_callback is not None:
                            other_ax._synchronized_zoom_y_callback = other_ax.callbacks.connect('ylim_changed', on_zoom)
                
                # Redraw the figure
                self.fig.canvas.draw_idle()
                
            finally:
                # Clear the flag
                self._in_zoom_callback = False
        
        # Connect the callback to all axes
        for ax in self.grid_axes:
            ax._synchronized_zoom_x_callback = ax.callbacks.connect('xlim_changed', on_zoom)
            ax._synchronized_zoom_y_callback = ax.callbacks.connect('ylim_changed', on_zoom)


